"""Independent, registration-free verifier for the FMB1 W04 final freeze.

This module intentionally does not import the W04 producer, ROS, Open3D, PCL,
or any registration package.  It validates the small final manifests, follows
their paths back to the actual raw/target/snapshot files, and hashes those
files itself.  Consequently a self-consistent but detached set of CSV files
does not qualify the final data set.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


BACKEND_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
FINAL_SCENES = (
    "FMB1_R01",
    "FMB1_R02",
    "FMB1_R03",
    "FMB1_W01",
    "FMB1_W03",
    "FMB1_W04",
)
RICH_SCENES = frozenset(("FMB1_R01", "FMB1_R02", "FMB1_R03"))
WEAK_SCENES = frozenset(("FMB1_W01", "FMB1_W03", "FMB1_W04"))
STATIONS = ("S01", "S02", "S03")
QUANTILES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
W04_PREFIXES = {
    "S01": "20260820_081749",
    "S02": "20260820_081954",
    "S03": "20260820_082207",
}
GEOMETRY_FIELDS = (
    "initial_correspondence_count",
    "initial_valid_normal_correspondence_count",
    "lambda_min_trans",
    "lambda_mid_trans",
    "lambda_max_trans",
    "normalized_lambda_min_trans",
    "normalized_lambda_mid_trans",
    "normalized_lambda_max_trans",
    "condition_number_trans",
    "spectral_entropy_trans",
)
GEOMETRY_METADATA_FIELDS = frozenset(
    ("scene_id", "station_id", "snapshot_id", "selection_index")
)
REQUIRED_FILES = (
    "final_dataset_manifest.json",
    "final_scene_registry.yaml",
    "final_station_registry.yaml",
    "final_raw_bag_manifest.csv",
    "final_target_manifest.csv",
    "final_snapshot_manifest.csv",
    "final_geometry_manifest.csv",
    "rejected_candidate_manifest.csv",
    "replacement_lineage.json",
    "final_dataset_readiness.json",
    "NO_ICP_ATTESTATION.json",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_RESULT_FRAGMENTS = (
    "t_est",
    "estimated_transform",
    "final_transform",
    "icp_displacement",
    "final_residual",
    "fitness",
    "turnover",
    "capture_radius",
    "backend_result",
    "solver_result",
    "registration_result",
)
ZERO_COUNTER_KEYS = (
    "open3d_registration_call_count",
    "pcl_cli_invocation_count",
    "other_registration_process_count",
    "formal_trial_count",
    "actual_open3d_trials",
    "actual_pcl_trials",
    "actual_formal_trials",
    "actual_registration_trials",
    "actual_trials",
    "registration_execution_count",
)
FALSE_AUTHORITY_KEYS = (
    "FORMAL_LOCK_ISSUED",
    "FORMAL_ICP_UNLOCKED",
    "FORMAL_REGISTRATION_AUTHORIZED",
    "MEASUREMENT_FINAL_RESULT",
    "FORMAL_MEASUREMENT_RESULT",
    "PROPOSED_AMENDMENT_ACTIVE",
)


class W04FinalVerificationError(RuntimeError):
    """A final-freeze invariant is absent, ambiguous, or false."""


def _fail(message: str) -> None:
    raise W04FinalVerificationError(f"FMB1_W04_FINAL_VERIFY_FAIL: {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSON-form mapping {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path.name} must contain a mapping")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                _fail(f"{path.name} has no header")
            rows = list(reader)
    except OSError as exc:
        _fail(f"cannot read {path}: {exc}")
    if any(None in row for row in rows):
        _fail(f"{path.name} contains fields beyond its header")
    return rows


def _rows_from_mapping(value: Mapping[str, Any], key: str, label: str) -> list[Mapping[str, Any]]:
    rows = value.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        _fail(f"{label}.{key} must be a list of mappings")
    return list(rows)


def _text(row: Mapping[str, Any], names: Sequence[str], label: str) -> str:
    present = [row[name] for name in names if name in row and row[name] not in (None, "")]
    if not present or any(value != present[0] for value in present[1:]):
        _fail(f"{label} is missing or has conflicting aliases")
    value = present[0]
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be non-empty text")
    return value


def _bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    _fail(f"{label} must be boolean")
    raise AssertionError("unreachable")


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        _fail(f"{label} must be an integer")
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text) is None:
            _fail(f"{label} must be an integer")
        number = int(text.split(".", 1)[0])
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        _fail(f"{label} must be an integer")
    if number < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return number


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        _fail(f"{label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(f"{label} must be finite numeric")
    if not math.isfinite(number):
        _fail(f"{label} must be finite numeric")
    return number


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _station(row: Mapping[str, Any], scene_id: str, label: str) -> str:
    value = _text(row, ("station_id",), f"{label}.station_id")
    prefix = f"{scene_id}_"
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if value not in STATIONS:
        _fail(f"{label}.station_id is invalid: {value}")
    return value


def _scene_station(row: Mapping[str, Any], label: str) -> tuple[str, str]:
    scene = _text(row, ("scene_id",), f"{label}.scene_id")
    if scene not in FINAL_SCENES:
        _fail(f"{label} has non-final scene: {scene}")
    return scene, _station(row, scene, label)


def _resolve_repository_file(repository: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repository / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository)
    except (OSError, ValueError) as exc:
        _fail(f"{label} is missing or escapes repository: {candidate}: {exc}")
    if not resolved.is_file() or resolved.is_symlink():
        _fail(f"{label} must be a regular non-symlink file: {resolved}")
    return resolved


def _verify_file_binding(
    repository: Path,
    row: Mapping[str, Any],
    *,
    path_names: Sequence[str],
    sha_names: Sequence[str],
    size_names: Sequence[str] = (),
    label: str,
) -> tuple[Path, str]:
    path = _resolve_repository_file(repository, _text(row, path_names, f"{label}.path"), label)
    expected = _sha(_text(row, sha_names, f"{label}.sha256"), f"{label}.sha256")
    actual = _sha256(path)
    if actual != expected:
        _fail(f"{label} actual SHA mismatch")
    if size_names:
        present = [row[name] for name in size_names if name in row and row[name] not in (None, "")]
        if not present:
            _fail(f"{label}.bytes is missing")
        if any(_int(value, f"{label}.bytes", 1) != path.stat().st_size for value in present):
            _fail(f"{label} actual size mismatch")
    return path, actual


def _assert_exact_keys(rows: Iterable[Mapping[str, Any]], key_fn: Any, expected: set[Any], label: str) -> dict[Any, Mapping[str, Any]]:
    indexed: dict[Any, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        key = key_fn(row, f"{label}[{index}]")
        if key in indexed:
            _fail(f"duplicate {label} key: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        _fail(f"{label} keys differ: missing={sorted(expected-set(indexed))}, extra={sorted(set(indexed)-expected)}")
    return indexed


def _recursive_firewall(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_RESULT_FRAGMENTS):
                _fail(f"registration-derived field is forbidden: {path}.{key}")
            if key in ZERO_COUNTER_KEYS and _int(child, f"{path}.{key}") != 0:
                _fail(f"registration counter is nonzero: {path}.{key}")
            if key in FALSE_AUTHORITY_KEYS and _bool(child, f"{path}.{key}") is not False:
                _fail(f"authority/lock flag must be false: {path}.{key}")
            _recursive_firewall(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _recursive_firewall(child, f"{path}[{index}]")


def _geometry_class(lambda_min: float, condition: float, entropy: float) -> str:
    if lambda_min >= 0.18 and condition <= 3.0 and entropy >= 0.90:
        return "RICH"
    if lambda_min <= 0.12 and condition >= 6.0 and entropy <= 0.80:
        return "WEAK"
    return "INTERMEDIATE"


def _verify_sha256sums(final_dir: Path) -> int:
    checksum_path = final_dir / "SHA256SUMS"
    if not checksum_path.is_file():
        _fail("SHA256SUMS is missing")
    rows: dict[str, str] = {}
    for line_no, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/][^\n]*)", line)
        if match is None:
            _fail(f"SHA256SUMS line {line_no} is malformed")
        digest, relative = match.groups()
        if relative in rows or relative == "SHA256SUMS":
            _fail(f"SHA256SUMS duplicate/self entry: {relative}")
        path = (final_dir / relative).resolve()
        try:
            path.relative_to(final_dir.resolve())
        except ValueError:
            _fail(f"SHA256SUMS entry escapes final directory: {relative}")
        if not path.is_file() or _sha256(path) != digest:
            _fail(f"SHA256SUMS mismatch: {relative}")
        rows[relative] = digest
    missing = sorted(set(REQUIRED_FILES) - set(rows))
    if missing:
        _fail(f"SHA256SUMS does not cover required artifacts: {missing}")
    # The verifier report may itself be added after the first successful
    # verification.  It is still fully rehashed above, but excluding that one
    # self-produced evidence file keeps the reported scientific-artifact count
    # stable across the required verify -> checksum refresh -> verify cycle.
    return len(rows) - int("independent_verification.json" in rows)


def verify_w04_final_dataset(
    repository: Path,
    final_dir: Path,
    *,
    verify_files: bool = True,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Verify a successful W04 replacement and final 3R+3W freeze.

    The success verifier is intentionally fail-closed: it is not applicable to
    an acquisition/geometry failure bundle and will reject readiness=false.
    """

    try:
        repository = Path(repository).resolve(strict=True)
        final_dir = Path(final_dir).resolve(strict=True)
    except OSError as exc:
        _fail(f"repository/final directory is missing: {exc}")
    if not final_dir.is_dir():
        _fail("final_dir must be a directory")
    for name in REQUIRED_FILES:
        if not (final_dir / name).is_file():
            _fail(f"required final artifact is missing: {name}")

    backend = repository / "frozen_assets/backend_parameter_contract.json"
    if not backend.is_file() or _sha256(backend) != BACKEND_CONTRACT_SHA256:
        _fail("protected backend parameter contract SHA changed")

    bundle = _read_mapping(final_dir / "final_dataset_manifest.json")
    scene_doc = _read_mapping(final_dir / "final_scene_registry.yaml")
    station_doc = _read_mapping(final_dir / "final_station_registry.yaml")
    lineage = _read_mapping(final_dir / "replacement_lineage.json")
    readiness = _read_mapping(final_dir / "final_dataset_readiness.json")
    no_icp = _read_mapping(final_dir / "NO_ICP_ATTESTATION.json")
    raw_rows = _read_csv(final_dir / "final_raw_bag_manifest.csv")
    target_rows = _read_csv(final_dir / "final_target_manifest.csv")
    snapshot_rows = _read_csv(final_dir / "final_snapshot_manifest.csv")
    geometry_rows = _read_csv(final_dir / "final_geometry_manifest.csv")
    rejected_rows = _read_csv(final_dir / "rejected_candidate_manifest.csv")
    for name, value in (
        ("final dataset bundle", bundle),
        ("scene registry", scene_doc),
        ("station registry", station_doc),
        ("replacement lineage", lineage),
        ("readiness", readiness),
        ("NO_ICP", no_icp),
    ):
        _recursive_firewall(value, name)
    for path, rows in (
        ("final_raw_bag_manifest.csv", raw_rows),
        ("final_target_manifest.csv", target_rows),
        ("final_snapshot_manifest.csv", snapshot_rows),
        ("final_geometry_manifest.csv", geometry_rows),
        ("rejected_candidate_manifest.csv", rejected_rows),
    ):
        _recursive_firewall(rows, path)

    if bundle.get("backend_parameter_contract_sha256") != BACKEND_CONTRACT_SHA256:
        _fail("final dataset bundle backend contract SHA changed")
    bundle_list_names = (
        "raw_candidate_bags",
        "final_raw_bags",
        "final_targets",
        "final_snapshots",
        "final_geometry_metrics",
        "final_geometry_scenes",
        "rejected_candidates",
    )
    bundle_rows: dict[str, list[Mapping[str, Any]]] = {
        key: _rows_from_mapping(bundle, key, "final dataset bundle")
        for key in bundle_list_names
    }
    for key, expected_count in (
        ("raw_candidate_bags", 42),
        ("final_raw_bags", 36),
        ("final_targets", 18),
        ("final_snapshots", 180),
        ("final_geometry_metrics", 180),
        ("final_geometry_scenes", 6),
        ("rejected_candidates", 6),
    ):
        if len(bundle_rows[key]) != expected_count:
            _fail(f"final dataset bundle {key} count must be {expected_count}")
    if bundle.get("final_scene_registry") != scene_doc:
        _fail("final scene registry is not byte-semantically bound to final bundle")
    if bundle.get("final_station_registry") != station_doc:
        _fail("final station registry is not byte-semantically bound to final bundle")
    if bundle.get("replacement_lineage") != lineage:
        _fail("replacement lineage is not bound to final bundle")
    if bundle.get("no_icp_attestation") != no_icp:
        _fail("NO_ICP attestation is not bound to final bundle")
    if bundle.get("readiness") != readiness:
        _fail("readiness is not bound to final bundle")

    expected_station_keys = {(scene, station) for scene in FINAL_SCENES for station in STATIONS}
    scene_rows = _rows_from_mapping(scene_doc, "scenes", "scene registry")
    scenes = _assert_exact_keys(
        scene_rows,
        lambda row, label: _text(row, ("scene_id",), f"{label}.scene_id"),
        set(FINAL_SCENES),
        "scenes",
    )
    class_counts: Counter[str] = Counter()
    for scene, row in scenes.items():
        expected_class = "RICH" if scene in RICH_SCENES else "WEAK"
        geometry = row.get("geometry", row)
        if not isinstance(geometry, Mapping):
            _fail(f"scene {scene} geometry must be a mapping")
        actual_class = _text(geometry, ("final_geometry_class", "geometry_class"), f"scene {scene} class")
        if actual_class != expected_class:
            _fail(f"scene {scene} class is {actual_class}, expected {expected_class}")
        admitted = geometry.get("admitted", row.get("admitted"))
        if admitted is not None and not _bool(admitted, f"scene {scene}.admitted"):
            _fail(f"scene {scene} is not admitted")
        class_counts[actual_class] += 1
    if class_counts != Counter({"RICH": 3, "WEAK": 3}):
        _fail(f"final scene class counts differ: {dict(class_counts)}")

    station_rows = _rows_from_mapping(station_doc, "stations", "station registry")
    stations = _assert_exact_keys(station_rows, _scene_station, expected_station_keys, "stations")
    for key, row in stations.items():
        status = _text(
            row,
            ("acquisition_status", "station_acquisition_status", "station_status", "status"),
            f"station {key}.status",
        )
        if status not in {"ACQUISITION_PASS", "PASS"}:
            _fail(f"station {key} acquisition did not pass")

    expected_bag_keys = {(scene, station, role) for scene, station in expected_station_keys for role in ("MAP", "QUERY")}

    def raw_key(row: Mapping[str, Any], label: str) -> tuple[str, str, str]:
        scene, station = _scene_station(row, label)
        role = _text(row, ("role",), f"{label}.role")
        if role not in {"MAP", "QUERY"}:
            _fail(f"{label}.role must be MAP or QUERY")
        return scene, station, role

    raw = _assert_exact_keys(raw_rows, raw_key, expected_bag_keys, "final raw bags")
    bundle_raw = _assert_exact_keys(
        bundle_rows["final_raw_bags"], raw_key, expected_bag_keys, "bundle final raw bags"
    )
    raw_shas: dict[tuple[str, str, str], str] = {}
    for key, row in raw.items():
        if key[0] == "FMB1_W02":
            _fail("W02 raw bag entered final manifest")
        filename = _text(row, ("raw_filename", "raw_name"), f"raw {key}.filename")
        if key[0] == "FMB1_W04":
            expected_prefix = W04_PREFIXES[key[1]]
            expected_suffix = "part1_20s.bag" if key[2] == "MAP" else "part2_15s.bag"
            if filename != f"mid360_{expected_prefix}_{expected_suffix}":
                _fail(f"W04 raw mapping changed for {key}")
        if verify_files:
            _, digest = _verify_file_binding(
                repository,
                row,
                path_names=("raw_absolute_path", "absolute_path", "raw_path"),
                sha_names=("sha256", "SHA256"),
                size_names=("bytes", "file_size"),
                label=f"raw bag {key}",
            )
        else:
            digest = _sha(_text(row, ("sha256", "SHA256"), f"raw {key}.sha"), f"raw {key}.sha")
        raw_shas[key] = digest
        if _sha(_text(bundle_raw[key], ("sha256", "SHA256"), f"bundle raw {key}.sha"), f"bundle raw {key}.sha") != digest:
            _fail(f"final raw CSV/bundle SHA binding changed for {key}")

    targets = _assert_exact_keys(target_rows, _scene_station, expected_station_keys, "targets")
    bundle_targets = _assert_exact_keys(
        bundle_rows["final_targets"], _scene_station, expected_station_keys, "bundle targets"
    )
    target_shas: dict[tuple[str, str], str] = {}
    for key, row in targets.items():
        map_sha = _sha(_text(row, ("map_bag_sha256", "map_sha256"), f"target {key}.map sha"), f"target {key}.map sha")
        if map_sha != raw_shas[(*key, "MAP")]:
            _fail(f"target {key} is not bound to its MAP bag")
        if _int(row.get("query_contribution_to_target", row.get("query_contribution", -1)), f"target {key}.query contribution") != 0:
            _fail(f"target {key} has QUERY contribution")
        roles = row.get("input_roles")
        if isinstance(roles, str):
            if roles.lstrip().startswith("["):
                try:
                    roles = json.loads(roles)
                except json.JSONDecodeError:
                    _fail(f"target {key} input_roles JSON is malformed")
            else:
                roles = [part for part in re.split(r"[;,]", roles) if part]
        if roles != ["MAP"]:
            _fail(f"target {key} input roles are not exactly MAP-only")
        for flag in ("registration_called", "odometry_called", "scan_matching_called"):
            if flag in row and _bool(row[flag], f"target {key}.{flag}") is not False:
                _fail(f"target {key}.{flag} must be false")
        construction = _text(row, ("construction",), f"target {key}.construction")
        if "NO_REGISTRATION" not in construction:
            _fail(f"target {key} construction lacks no-registration lineage")
        if verify_files:
            _, digest = _verify_file_binding(
                repository,
                row,
                path_names=("target_path", "target_npy_path"),
                sha_names=("target_sha256", "target_npy_sha256"),
                label=f"target {key}",
            )
        else:
            digest = _sha(_text(row, ("target_sha256", "target_npy_sha256"), f"target {key}.sha"), f"target {key}.sha")
        target_shas[key] = digest
        bundle_target = bundle_targets[key]
        if _sha(_text(bundle_target, ("target_sha256", "target_npy_sha256"), f"bundle target {key}.sha"), f"bundle target {key}.sha") != digest:
            _fail(f"target CSV/bundle SHA binding changed for {key}")
        if _sha(_text(bundle_target, ("map_bag_sha256", "map_sha256"), f"bundle target {key}.map sha"), f"bundle target {key}.map sha") != map_sha:
            _fail(f"target CSV/bundle MAP binding changed for {key}")

    snapshots: dict[str, Mapping[str, Any]] = {}
    by_station: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(snapshot_rows):
        key = _scene_station(row, f"snapshots[{index}]")
        snapshot_id = _text(row, ("snapshot_id",), f"snapshots[{index}].snapshot_id")
        if snapshot_id in snapshots or not snapshot_id.startswith(f"{key[0]}_{key[1]}_Q"):
            _fail(f"duplicate/unbound snapshot: {snapshot_id}")
        if key[0] == "FMB1_W02":
            _fail("W02 snapshot entered final manifest")
        if _sha(_text(row, ("query_bag_sha256",), f"snapshot {snapshot_id}.query sha"), f"snapshot {snapshot_id}.query sha") != raw_shas[(*key, "QUERY")]:
            _fail(f"snapshot {snapshot_id} is not bound to its QUERY bag")
        if _sha(_text(row, ("target_sha256", "target_npy_sha256"), f"snapshot {snapshot_id}.target sha"), f"snapshot {snapshot_id}.target sha") != target_shas[key]:
            _fail(f"snapshot {snapshot_id} is not bound to its target")
        if verify_files:
            _verify_file_binding(
                repository,
                row,
                path_names=("source_path", "source_npy_path"),
                sha_names=("source_sha256", "source_npy_sha256"),
                label=f"snapshot {snapshot_id}",
            )
        snapshots[snapshot_id] = row
        by_station[key].append(row)
    bundle_snapshots: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(bundle_rows["final_snapshots"]):
        snapshot_id = _text(row, ("snapshot_id",), f"bundle snapshot[{index}].id")
        if snapshot_id in bundle_snapshots:
            _fail(f"duplicate bundle snapshot: {snapshot_id}")
        bundle_snapshots[snapshot_id] = row
    if len(snapshots) != 180 or set(by_station) != expected_station_keys:
        _fail("final snapshot inventory must contain exactly all 18 stations and 180 unique snapshots")
    if set(bundle_snapshots) != set(snapshots):
        _fail("snapshot CSV/bundle IDs differ")
    for snapshot_id, row in snapshots.items():
        bundle_row = bundle_snapshots[snapshot_id]
        for names, label in (
            (("source_sha256", "source_npy_sha256"), "source"),
            (("target_sha256", "target_npy_sha256"), "target"),
            (("query_bag_sha256",), "query bag"),
        ):
            if _sha(_text(row, names, f"snapshot {snapshot_id}.{label}"), f"snapshot {snapshot_id}.{label}") != _sha(_text(bundle_row, names, f"bundle snapshot {snapshot_id}.{label}"), f"bundle snapshot {snapshot_id}.{label}"):
                _fail(f"snapshot CSV/bundle {label} binding changed: {snapshot_id}")
    for key, rows in by_station.items():
        ordered = sorted(rows, key=lambda row: _int(row.get("selection_index"), f"snapshot {key}.selection index"))
        if len(ordered) != 10 or [_int(row.get("selection_index"), "selection index") for row in ordered] != list(range(10)):
            _fail(f"station {key} does not contain selection indexes 0..9 exactly once")
        actual_quantiles = tuple(_float(row.get("quantile"), f"snapshot {key}.quantile") for row in ordered)
        if actual_quantiles != QUANTILES:
            _fail(f"station {key} quantiles changed")
    if sum(len(rows) for key, rows in by_station.items() if key[0] == "FMB1_W04") != 30:
        _fail("W04 must contribute exactly 30 final snapshots")

    allowed_geometry = GEOMETRY_METADATA_FIELDS | frozenset(GEOMETRY_FIELDS)
    if not geometry_rows or set(geometry_rows[0]) != allowed_geometry:
        _fail(f"geometry manifest fields must be exactly geometry-only whitelist: {sorted(allowed_geometry)}")
    geometry_by_scene: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    geometry_ids: set[str] = set()
    for index, row in enumerate(geometry_rows):
        if set(row) != allowed_geometry:
            _fail(f"geometry row {index} contains missing/extra fields")
        key = _scene_station(row, f"geometry[{index}]")
        snapshot_id = _text(row, ("snapshot_id",), f"geometry[{index}].snapshot_id")
        if snapshot_id in geometry_ids or snapshot_id not in snapshots:
            _fail(f"geometry snapshot is duplicate/orphan: {snapshot_id}")
        if _int(row["selection_index"], f"geometry {snapshot_id}.selection index") != _int(snapshots[snapshot_id]["selection_index"], f"snapshot {snapshot_id}.selection index"):
            _fail(f"geometry selection binding changed: {snapshot_id}")
        for field in GEOMETRY_FIELDS:
            _float(row[field], f"geometry {snapshot_id}.{field}")
        if _int(row["initial_correspondence_count"], "initial correspondence", 1) < _int(row["initial_valid_normal_correspondence_count"], "valid normal correspondence", 1):
            _fail(f"valid correspondence count exceeds initial count: {snapshot_id}")
        geometry_ids.add(snapshot_id)
        geometry_by_scene[key[0]].append(row)
    if geometry_ids != set(snapshots) or len(geometry_rows) != 180:
        _fail("geometry manifest must bind one row to every final snapshot")
    bundle_geometry: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(bundle_rows["final_geometry_metrics"]):
        if set(row) != allowed_geometry:
            _fail(f"bundle geometry row {index} violates geometry-only whitelist")
        snapshot_id = _text(row, ("snapshot_id",), f"bundle geometry[{index}].snapshot_id")
        if snapshot_id in bundle_geometry:
            _fail(f"duplicate bundle geometry snapshot: {snapshot_id}")
        bundle_geometry[snapshot_id] = row
    if set(bundle_geometry) != geometry_ids:
        _fail("geometry CSV/bundle snapshot IDs differ")
    geometry_csv_by_id = {str(row["snapshot_id"]): row for row in geometry_rows}
    for snapshot_id, bundle_row in bundle_geometry.items():
        csv_row = geometry_csv_by_id[snapshot_id]
        if _int(bundle_row["selection_index"], f"bundle geometry {snapshot_id}.selection") != _int(csv_row["selection_index"], f"geometry {snapshot_id}.selection"):
            _fail(f"geometry CSV/bundle selection differs: {snapshot_id}")
        for field in GEOMETRY_FIELDS:
            if _float(bundle_row[field], f"bundle geometry {snapshot_id}.{field}") != _float(csv_row[field], f"geometry {snapshot_id}.{field}"):
                _fail(f"geometry CSV/bundle metric differs: {snapshot_id}/{field}")

    recomputed_classes: dict[str, str] = {}
    recomputed_medians: dict[str, dict[str, float]] = {}
    for scene in FINAL_SCENES:
        rows = geometry_by_scene[scene]
        if len(rows) != 30 or len({_station(row, scene, "geometry") for row in rows}) != 3:
            _fail(f"scene {scene} geometry must contain 30 rows from 3 stations")
        lambda_min = median(_float(row["normalized_lambda_min_trans"], "normalized lambda min") for row in rows)
        condition = median(_float(row["condition_number_trans"], "condition number") for row in rows)
        entropy = median(_float(row["spectral_entropy_trans"], "spectral entropy") for row in rows)
        classification = _geometry_class(lambda_min, condition, entropy)
        expected = "RICH" if scene in RICH_SCENES else "WEAK"
        if classification != expected:
            _fail(f"scene {scene} recomputed class is {classification}, expected {expected}")
        recomputed_classes[scene] = classification
        recomputed_medians[scene] = {
            "median_normalized_lambda_min_trans": lambda_min,
            "median_condition_number_trans": condition,
            "median_spectral_entropy_trans": entropy,
        }
    bundle_geometry_scenes = {
        _text(row, ("scene_id",), "bundle geometry scene id"): row
        for row in bundle_rows["final_geometry_scenes"]
    }
    if set(bundle_geometry_scenes) != set(FINAL_SCENES):
        _fail("bundle geometry scene summaries differ from final scenes")
    for scene, medians in recomputed_medians.items():
        row = bundle_geometry_scenes[scene]
        expected_class = recomputed_classes[scene]
        if _text(row, ("final_geometry_class", "geometry_class"), f"bundle geometry scene {scene}.class") != expected_class:
            _fail(f"bundle geometry scene class differs for {scene}")
        for field, expected in medians.items():
            if _float(row.get(field), f"bundle geometry scene {scene}.{field}") != expected:
                _fail(f"bundle geometry scene median differs for {scene}/{field}")

    if not rejected_rows:
        _fail("rejected candidate manifest is empty")
    if any(_text(row, ("scene_id",), "rejected scene") != "FMB1_W02" for row in rejected_rows):
        _fail("rejected manifest contains a scene other than W02")
    rejected_bag_rows = [row for row in rejected_rows if row.get("role") in {"MAP", "QUERY"}]
    if rejected_bag_rows:
        if len(rejected_bag_rows) != 6:
            _fail("W02 rejected manifest must retain all six raw bags")
        rejected_keys: set[tuple[str, str]] = set()
        for index, row in enumerate(rejected_bag_rows):
            station = _station(row, "FMB1_W02", f"rejected[{index}]")
            role = _text(row, ("role",), f"rejected[{index}].role")
            rejected_keys.add((station, role))
            reason = _text(row, ("exclusion_reason", "rejection_reason"), f"rejected[{index}].reason")
            if reason != "GEOMETRY_ONLY_INELIGIBLE":
                _fail("W02 rejection reason changed")
            if verify_files:
                _verify_file_binding(
                    repository,
                    row,
                    path_names=("raw_absolute_path", "absolute_path", "raw_path"),
                    sha_names=("sha256", "SHA256"),
                    label=f"rejected W02 raw {station}/{role}",
                )
        if rejected_keys != {(station, role) for station in STATIONS for role in ("MAP", "QUERY")}:
            _fail("W02 rejected raw station/role coverage changed")
    else:
        row = rejected_rows[0]
        if _text(row, ("candidate_status", "status"), "W02 status") != "GEOMETRY_REJECTED":
            _fail("W02 candidate status changed")
        if _text(row, ("exclusion_reason", "rejection_reason"), "W02 reason") != "GEOMETRY_ONLY_INELIGIBLE":
            _fail("W02 rejection reason changed")

    def rejected_key(row: Mapping[str, Any], label: str) -> tuple[str, str]:
        if _text(row, ("scene_id",), f"{label}.scene") != "FMB1_W02":
            _fail(f"{label} is not W02")
        station = _station(row, "FMB1_W02", label)
        role = _text(row, ("role",), f"{label}.role")
        if role not in {"MAP", "QUERY"}:
            _fail(f"{label}.role is invalid")
        return station, role

    expected_rejected_keys = {(station, role) for station in STATIONS for role in ("MAP", "QUERY")}
    bundle_rejected = _assert_exact_keys(
        bundle_rows["rejected_candidates"], rejected_key, expected_rejected_keys, "bundle rejected W02"
    )
    csv_rejected = _assert_exact_keys(
        rejected_rows, rejected_key, expected_rejected_keys, "rejected W02 CSV"
    )
    for key in expected_rejected_keys:
        if _sha(_text(bundle_rejected[key], ("sha256", "SHA256"), f"bundle rejected {key}.sha"), f"bundle rejected {key}.sha") != _sha(_text(csv_rejected[key], ("sha256", "SHA256"), f"rejected {key}.sha"), f"rejected {key}.sha"):
            _fail(f"rejected W02 CSV/bundle SHA binding changed for {key}")

    candidate_keys: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(bundle_rows["raw_candidate_bags"]):
        scene = _text(row, ("scene_id",), f"candidate[{index}].scene")
        if scene == "FMB1_W02":
            station = _station(row, scene, f"candidate[{index}]")
        elif scene in FINAL_SCENES:
            station = _station(row, scene, f"candidate[{index}]")
        else:
            _fail(f"unexpected raw candidate scene: {scene}")
        role = _text(row, ("role",), f"candidate[{index}].role")
        key = (scene, station, role)
        if role not in {"MAP", "QUERY"} or key in candidate_keys:
            _fail(f"duplicate/invalid raw candidate key: {key}")
        candidate_keys[key] = row
    expected_candidate_keys = expected_bag_keys | {
        ("FMB1_W02", station, role)
        for station in STATIONS
        for role in ("MAP", "QUERY")
    }
    if set(candidate_keys) != expected_candidate_keys:
        _fail("raw candidate bag set is not exactly final 36 plus rejected W02 six")
    for key in expected_bag_keys:
        if _sha(_text(candidate_keys[key], ("sha256", "SHA256"), f"candidate {key}.sha"), f"candidate {key}.sha") != raw_shas[key]:
            _fail(f"raw candidate/final SHA differs for {key}")
    for station, role in expected_rejected_keys:
        key = ("FMB1_W02", station, role)
        if _sha(_text(candidate_keys[key], ("sha256", "SHA256"), f"candidate {key}.sha"), f"candidate {key}.sha") != _sha(_text(csv_rejected[(station, role)], ("sha256", "SHA256"), f"rejected {(station, role)}.sha"), f"rejected {(station, role)}.sha"):
            _fail(f"raw candidate/rejected SHA differs for {key}")

    if _text(lineage, ("rejected_scene_id", "rejected_candidate_scene_id", "source_scene_id"), "lineage rejected scene") != "FMB1_W02":
        _fail("replacement lineage does not originate at W02")
    if _text(lineage, ("replacement_scene_id",), "lineage replacement scene") != "FMB1_W04":
        _fail("replacement lineage does not end at W04")
    if _text(lineage, ("rejection_reason", "exclusion_reason"), "lineage reason") != "GEOMETRY_ONLY_INELIGIBLE":
        _fail("replacement lineage reason changed")
    if lineage.get("decision_before_any_icp", lineage.get("rejected_before_any_icp", lineage.get("W02_REJECTED_BEFORE_ANY_ICP"))) is not True or lineage.get("w04_admission_before_any_icp", lineage.get("w04_admitted_before_any_icp", lineage.get("admission_before_any_icp", lineage.get("W04_ADMISSION_OCCURRED_BEFORE_ANY_ICP")))) is not True:
        _fail("replacement/admission was not attested before ICP")
    for key in (
        "formal_trial_count_at_decision",
        "formal_trial_count_at_admission",
        "formal_trial_count_at_w04_admission",
    ):
        if key in lineage and _int(lineage[key], f"lineage.{key}") != 0:
            _fail(f"replacement lineage {key} must be zero")

    expected_readiness = {
        "FMB1_FINAL_DATASET_READY": True,
        "READY_FOR_ZERO_PERTURBATION_AMENDMENT_ACTIVATION": True,
        "FINAL_RICH_SCENE_COUNT": 3,
        "FINAL_WEAK_SCENE_COUNT": 3,
        "FINAL_SCENE_COUNT": 6,
        "FINAL_STATION_COUNT": 18,
        "FINAL_SNAPSHOT_COUNT": 180,
        "RAW_CANDIDATE_BAG_COUNT": 42,
        "FINAL_ADMITTED_BAG_COUNT": 36,
        "REJECTED_BAG_COUNT": 6,
        "W02_RETAINED": True,
        "W02_INCLUDED_IN_FINAL_SET": False,
        "FMB1_W04_ACQUISITION_PASS": True,
        "FMB1_W04_FINAL_GEOMETRY_CLASS": "WEAK",
        "FMB1_W04_ADMISSION_PASS": True,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
    }
    for key, expected in expected_readiness.items():
        if readiness.get(key) != expected:
            _fail(f"readiness field {key} differs: {readiness.get(key)!r}")
    if readiness.get("backend_parameter_contract_sha256", readiness.get("backend_contract_sha256")) != BACKEND_CONTRACT_SHA256:
        _fail("readiness backend contract SHA is missing/changed")

    for key in ("FORMAL_ICP_UNLOCKED", "FORMAL_REGISTRATION_AUTHORIZED"):
        if no_icp.get(key) is not False:
            _fail(f"NO_ICP {key} must be false")
    for key in ("open3d_registration_call_count", "pcl_cli_invocation_count", "other_registration_process_count", "formal_trial_count"):
        if _int(no_icp.get(key), f"NO_ICP.{key}") != 0:
            _fail(f"NO_ICP {key} must be zero")
    if no_icp.get("NO_ICP_ATTESTATION_PASS", no_icp.get("pass")) is not True:
        _fail("NO_ICP attestation does not pass")
    if no_icp.get("backend_parameter_contract_sha256", no_icp.get("backend_contract_sha256")) != BACKEND_CONTRACT_SHA256:
        _fail("NO_ICP backend contract SHA is missing/changed")

    proposed = repository / "experiments/mid360_formal_batch1/zero_perturbation_mainline_amendment_v1_1_PROPOSED.json"
    proposal = _read_mapping(proposed)
    if proposal.get("status") not in {"PROPOSED_NOT_ACTIVE", "PROPOSED"} or proposal.get("FORMAL_AUTHORITY") is not False:
        _fail("zero-perturbation v1.1 proposal was activated")
    forbidden_lock_names = (
        "formal_batch1_lock.json",
        "formal_execution_lock.json",
        "formal_registration_execution_lock.json",
    )
    for name in forbidden_lock_names:
        if (final_dir / name).exists():
            _fail(f"formal execution lock was issued: {name}")

    checksum_entry_count = _verify_sha256sums(final_dir) if verify_checksums else 0
    return {
        "schema": "mid360_fmb1_w04_final_independent_verification_v1",
        "status": "PASS",
        "pass": True,
        "failure_count": 0,
        "checks": {
            "final_scene_count": 6,
            "final_rich_scene_count": 3,
            "final_weak_scene_count": 3,
            "final_station_count": 18,
            "final_raw_bag_count": 36,
            "rejected_raw_bag_count": 6,
            "raw_candidate_bag_count": 42,
            "final_target_count": 18,
            "final_snapshot_count": 180,
            "w04_snapshot_count": 30,
            "w02_snapshot_count": 0,
            "final_geometry_row_count": 180,
            "query_contribution_total": 0,
            "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
            "actual_open3d_trials": 0,
            "actual_pcl_trials": 0,
            "actual_formal_trials": 0,
            "formal_lock_issued": False,
            "formal_registration_authorized": False,
            "zero_perturbation_amendment_active": False,
            "checksum_entry_count": checksum_entry_count,
        },
        "w04_recomputed_geometry": recomputed_medians["FMB1_W04"],
        "scene_geometry_classes": recomputed_classes,
        "registration_backend_imports_or_calls": 0,
    }
