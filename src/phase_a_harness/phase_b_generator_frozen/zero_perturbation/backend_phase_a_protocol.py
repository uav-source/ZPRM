"""Frozen Phase A protocol loader and RNG-free plan enumeration."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


PROTOCOL_RELATIVE = Path("configs/zero_perturbation/backend_phase_a_v1.yaml")
PROTOCOL_SHA256 = "d5e657862294a79c1d01fc349d6f668501d07d3ec965d2d8807171dc3b952b8c"
DOCUMENT_RELATIVE = Path("docs/zero_perturbation_backend_phase_a_protocol.md")
DOCUMENT_SHA256 = "acaf543dbd7a1ca517e1584187c9437b33be0fe12b83de4d9d472bffb00a9905"
PROTOCOL_LOCK_COMMIT = "9f15d044530cfcf74ac4d1a61400346bce004317"
PROTOCOL_LOCK_TAG = "archive/zero-perturbation-backend-phase-a-v1-protocol-lock"
EXPECTED_SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
EXPECTED_GEOMETRY_SEEDS = (1850310744, 1957656152, 1334931069)
EXPECTED_MEASUREMENT_SEEDS = (217775206, 1664898153)
EXPECTED_REPEATS = (0, 1, 2, 3, 4)
EXPECTED_BACKENDS = (
    "open3d_point_to_plane",
    "pcl_iterative_closest_point_with_normals",
)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_json_object(text: str, marker: str, start: int = 0) -> str:
    marker_index = text.index(marker, start)
    object_start = text.index("{", marker_index + len(marker))
    depth = 0
    in_string = False
    escaped = False
    for index in range(object_start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[object_start : index + 1]
    raise ValueError(f"unterminated JSON object after {marker!r}")


def development_seed_values(path: Path, expected_sha256: str) -> dict[str, dict[str, int]]:
    """Parse only labels.development without constructing a seed generator."""

    if file_sha256(path) != expected_sha256:
        raise ValueError("frozen seed schedule hash changed")
    text = path.read_text(encoding="utf-8")
    labels_start = text.index('"labels"')
    payload = _extract_json_object(text, '"development"', labels_start)
    value = json.loads(payload)
    if type(value) is not dict:
        raise ValueError("Development seed schedule is not a mapping")
    return {
        str(domain): {str(label): int(seed) for label, seed in labels.items()}
        for domain, labels in value.items()
    }


@dataclass(frozen=True)
class PlannedSnapshot:
    snapshot_id: str
    scene_variant: str
    geometry_seed_index: int
    geometry_seed_value: int
    measurement_seed_index: int
    measurement_seed_value: int
    repeat_index: int
    condition: str

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedTrial:
    snapshot_id: str
    scene_variant: str
    geometry_seed_index: int
    geometry_seed_value: int
    measurement_seed_index: int
    measurement_seed_value: int
    repeat_index: int
    condition: str
    backend: str
    planned_trial_id: str

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendPhaseAProtocol:
    root: Path
    data: Mapping[str, Any]
    source_sha256: str
    document_sha256: str
    development_seeds: Mapping[str, Mapping[str, int]]

    @property
    def scenes(self) -> tuple[str, ...]:
        return tuple(self.data["phase_a_matrix"]["scenes"])

    @property
    def geometry_seeds(self) -> tuple[int, ...]:
        return tuple(
            int(row["value"])
            for row in self.data["phase_a_matrix"]["development_geometry_seeds"]
        )

    @property
    def measurement_seeds(self) -> tuple[int, ...]:
        return tuple(
            int(row["value"])
            for row in self.data["phase_a_matrix"]["development_measurement_seeds"]
        )

    @property
    def repeats(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["phase_a_matrix"]["repeat_indices"])

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(self.data["formal_backends"]["allowed_in_order"])

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(self.data["phase_a_matrix"]["conditions"])

    def planned_snapshots(self) -> Iterator[PlannedSnapshot]:
        for scene in self.scenes:
            for geometry_index, geometry_seed in enumerate(self.geometry_seeds):
                for measurement_index, measurement_seed in enumerate(
                    self.measurement_seeds
                ):
                    for repeat_index in self.repeats:
                        snapshot_id = (
                            f"phase-a-v1/{scene}/{geometry_index}/"
                            f"{measurement_index}/{repeat_index}"
                        )
                        yield PlannedSnapshot(
                            snapshot_id=snapshot_id,
                            scene_variant=scene,
                            geometry_seed_index=geometry_index,
                            geometry_seed_value=geometry_seed,
                            measurement_seed_index=measurement_index,
                            measurement_seed_value=measurement_seed,
                            repeat_index=repeat_index,
                            condition="IDEAL_MATCHED",
                        )

    def planned_trials(self) -> Iterator[PlannedTrial]:
        for snapshot in self.planned_snapshots():
            for backend in self.backends:
                yield PlannedTrial(
                    **snapshot.row(),
                    backend=backend,
                    planned_trial_id=f"{snapshot.snapshot_id}/{backend}",
                )

    def self_audit(self) -> dict[str, Any]:
        snapshots = tuple(self.planned_snapshots())
        trials = tuple(self.planned_trials())
        backend_counts = Counter(trial.backend for trial in trials)
        return {
            "PROTOCOL_HAS_EXACTLY_7_SCENES": self.scenes == EXPECTED_SCENES,
            "PROTOCOL_HAS_EXACTLY_3_GEOMETRY_SEEDS": (
                self.geometry_seeds == EXPECTED_GEOMETRY_SEEDS
            ),
            "PROTOCOL_HAS_EXACTLY_2_MEASUREMENT_SEEDS": (
                self.measurement_seeds == EXPECTED_MEASUREMENT_SEEDS
            ),
            "PROTOCOL_HAS_EXACTLY_5_REPEATS": self.repeats == EXPECTED_REPEATS,
            "PROTOCOL_HAS_ONLY_IDEAL_MATCHED": self.conditions == ("IDEAL_MATCHED",),
            "PROTOCOL_HAS_EXACTLY_2_BACKENDS": self.backends == EXPECTED_BACKENDS,
            "PLANNED_SNAPSHOT_COUNT": len(snapshots),
            "PLANNED_TRIAL_COUNT": len(trials),
            "NATIVE_PLANNED_TRIAL_COUNT": sum(
                "native" in trial.backend.lower() for trial in trials
            ),
            "OPEN3D_PLANNED_TRIAL_COUNT": backend_counts[
                "open3d_point_to_plane"
            ],
            "PCL_PLANNED_TRIAL_COUNT": backend_counts[
                "pcl_iterative_closest_point_with_normals"
            ],
            "PHASE_A_RNG_INSTANTIATION_COUNT": 0,
            "PHASE_A_SNAPSHOT_GENERATION_COUNT": 0,
            "PHASE_A_BACKEND_EXECUTION_COUNT": 0,
            "PHASE_A_TRIAL_RESULT_COUNT": 0,
            "NEW_PROTOCOL_AMBIGUITIES_FOUND": False,
        }


def _validate(data: Mapping[str, Any], repository: Path) -> None:
    if data.get("schema_version") != "zero_perturbation_backend_phase_a_v1":
        raise ValueError("Phase A protocol schema changed")
    authority = data["protocol"]
    expected_false = (
        "scientific_claim_authorized",
        "phase_a_run_authorized_before_lock",
        "phase_b_authorized",
        "full_development_authorized",
        "confirmatory_authorized",
        "real_data_authorized",
        "measurement_paper_authorized",
    )
    if authority["protocol_type"] != "dual_independent_backend_phase_a_qualification":
        raise ValueError("Phase A protocol type changed")
    if authority["protocol_version"] != 1 or any(authority[key] for key in expected_false):
        raise ValueError("Phase A authority changed")
    for declaration in data["immutable_inputs"].values():
        path = repository / declaration["path"]
        if file_sha256(path) != declaration["sha256"]:
            raise ValueError(f"immutable Phase A input changed: {declaration['path']}")
    for section in ("open3d_parameter_contract", "pcl_parameter_contract"):
        if canonical_json_sha256(data[section]["parameters"]) != data[section][
            "canonical_sha256"
        ]:
            raise ValueError(f"parameter hash mismatch: {section}")


def load_backend_phase_a_protocol(root: str | Path) -> BackendPhaseAProtocol:
    repository = Path(root).resolve()
    source = repository / PROTOCOL_RELATIVE
    actual = file_sha256(source)
    if actual != PROTOCOL_SHA256:
        raise ValueError(f"Phase A protocol hash changed: {actual}")
    if file_sha256(repository / DOCUMENT_RELATIVE) != DOCUMENT_SHA256:
        raise ValueError("Phase A protocol document hash changed")
    try:
        data = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError("invalid Phase A protocol YAML") from error
    if type(data) is not dict:
        raise ValueError("Phase A protocol root must be a mapping")
    _validate(data, repository)
    seed_declaration = data["immutable_inputs"]["seed_schedule"]
    seeds = development_seed_values(
        repository / seed_declaration["path"], seed_declaration["sha256"]
    )
    protocol = BackendPhaseAProtocol(
        root=repository,
        data=data,
        source_sha256=actual,
        document_sha256=DOCUMENT_SHA256,
        development_seeds=seeds,
    )
    if protocol.geometry_seeds != tuple(seeds["geometry"].values()):
        raise ValueError("Phase A geometry seeds differ from Development schedule")
    if protocol.measurement_seeds != tuple(seeds["measurement"].values()):
        raise ValueError("Phase A measurement seeds differ from Development schedule")
    audit = protocol.self_audit()
    if not all(
        value is True
        for key, value in audit.items()
        if key.startswith("PROTOCOL_HAS_")
    ):
        raise ValueError("Phase A protocol cardinality audit failed")
    if audit["PLANNED_SNAPSHOT_COUNT"] != 210 or audit["PLANNED_TRIAL_COUNT"] != 420:
        raise ValueError("Phase A planned cardinality changed")
    return protocol


def validate_protocol_lock_document(path: str | Path, root: str | Path) -> dict[str, Any]:
    """Validate a lock before any future execution boundary is reachable."""

    lock_path = Path(path)
    if not lock_path.is_file():
        raise FileNotFoundError("--protocol-lock file does not exist")
    try:
        document = json.loads(lock_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Phase A protocol lock JSON") from error
    if type(document) is not dict or "lock_payload_sha256" not in document:
        raise ValueError("Phase A protocol lock payload hash is missing")
    stored = str(document["lock_payload_sha256"])
    payload = {key: value for key, value in document.items() if key != "lock_payload_sha256"}
    if canonical_json_sha256(payload) != stored:
        raise ValueError("Phase A protocol lock payload hash mismatch")
    protocol = load_backend_phase_a_protocol(root)
    if document.get("protocol_sha256") != protocol.source_sha256:
        raise ValueError("Phase A protocol lock references the wrong protocol hash")
    if document.get("BACKEND_PHASE_A_RUN_AUTHORIZED") is not True:
        raise PermissionError("Phase A run is not authorized by this lock")
    return document


__all__ = [
    "BackendPhaseAProtocol",
    "DOCUMENT_RELATIVE",
    "DOCUMENT_SHA256",
    "EXPECTED_BACKENDS",
    "EXPECTED_GEOMETRY_SEEDS",
    "EXPECTED_MEASUREMENT_SEEDS",
    "EXPECTED_REPEATS",
    "EXPECTED_SCENES",
    "PROTOCOL_LOCK_COMMIT",
    "PROTOCOL_LOCK_TAG",
    "PROTOCOL_RELATIVE",
    "PROTOCOL_SHA256",
    "PlannedSnapshot",
    "PlannedTrial",
    "canonical_json_sha256",
    "file_sha256",
    "load_backend_phase_a_protocol",
    "validate_protocol_lock_document",
]
