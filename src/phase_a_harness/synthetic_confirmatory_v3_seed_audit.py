"""Static provenance audit for declared Synthetic Confirmatory v3 seeds.

The audit is intentionally incapable of consuming a seed.  It reads text,
CSV, JSON, Git objects, and the preserved failure registries; it imports no
random-number, snapshot-construction, runner, or backend module.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import canonical_json_sha256, file_sha256, write_json
from .synthetic_confirmatory_v3_contract import (
    BOOTSTRAP_SEED,
    DERIVATION_PREREQUISITE_COMMIT,
    FORMAL_RUNTIME_ROOT,
    GEOMETRY_SEEDS,
    MANIFEST_RELATIVE,
    MEASUREMENT_SEEDS,
    NAMESPACE,
    PROTOCOL_DOCUMENT_RELATIVE,
    PROTOCOL_RELATIVE,
    RUNTIME_ARCHIVE_ROOT,
    RUNTIME_LIFECYCLE_BASELINE_COMMIT,
    SEED_SCHEDULE_RELATIVE,
    SNAPSHOT_PLAN_RELATIVE,
    SOURCE_REPOSITORY,
    TRIAL_PLAN_RELATIVE,
    V1_SEED_RETIREMENT_RELATIVE,
    V2_FAILURE_ARCHIVE_ROOT,
    V2_SEED_RETIREMENT_RELATIVE,
    canonical_identity_sha256,
    derive_seed,
)


PREWRITE_SCAN_SCHEMA = "synthetic_confirmatory_v3_prewrite_seed_collision_scan_v1"
AUDIT_SCHEMA = "synthetic_confirmatory_v3_static_seed_provenance_audit_v1"
DEVELOPMENT_PROTOCOL_RELATIVE = Path(
    "frozen_assets/full_synthetic_development_protocol_v1.json"
)
QUALIFICATION_RUNTIME_ROOT = Path(
    "/home/lj/zero_perturbation_runtime/qualification/"
    "synthetic_confirmatory_v3_prerun"
)

TEXT_SUFFIXES = frozenset(
    {".csv", ".json", ".md", ".ndjson", ".py", ".sh", ".txt", ".yaml", ".yml"}
)
TEXT_ROOTS = ("src", "scripts", "tests", "protocols", "frozen_assets")
GEOMETRY_OR_MEASUREMENT_LITERAL_PATHS = frozenset(
    {
        MANIFEST_RELATIVE.as_posix(),
        PROTOCOL_RELATIVE.as_posix(),
        SEED_SCHEDULE_RELATIVE.as_posix(),
        SNAPSHOT_PLAN_RELATIVE.as_posix(),
        TRIAL_PLAN_RELATIVE.as_posix(),
    }
)
BOOTSTRAP_LITERAL_PATHS = frozenset(
    {
        MANIFEST_RELATIVE.as_posix(),
        PROTOCOL_RELATIVE.as_posix(),
        SEED_SCHEDULE_RELATIVE.as_posix(),
    }
)
NAMESPACE_LITERAL_PATHS = frozenset(
    {
        MANIFEST_RELATIVE.as_posix(),
        PROTOCOL_DOCUMENT_RELATIVE.as_posix(),
        PROTOCOL_RELATIVE.as_posix(),
        SEED_SCHEDULE_RELATIVE.as_posix(),
    }
)


def _run(
    command: Sequence[str], *, cwd: Path, allowed_returncodes: Sequence[int] = (0,)
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode not in allowed_returncodes:
        raise RuntimeError(
            f"static seed audit command failed ({result.returncode}): "
            f"{' '.join(command)}\n{result.stderr}"
        )
    return result


def _git_grep_paths(
    repository: Path, revision: str, needle: str, pathspecs: Sequence[str] = ()
) -> list[str]:
    command = ["git", "grep", "-l", "-F", "-e", needle, revision, "--"]
    command.extend(pathspecs)
    result = _run(command, cwd=repository, allowed_returncodes=(0, 1))
    return sorted(line for line in result.stdout.splitlines() if line)


def _git_pickaxe_commits(repository: Path, revision: str, needle: str) -> list[str]:
    result = _run(
        ["git", "log", revision, "--format=%H", f"-S{needle}", "--"],
        cwd=repository,
    )
    return sorted(set(line for line in result.stdout.splitlines() if line))


def _rg_paths(
    root: Path, needle: str, *, numeric_token: bool, relative_to: Path | None = None
) -> list[str]:
    if not root.exists():
        return []
    command = [
        "rg",
        "--hidden",
        "--files-with-matches",
        "--fixed-strings",
        "--glob",
        "!.git/**",
        "--glob",
        "!*.bundle",
        "--glob",
        "!*.tar.gz",
    ]
    if numeric_token:
        command.append("--word-regexp")
    command.extend(["--", needle, str(root)])
    result = _run(command, cwd=root.parent, allowed_returncodes=(0, 1))
    base = root if relative_to is None else relative_to
    paths = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        candidate = Path(line).resolve()
        try:
            paths.append(candidate.relative_to(base.resolve()).as_posix())
        except ValueError:
            paths.append(str(candidate))
    return sorted(set(paths))


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"seed registry root must be an object: {path}")
    return value


def _historical_seed_registries(repository: Path) -> dict[str, frozenset[int]]:
    v1 = _strict_object(repository / V1_SEED_RETIREMENT_RELATIVE)
    v2 = _strict_object(repository / V2_SEED_RETIREMENT_RELATIVE)
    development = _strict_object(repository / DEVELOPMENT_PROTOCOL_RELATIVE)
    v1_values = frozenset(
        {
            *map(int, v1["old_geometry_seeds"]),
            *map(int, v1["old_measurement_seeds"]),
            int(v1["old_bootstrap_seed"]),
        }
    )
    v2_values = frozenset(
        {
            *map(int, v2["geometry_seeds"]),
            *map(int, v2["measurement_seeds"]),
            *map(int, v2["bootstrap_seeds"]),
        }
    )
    development_values = frozenset(
        {
            *map(int, development["geometry_seeds"]),
            *map(int, development["measurement_seeds"]),
            int(development["bootstrap"]["seed"]),
        }
    )
    return {"development": development_values, "v1": v1_values, "v2": v2_values}


def _external_collision_paths(
    values: Iterable[str], *, numeric_token: bool
) -> dict[str, list[str]]:
    roots = {
        "source_repository": SOURCE_REPOSITORY,
        "v2_failure_archive": V2_FAILURE_ARCHIVE_ROOT,
        "qualification_runtime": QUALIFICATION_RUNTIME_ROOT,
    }
    output: dict[str, list[str]] = {}
    for label, root in roots.items():
        hits = []
        for value in values:
            hits.extend(_rg_paths(root, value, numeric_token=numeric_token))
        output[label] = sorted(set(hits))
    return output


def scan_v3_prewrite_collisions(root: str | Path) -> dict[str, Any]:
    """Reproduce the collision scan against commits preceding declaration.

    The lifecycle baseline and adapter-fixture prerequisite are immutable Git
    objects, so this remains meaningful after the declaration assets are later
    committed.  Current v3 declaration files are intentionally not inputs.
    """

    repository = Path(root).resolve()
    formal_seeds = frozenset((*GEOMETRY_SEEDS, *MEASUREMENT_SEEDS, BOOTSTRAP_SEED))
    if len(formal_seeds) != 9:
        raise ValueError("v3 derivation did not produce nine unique declarations")
    registries = _historical_seed_registries(repository)

    baseline_namespace = _git_grep_paths(
        repository, RUNTIME_LIFECYCLE_BASELINE_COMMIT, NAMESPACE
    )
    prerequisite_namespace = _git_grep_paths(
        repository, DERIVATION_PREREQUISITE_COMMIT, NAMESPACE
    )
    baseline_seed_paths: dict[str, list[str]] = {}
    prerequisite_seed_paths: dict[str, list[str]] = {}
    old_test_paths: dict[str, list[str]] = {}
    for seed in sorted(formal_seeds):
        label = str(seed)
        baseline_seed_paths[label] = _git_grep_paths(
            repository, RUNTIME_LIFECYCLE_BASELINE_COMMIT, label
        )
        prerequisite_seed_paths[label] = _git_grep_paths(
            repository, DERIVATION_PREREQUISITE_COMMIT, label
        )
        old_test_paths[label] = _git_grep_paths(
            repository, DERIVATION_PREREQUISITE_COMMIT, label, ("tests",)
        )

    baseline_namespace_history = _git_pickaxe_commits(
        repository, RUNTIME_LIFECYCLE_BASELINE_COMMIT, NAMESPACE
    )
    prerequisite_namespace_history = _git_pickaxe_commits(
        repository, DERIVATION_PREREQUISITE_COMMIT, NAMESPACE
    )
    all_refs_namespace_history = _git_pickaxe_commits(repository, "--all", NAMESPACE)
    baseline_seed_history: dict[str, list[str]] = {}
    prerequisite_seed_history: dict[str, list[str]] = {}
    all_refs_seed_history: dict[str, list[str]] = {}
    for seed in sorted(formal_seeds):
        label = str(seed)
        baseline_seed_history[label] = _git_pickaxe_commits(
            repository, RUNTIME_LIFECYCLE_BASELINE_COMMIT, label
        )
        prerequisite_seed_history[label] = _git_pickaxe_commits(
            repository, DERIVATION_PREREQUISITE_COMMIT, label
        )
        all_refs_seed_history[label] = _git_pickaxe_commits(
            repository, "--all", label
        )

    source_namespace_history = (
        _git_pickaxe_commits(SOURCE_REPOSITORY, "--all", NAMESPACE)
        if (SOURCE_REPOSITORY / ".git").exists()
        else []
    )
    source_seed_history: dict[str, list[str]] = {}
    if (SOURCE_REPOSITORY / ".git").exists():
        for seed in sorted(formal_seeds):
            source_seed_history[str(seed)] = _git_pickaxe_commits(
                SOURCE_REPOSITORY, "--all", str(seed)
            )

    namespace_external = _external_collision_paths((NAMESPACE,), numeric_token=False)
    seed_external = _external_collision_paths(
        (str(seed) for seed in sorted(formal_seeds)), numeric_token=True
    )
    v1_collisions = sorted(formal_seeds & registries["v1"])
    v2_collisions = sorted(formal_seeds & registries["v2"])
    development_collisions = sorted(formal_seeds & registries["development"])
    old_test_collisions = sorted(
        int(seed) for seed, paths in old_test_paths.items() if paths
    )

    namespace_collision_count = (
        len(baseline_namespace)
        + len(prerequisite_namespace)
        + len(baseline_namespace_history)
        + len(prerequisite_namespace_history)
        + len(all_refs_namespace_history)
        + len(source_namespace_history)
        + sum(len(paths) for paths in namespace_external.values())
    )
    seed_context_collision_count = (
        sum(len(paths) for paths in baseline_seed_paths.values())
        + sum(len(paths) for paths in prerequisite_seed_paths.values())
        + sum(len(paths) for paths in baseline_seed_history.values())
        + sum(len(paths) for paths in prerequisite_seed_history.values())
        + sum(len(paths) for paths in all_refs_seed_history.values())
        + sum(len(paths) for paths in source_seed_history.values())
        + sum(len(paths) for paths in seed_external.values())
    )
    provenance_collision_count = (
        seed_context_collision_count
        + len(v1_collisions)
        + len(v2_collisions)
        + len(development_collisions)
        + len(old_test_collisions)
    )
    passed = namespace_collision_count == provenance_collision_count == 0
    return {
        "NEW_V3_NAMESPACE_COLLISION": namespace_collision_count != 0,
        "V3_SEED_COLLISION_WITH_DEVELOPMENT_COUNT": len(development_collisions),
        "V3_SEED_COLLISION_WITH_OLD_TEST_COUNT": len(old_test_collisions),
        "V3_SEED_COLLISION_WITH_V1_COUNT": len(v1_collisions),
        "V3_SEED_COLLISION_WITH_V2_COUNT": len(v2_collisions),
        "V3_SEED_PROVENANCE_COLLISION_COUNT": provenance_collision_count,
        "V3_PREWRITE_COLLISION_SCAN_PASS": passed,
        "archive_tar_binding": {
            "extracted_archive_path": str(V2_FAILURE_ARCHIVE_ROOT),
            "qualification_verification_path": (
                "artifacts/runtime_lifecycle_qualification_v1/"
                "v2_failure_archive_verification.json"
            ),
            "scan_uses_verified_extracted_tree": True,
        },
        "all_refs_namespace_history_hits": all_refs_namespace_history,
        "all_refs_seed_history_hits": all_refs_seed_history,
        "baseline_commit": RUNTIME_LIFECYCLE_BASELINE_COMMIT,
        "baseline_namespace_history_hits": baseline_namespace_history,
        "baseline_namespace_paths": baseline_namespace,
        "baseline_seed_history_hits": baseline_seed_history,
        "baseline_seed_paths": baseline_seed_paths,
        "development_collisions": development_collisions,
        "derivation_prerequisite_commit": DERIVATION_PREREQUISITE_COMMIT,
        "external_namespace_paths": namespace_external,
        "external_seed_paths": seed_external,
        "namespace_collision_count": namespace_collision_count,
        "old_test_collisions": old_test_collisions,
        "old_test_seed_paths": old_test_paths,
        "prerequisite_namespace_history_hits": prerequisite_namespace_history,
        "prerequisite_namespace_paths": prerequisite_namespace,
        "prerequisite_seed_history_hits": prerequisite_seed_history,
        "prerequisite_seed_paths": prerequisite_seed_paths,
        "scan_scope": [
            "runtime lifecycle baseline commit and ancestry",
            "adapter fixture prerequisite commit and ancestry",
            "source repository current tree and all Git refs",
            "verified extracted v2 failure archive",
            "v3 qualification runtime",
            "Development seed registry",
            "v1 retired Confirmatory seed registry",
            "v2 retired Confirmatory seed registry",
            "historical Test tree at the prerequisite commit",
        ],
        "schema_version": PREWRITE_SCAN_SCHEMA,
        "seed_context_collision_count": seed_context_collision_count,
        "source_namespace_history_hits": source_namespace_history,
        "source_seed_history_hits": source_seed_history,
        "v1_collisions": v1_collisions,
        "v2_collisions": v2_collisions,
    }


def _text_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for relative in TEXT_ROOTS
        for path in (root / relative).rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and "__pycache__" not in path.parts
    )


def _literal_paths(
    root: Path, needle: str, paths: Iterable[Path], *, numeric_token: bool
) -> list[str]:
    if numeric_token:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])"
        )
    else:
        pattern = re.compile(re.escape(needle))
    hits = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if pattern.search(text):
            hits.append(path.relative_to(root).as_posix())
    return sorted(hits)


def _plan_reference_counts(repository: Path) -> dict[str, int]:
    values = {"geometry": 0, "measurement": 0}
    for relative in (SNAPSHOT_PLAN_RELATIVE, TRIAL_PLAN_RELATIVE):
        with (repository / relative).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"seed plan header is missing: {relative}")
            for row in reader:
                geometry = int(row["geometry_seed"])
                if geometry not in GEOMETRY_SEEDS:
                    raise ValueError("plan references a non-v3 geometry seed")
                values["geometry"] += 1
                if row["measurement_seed"] not in (None, ""):
                    measurement = int(row["measurement_seed"])
                    if measurement not in MEASUREMENT_SEEDS:
                        raise ValueError("plan references a non-v3 measurement seed")
                    values["measurement"] += 1
    values["total"] = values["geometry"] + values["measurement"]
    return values


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        parts = [function.attr]
        value = function.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _test_constructor_hits(
    repository: Path, formal_seeds: frozenset[int]
) -> list[dict[str, Any]]:
    constructor_terms = (
        "backend",
        "build_snapshot",
        "execute",
        "generate",
        "random",
        "rng",
        "seedsequence",
    )
    seed_names = {
        "BOOTSTRAP_SEED",
        "GEOMETRY_SEEDS",
        "MEASUREMENT_SEEDS",
        "derive_seed",
    }
    hits: list[dict[str, Any]] = []
    for path in sorted((repository / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_source_names = {
                child.id for child in ast.walk(node) if isinstance(child, ast.Name)
            }
            literal_seeds = sorted(
                {
                    int(child.value)
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant)
                    and type(child.value) is int
                    and child.value in formal_seeds
                }
            )
            name = _call_name(node).lower()
            named_seed_use = bool(
                call_source_names & seed_names
                and any(term in name for term in constructor_terms)
            )
            if literal_seeds or named_seed_use:
                hits.append(
                    {
                        "call": _call_name(node),
                        "line": int(node.lineno),
                        "literal_seeds": literal_seeds,
                        "named_seed_use": named_seed_use,
                        "relative_path": path.relative_to(repository).as_posix(),
                    }
                )
    return hits


def audit_v3_seed_provenance(root: str | Path) -> dict[str, Any]:
    """Audit declaration, plan reference, collision, and zero-use evidence."""

    repository = Path(root).resolve()
    paths = _text_paths(repository)
    formal_seeds = frozenset((*GEOMETRY_SEEDS, *MEASUREMENT_SEEDS, BOOTSTRAP_SEED))
    if len(formal_seeds) != 9:
        raise ValueError("v3 schedule does not contain nine unique values")

    schedule = _strict_object(repository / SEED_SCHEDULE_RELATIVE)
    records = schedule.get("records")
    if type(records) is not list or len(records) != 9:
        raise ValueError("v3 seed schedule declaration inventory is not exact")
    actual_records = sorted(
        [
            {
                "domain": str(row["domain"]),
                "index": int(row["index"]),
                "seed": int(row["seed"]),
            }
            for row in records
            if type(row) is dict
        ],
        key=lambda row: (row["domain"], row["index"]),
    )
    expected_records = sorted(
        [
            *(
                {"domain": "geometry", "index": index, "seed": derive_seed("geometry", index)}
                for index in range(5)
            ),
            *(
                {
                    "domain": "measurement",
                    "index": index,
                    "seed": derive_seed("measurement", index),
                }
                for index in range(3)
            ),
            {"domain": "bootstrap", "index": 0, "seed": derive_seed("bootstrap", 0)},
        ],
        key=lambda row: (row["domain"], row["index"]),
    )
    schedule_mismatch_count = int(actual_records != expected_records)
    derivation_mismatch_count = sum(
        int(row["seed"] != derive_seed(row["domain"], row["index"]))
        for row in actual_records
    )

    prewrite = schedule.get("prewrite_collision_scan")
    if type(prewrite) is not dict or prewrite.get("schema_version") != PREWRITE_SCAN_SCHEMA:
        raise ValueError("v3 schedule lacks the pre-write collision evidence")
    if prewrite.get("V3_PREWRITE_COLLISION_SCAN_PASS") is not True:
        raise ValueError("v3 pre-write collision scan did not pass")

    current_literal_hits = []
    current_unexpected = []
    for seed in sorted(formal_seeds):
        actual = _literal_paths(
            repository, str(seed), paths, numeric_token=True
        )
        expected = (
            BOOTSTRAP_LITERAL_PATHS
            if seed == BOOTSTRAP_SEED
            else GEOMETRY_OR_MEASUREMENT_LITERAL_PATHS
        )
        current_literal_hits.append({"paths": actual, "seed": seed})
        for relative in sorted(set(actual) ^ set(expected)):
            current_unexpected.append({"relative_path": relative, "seed": seed})

    namespace_hits = _literal_paths(
        repository, NAMESPACE, paths, numeric_token=False
    )
    namespace_unexpected = sorted(set(namespace_hits) ^ set(NAMESPACE_LITERAL_PATHS))
    constructor_hits = _test_constructor_hits(repository, formal_seeds)
    references = _plan_reference_counts(repository)
    runtime_exists = FORMAL_RUNTIME_ROOT.exists()
    prewrite_collision_count = int(prewrite.get("namespace_collision_count", -1)) + int(
        prewrite.get("V3_SEED_PROVENANCE_COLLISION_COUNT", -1)
    )
    unexpected_path_count = len(current_unexpected) + len(namespace_unexpected)
    zero_use = not runtime_exists and not constructor_hits
    passed = bool(
        schedule_mismatch_count == 0
        and derivation_mismatch_count == 0
        and prewrite_collision_count == 0
        and unexpected_path_count == 0
        and not constructor_hits
        and references == {"geometry": 1785, "measurement": 1575, "total": 3360}
        and zero_use
    )
    return {
        "NEW_V3_NAMESPACE_COLLISION": prewrite["NEW_V3_NAMESPACE_COLLISION"],
        "STATIC_V3_SEED_PROVENANCE_AUDIT_PASS": passed,
        "V3_BACKEND_EXECUTION_COUNT": 0,
        "V3_RNG_INSTANTIATION_COUNT": 0,
        "V3_SCIENTIFIC_RESULT_COUNT": 0,
        "V3_SEED_COLLISION_WITH_DEVELOPMENT_COUNT": prewrite[
            "V3_SEED_COLLISION_WITH_DEVELOPMENT_COUNT"
        ],
        "V3_SEED_COLLISION_WITH_OLD_TEST_COUNT": prewrite[
            "V3_SEED_COLLISION_WITH_OLD_TEST_COUNT"
        ],
        "V3_SEED_COLLISION_WITH_V1_COUNT": prewrite[
            "V3_SEED_COLLISION_WITH_V1_COUNT"
        ],
        "V3_SEED_COLLISION_WITH_V2_COUNT": prewrite[
            "V3_SEED_COLLISION_WITH_V2_COUNT"
        ],
        "V3_SEED_PROVENANCE_COLLISION_COUNT": prewrite[
            "V3_SEED_PROVENANCE_COLLISION_COUNT"
        ],
        "V3_SNAPSHOT_CONSTRUCTION_COUNT": 0,
        "V3_TRIAL_RESULT_COUNT": 0,
        "V3_UNEXPECTED_SEED_PATH_COUNT": unexpected_path_count,
        "allowed_seed_declaration_paths": sorted(
            {
                MANIFEST_RELATIVE.as_posix(),
                PROTOCOL_RELATIVE.as_posix(),
                SEED_SCHEDULE_RELATIVE.as_posix(),
            }
        ),
        "allowed_seed_plan_reference_paths": sorted(
            {SNAPSHOT_PLAN_RELATIVE.as_posix(), TRIAL_PLAN_RELATIVE.as_posix()}
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "current_seed_literal_hits": current_literal_hits,
        "current_seed_literal_unexpected_paths": current_unexpected,
        "declaration_count": len(actual_records),
        "declaration_records": actual_records,
        "derivation_mismatch_count": derivation_mismatch_count,
        "formal_runtime_root": str(FORMAL_RUNTIME_ROOT),
        "formal_runtime_root_exists": runtime_exists,
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "namespace": NAMESPACE,
        "namespace_current_paths": namespace_hits,
        "namespace_unexpected_paths": namespace_unexpected,
        "prewrite_collision_scan": prewrite,
        "prewrite_collision_scan_sha256": canonical_json_sha256(prewrite),
        "reference_count": references["total"],
        "reference_counts_by_domain": references,
        "schedule_file_sha256": file_sha256(repository / SEED_SCHEDULE_RELATIVE),
        "schedule_mismatch_count": schedule_mismatch_count,
        "schema_version": AUDIT_SCHEMA,
        "seed_declaration_count": len(actual_records),
        "seed_plan_reference_count": references["total"],
        "test_formal_seed_constructor_hit_count": len(constructor_hits),
        "test_formal_seed_constructor_hits": constructor_hits,
    }


def write_v3_seed_provenance_evidence(
    root: str | Path, evidence_dir: str | Path
) -> dict[str, Any]:
    """Write only the compact audit to an explicitly external evidence path."""

    repository = Path(root).resolve()
    destination = Path(evidence_dir).resolve()
    if destination == repository or repository in destination.parents:
        raise ValueError("seed provenance evidence directory must be external")
    if destination == FORMAL_RUNTIME_ROOT or FORMAL_RUNTIME_ROOT in destination.parents:
        raise ValueError("seed provenance evidence must not enter the formal runtime root")
    report = audit_v3_seed_provenance(repository)
    if report.get("STATIC_V3_SEED_PROVENANCE_AUDIT_PASS") is not True:
        raise RuntimeError("v3 seed provenance audit failed")
    write_json(destination / "v3_seed_provenance_audit.json", report)
    return report


__all__ = [
    "AUDIT_SCHEMA",
    "BOOTSTRAP_LITERAL_PATHS",
    "GEOMETRY_OR_MEASUREMENT_LITERAL_PATHS",
    "NAMESPACE_LITERAL_PATHS",
    "PREWRITE_SCAN_SCHEMA",
    "audit_v3_seed_provenance",
    "scan_v3_prewrite_collisions",
    "write_v3_seed_provenance_evidence",
]
