"""Independent, static provenance audit for the declared Confirmatory v2 seeds.

The audit reads text/CSV/JSON and Git history only.  It has no NumPy, random,
snapshot-builder, runner, or backend import and cannot instantiate a seed.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .synthetic_confirmatory_v2_contract import (
    BOOTSTRAP_SEED,
    GEOMETRY_SEEDS,
    MEASUREMENT_SEEDS,
    NAMESPACE,
    OLD_V1_BOOTSTRAP_SEED,
    OLD_V1_GEOMETRY_SEEDS,
    OLD_V1_MEASUREMENT_SEEDS,
)


BASE_COMMIT = "1c78372ef6f2c62e441f69c028b58f7bc48f6c35"
DEVELOPMENT_SEEDS = frozenset(
    {1850310744, 1957656152, 1334931069, 217775206, 1664898153}
)
TEXT_SUFFIXES = frozenset({".csv", ".json", ".md", ".py", ".yaml", ".yml"})
TEXT_ROOTS = ("src", "scripts", "tests", "protocols", "frozen_assets")
GEOMETRY_OR_MEASUREMENT_LITERAL_PATHS = frozenset(
    {
        "frozen_assets/synthetic_confirmatory_formal_manifest_v2.json",
        "frozen_assets/synthetic_confirmatory_v2_seed_schedule.json",
        "protocols/synthetic_confirmatory_planned_snapshots_v2.csv",
        "protocols/synthetic_confirmatory_planned_trials_v2.csv",
        "protocols/synthetic_confirmatory_protocol_v2.json",
        "src/phase_a_harness/synthetic_confirmatory_v2_contract.py",
    }
)
BOOTSTRAP_LITERAL_PATHS = frozenset(
    {
        "frozen_assets/synthetic_confirmatory_formal_manifest_v2.json",
        "frozen_assets/synthetic_confirmatory_v2_seed_schedule.json",
        "protocols/synthetic_confirmatory_protocol_v2.json",
        "src/phase_a_harness/synthetic_confirmatory_v2_contract.py",
    }
)
NAMESPACE_LITERAL_PATHS = frozenset(
    {
        "frozen_assets/synthetic_confirmatory_formal_manifest_v2.json",
        "frozen_assets/synthetic_confirmatory_v2_seed_schedule.json",
        "protocols/synthetic_confirmatory_protocol_v2.json",
        "protocols/synthetic_confirmatory_protocol_v2.md",
        "src/phase_a_harness/synthetic_confirmatory_v2_contract.py",
        "tests/test_synthetic_confirmatory_v2.py",
    }
)


def _text_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for relative in TEXT_ROOTS
        for path in (root / relative).rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and "__pycache__" not in path.parts
    )


def _literal_paths(root: Path, needle: str, paths: Iterable[Path]) -> list[str]:
    hits = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if needle in text:
            hits.append(path.relative_to(root).as_posix())
    return sorted(hits)


def _base_hits(root: Path, needle: str) -> list[str]:
    result = subprocess.run(
        ["git", "grep", "-n", "-F", "-e", needle, BASE_COMMIT, "--"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git grep failed for provenance audit: {result.stderr}")
    return sorted(line for line in result.stdout.splitlines() if line)


def _strict_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"seed audit JSON root must be an object: {path}")
    return value


def _plan_reference_counts(root: Path) -> dict[str, int]:
    values = {"geometry": 0, "measurement": 0}
    for relative in (
        "protocols/synthetic_confirmatory_planned_snapshots_v2.csv",
        "protocols/synthetic_confirmatory_planned_trials_v2.csv",
    ):
        with (root / relative).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"seed plan header is missing: {relative}")
            for row in reader:
                geometry = int(row["geometry_seed"])
                if geometry not in GEOMETRY_SEEDS:
                    raise ValueError("plan references a non-v2 geometry seed")
                values["geometry"] += 1
                if row["measurement_seed"] not in (None, ""):
                    measurement = int(row["measurement_seed"])
                    if measurement not in MEASUREMENT_SEEDS:
                        raise ValueError("plan references a non-v2 measurement seed")
                    values["measurement"] += 1
    values["total"] = values["geometry"] + values["measurement"]
    return values


def _test_constructor_hits(root: Path, seeds: frozenset[int]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in sorted((root / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            literals = sorted(
                {
                    int(child.value)
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant)
                    and type(child.value) is int
                    and child.value in seeds
                }
            )
            if literals:
                hits.append(
                    {
                        "line": int(node.lineno),
                        "relative_path": path.relative_to(root).as_posix(),
                        "seeds": literals,
                    }
                )
    return hits


def _derive(domain: str, index: int) -> int:
    digest = hashlib.sha256(f"{NAMESPACE}|{domain}|{index}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) % 2147483647
    return 1 if value == 0 else value


def audit_v2_seed_provenance(root: str | Path) -> dict[str, Any]:
    """Recompute declaration, plan-reference, collision, and static-use evidence."""

    repository = Path(root).resolve()
    paths = _text_paths(repository)
    formal_seeds = frozenset((*GEOMETRY_SEEDS, *MEASUREMENT_SEEDS, BOOTSTRAP_SEED))
    retired = frozenset(
        (*OLD_V1_GEOMETRY_SEEDS, *OLD_V1_MEASUREMENT_SEEDS, OLD_V1_BOOTSTRAP_SEED)
    )
    if len(formal_seeds) != 9:
        raise ValueError("v2 seed declarations are not nine unique values")

    schedule = _strict_object(
        repository / "frozen_assets/synthetic_confirmatory_v2_seed_schedule.json"
    )
    records = schedule.get("records")
    if type(records) is not list or len(records) != 9:
        raise ValueError("v2 seed schedule declaration inventory is not exact")
    declaration_records = sorted(
        (
            {
                "domain": str(row["domain"]),
                "index": int(row["index"]),
                "seed": int(row["seed"]),
            }
            for row in records
            if type(row) is dict
        ),
        key=lambda row: (row["domain"], row["index"]),
    )
    expected_records = sorted(
        [
            *(
                {"domain": "geometry", "index": index, "seed": seed}
                for index, seed in enumerate(GEOMETRY_SEEDS)
            ),
            *(
                {"domain": "measurement", "index": index, "seed": seed}
                for index, seed in enumerate(MEASUREMENT_SEEDS)
            ),
            {"domain": "bootstrap", "index": 0, "seed": BOOTSTRAP_SEED},
        ],
        key=lambda row: (row["domain"], row["index"]),
    )
    derivation_mismatches = [
        row
        for row in expected_records
        if _derive(row["domain"], row["index"]) != row["seed"]
    ]
    schedule_mismatch_count = int(declaration_records != expected_records)

    current_literal_hits = []
    current_unexpected = []
    base_collision_hits = []
    for seed in sorted(formal_seeds):
        actual = _literal_paths(repository, str(seed), paths)
        expected = (
            BOOTSTRAP_LITERAL_PATHS
            if seed == BOOTSTRAP_SEED
            else GEOMETRY_OR_MEASUREMENT_LITERAL_PATHS
        )
        current_literal_hits.append({"paths": actual, "seed": seed})
        for relative in sorted(set(actual) ^ set(expected)):
            current_unexpected.append({"relative_path": relative, "seed": seed})
        for hit in _base_hits(repository, str(seed)):
            base_collision_hits.append({"git_grep_hit": hit, "seed": seed})

    namespace_hits = _literal_paths(repository, NAMESPACE, paths)
    namespace_unexpected = sorted(set(namespace_hits) ^ set(NAMESPACE_LITERAL_PATHS))
    namespace_base_hits = _base_hits(repository, NAMESPACE)
    constructor_hits = _test_constructor_hits(repository, formal_seeds)
    references = _plan_reference_counts(repository)
    development_collisions = sorted(formal_seeds & DEVELOPMENT_SEEDS)
    retired_collisions = sorted(formal_seeds & retired)
    collision_count = (
        len(base_collision_hits)
        + len(development_collisions)
        + len(retired_collisions)
        + schedule_mismatch_count
        + len(derivation_mismatches)
    )
    passed = bool(
        collision_count == 0
        and not current_unexpected
        and not namespace_base_hits
        and not namespace_unexpected
        and not constructor_hits
        and references == {"geometry": 1785, "measurement": 1575, "total": 3360}
    )
    return {
        "NEW_V2_NAMESPACE_COLLISION": bool(namespace_base_hits),
        "NEW_V2_SEED_PROVENANCE_COLLISION_COUNT": collision_count,
        "STATIC_V2_SEED_PROVENANCE_AUDIT_PASS": passed,
        "base_commit": BASE_COMMIT,
        "base_commit_seed_literal_hits": base_collision_hits,
        "current_seed_literal_hits": current_literal_hits,
        "current_seed_literal_unexpected_path_count": len(current_unexpected),
        "current_seed_literal_unexpected_paths": current_unexpected,
        "declaration_count": len(declaration_records),
        "declaration_records": declaration_records,
        "derivation_mismatch_count": len(derivation_mismatches),
        "development_seed_collision_count": len(development_collisions),
        "namespace_base_commit_hits": namespace_base_hits,
        "namespace_current_paths": namespace_hits,
        "namespace_unexpected_path_count": len(namespace_unexpected),
        "namespace_unexpected_paths": namespace_unexpected,
        "reference_count": references["total"],
        "reference_counts_by_domain": references,
        "retired_v1_seed_collision_count": len(retired_collisions),
        "schedule_mismatch_count": schedule_mismatch_count,
        "schema_version": "synthetic_confirmatory_v2_static_seed_provenance_audit_v1",
        "test_formal_seed_constructor_hit_count": len(constructor_hits),
        "test_formal_seed_constructor_hits": constructor_hits,
    }


__all__ = ["audit_v2_seed_provenance"]
