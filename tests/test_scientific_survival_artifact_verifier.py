from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from phase_a_harness.scientific_survival_artifact_verifier import (
    FIGURES,
    REQUIRED_FILES,
    SHA_EXCLUDED,
    TABLES,
    verify_scientific_survival_artifact,
)


def _make_artifact(root: Path) -> None:
    (root / "tables").mkdir(parents=True)
    (root / "figures").mkdir()
    for name in TABLES:
        with (root / "tables" / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("field",))
            writer.writerow(("value",))
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + b"data"
    for name in FIGURES:
        (root / "figures" / name).write_bytes(png)
    decision = {
        name: True
        for name in (
            "RAW_EVIDENCE_INTEGRITY_PASS",
            "ARTIFACT_PUBLICATION_PASS",
            "REPLICATE_UNIQUENESS_AUDIT_PASS",
            "PRIMARY_SCENE_EFFECT_UNIQUE_UNIT_PASS",
            "CROSS_BACKEND_UNIQUE_UNIT_PASS",
            "REASSOCIATION_ROBUSTNESS_PASS",
            "MODEL_DATA_LEAKAGE_AUDIT_PASS",
            "MODEL_INCREMENTAL_VALUE_ROBUST_PASS",
            "CLAIM_WORDING_BOUNDARY_PASS",
        )
    }
    decision.update(
        {
            "SCIENTIFIC_SURVIVAL_AUDIT_PASS": True,
            "CONFIRMATORY_RUN_AUTHORIZED": False,
            "REAL_DATA_RUN_AUTHORIZED": False,
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
            "COUNTEREXAMPLE_CLAIM_AUTHORIZED": False,
        }
    )
    for name in ("publisher_only_change_scope.json", "run_manifest.json"):
        (root / name).write_text("{}\n", encoding="utf-8")
    (root / "final_decision.json").write_text(json.dumps(decision), encoding="utf-8")
    for name in (
        "pre_repair_result_inventory.csv",
        "publisher_repair_report.md",
        "replicate_uniqueness_audit.md",
        "counterexample_review_packet.md",
    ):
        (root / name).write_text("evidence\n", encoding="utf-8")
    references = "\n".join(
        [f"tables/{name}" for name in TABLES]
        + [f"figures/{name}" for name in FIGURES]
    )
    (root / "scientific_survival_audit_report.md").write_text(references, encoding="utf-8")
    expected = set(REQUIRED_FILES) - SHA_EXCLUDED
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((root / relative).read_bytes()).hexdigest()}  {relative}\n"
            for relative in sorted(expected)
        ),
        encoding="utf-8",
    )


def test_survival_artifact_verifier_accepts_complete_exact_artifact(tmp_path: Path) -> None:
    _make_artifact(tmp_path)
    first = verify_scientific_survival_artifact(tmp_path, write_report=True)
    second = verify_scientific_survival_artifact(tmp_path, write_report=False)
    assert first == second
    assert second["ARTIFACT_VERIFICATION_PASS"] is True
    assert second["sha256_mismatch_count"] == 0


def test_survival_artifact_verifier_rejects_unlisted_mutation(tmp_path: Path) -> None:
    _make_artifact(tmp_path)
    (tmp_path / "tables" / TABLES[0]).write_text("field\nmutated\n", encoding="utf-8")
    report = verify_scientific_survival_artifact(tmp_path, write_report=False)
    assert report["ARTIFACT_VERIFICATION_PASS"] is False
    assert report["sha256_mismatch_count"] == 1
