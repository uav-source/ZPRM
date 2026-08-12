from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from phase_a_harness.real_data_preparation.io import canonical_json_bytes, sha256_file
from phase_a_harness.real_data_preparation.public_data_v1_closure import (
    CANDIDATES,
    CANDIDATE_FIELDS,
    CLOSURE_FILES,
    PublicDataV1ClosureError,
    build_public_data_v1_closure,
)
from phase_a_harness.real_data_preparation.public_data_v1_closure_verifier import (
    EXPECTED_PROTOCOLS,
    EXPECTED_SOURCE_BUNDLES,
    PublicDataV1ClosureVerificationError,
    verify_public_data_v1_closure,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN_CLOSURE = REPOSITORY / "frozen_assets/public_data_validation_v1_closure"


def _source_hashes(repository: Path) -> dict[str, str]:
    paths = [repository / row["path"] for row in EXPECTED_PROTOCOLS]
    for bundle in EXPECTED_SOURCE_BUNDLES:
        root = repository / bundle["path"]
        paths.extend(path for path in root.rglob("*") if path.is_file())
    return {
        path.relative_to(repository).as_posix(): sha256_file(path)
        for path in sorted(paths)
    }


def _copy_historical_repository(destination: Path) -> Path:
    for row in EXPECTED_PROTOCOLS:
        source = REPOSITORY / row["path"]
        target = destination / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for bundle in EXPECTED_SOURCE_BUNDLES:
        source = REPOSITORY / bundle["path"]
        target = destination / bundle["path"]
        shutil.copytree(source, target)
    return destination


def _refresh_closure_checksum(root: Path, name: str) -> None:
    checksum = root / "SHA256SUMS"
    lines = checksum.read_text(encoding="utf-8").splitlines()
    rewritten = []
    for line in lines:
        _, recorded = line.split("  ", 1)
        digest = sha256_file(root / name) if recorded == name else line.split("  ", 1)[0]
        rewritten.append(f"{digest}  {recorded}")
    checksum.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def test_frozen_closure_independent_verifier_passes() -> None:
    report = verify_public_data_v1_closure(
        repository=REPOSITORY,
        closure_root=FROZEN_CLOSURE,
    )
    assert report["PUBLIC_DATA_V1_CLOSURE_VERIFICATION_PASS"] is True
    assert report["PUBLIC_DATA_V1_CANDIDATE_COUNT"] == 5
    assert report["PUBLIC_DATA_V1_ELIGIBLE_DATASET_COUNT"] == 0
    assert report["PUBLIC_DATA_V1_REAL_REGISTRATION_COUNT"] == 0
    assert report["PUBLIC_DATA_V1_ICP_COUNT"] == 0


def test_builder_is_deterministic_and_does_not_mutate_v1_sources(tmp_path: Path) -> None:
    before = _source_hashes(REPOSITORY)
    closure = tmp_path / "closure"
    first = build_public_data_v1_closure(repository=REPOSITORY, output_root=closure)
    first_bytes = {path.name: path.read_bytes() for path in closure.iterdir()}
    second = build_public_data_v1_closure(repository=REPOSITORY, output_root=closure)
    assert second == first
    assert {path.name: path.read_bytes() for path in closure.iterdir()} == first_bytes
    assert _source_hashes(REPOSITORY) == before
    assert {path.name for path in closure.iterdir()} == CLOSURE_FILES


def test_candidate_rows_preserve_all_five_historical_decisions() -> None:
    with (FROZEN_CLOSURE / "public_data_v1_candidate_screening.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(CANDIDATE_FIELDS)
        rows = list(reader)
    assert rows == list(CANDIDATES)
    assert {row["candidate_id"] for row in rows} == {
        "IILABS",
        "GrandTour",
        "RTS-GT",
        "CAVERS",
        "Boreas v1",
    }
    boreas = next(row for row in rows if row["candidate_id"] == "Boreas v1")
    assert boreas["historical_status"] == "R02=FAIL;R10=PARTIAL;BOREAS_STAGE1_READY=false"


def test_closure_has_exact_fixed_flags() -> None:
    closure = json.loads(
        (FROZEN_CLOSURE / "public_data_v1_closure.json").read_text(encoding="utf-8")
    )
    assert closure["PUBLIC_DATA_V1_SCREENING_COMPLETE"] is True
    assert closure["PUBLIC_DATA_V1_CANDIDATE_COUNT"] == 5
    assert closure["PUBLIC_DATA_V1_ELIGIBLE_DATASET_COUNT"] == 0
    assert closure["PUBLIC_DATA_V1_REAL_REGISTRATION_COUNT"] == 0
    assert closure["PUBLIC_DATA_V1_ICP_COUNT"] == 0
    assert closure["PUBLIC_DATA_V1_RUN_AUTHORIZED"] is False
    assert closure["PUBLIC_DATA_V1_CLOSED"] is True
    assert closure["boreas_v1_conclusion_preserved"] is True


def test_builder_refuses_to_write_inside_historical_bundle() -> None:
    with pytest.raises(PublicDataV1ClosureError, match="may not overlap"):
        build_public_data_v1_closure(
            repository=REPOSITORY,
            output_root=REPOSITORY / "frozen_assets/real_data_validation_v1/closure",
        )


def test_semantic_tamper_fails_even_after_outer_checksum_is_refreshed(tmp_path: Path) -> None:
    closure = tmp_path / "closure"
    shutil.copytree(FROZEN_CLOSURE, closure)
    path = closure / "public_data_v1_closure.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["PUBLIC_DATA_V1_RUN_AUTHORIZED"] = True
    path.write_bytes(canonical_json_bytes(value))
    _refresh_closure_checksum(closure, path.name)
    with pytest.raises(PublicDataV1ClosureVerificationError, match="closure decision"):
        verify_public_data_v1_closure(repository=REPOSITORY, closure_root=closure)


def test_candidate_reason_tamper_fails_after_checksum_refresh(tmp_path: Path) -> None:
    closure = tmp_path / "closure"
    shutil.copytree(FROZEN_CLOSURE, closure)
    path = closure / "public_data_v1_candidate_screening.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "cross-acquisition fixed world frame not proven", "retroactively eligible"
        ),
        encoding="utf-8",
    )
    _refresh_closure_checksum(closure, path.name)
    with pytest.raises(PublicDataV1ClosureVerificationError, match="candidate screening rows"):
        verify_public_data_v1_closure(repository=REPOSITORY, closure_root=closure)


def test_no_registration_tamper_fails_after_checksum_refresh(tmp_path: Path) -> None:
    closure = tmp_path / "closure"
    shutil.copytree(FROZEN_CLOSURE, closure)
    path = closure / "NO_REAL_REGISTRATION_ATTESTATION.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["registration_execution_count"] = 1
    path.write_bytes(canonical_json_bytes(value))
    _refresh_closure_checksum(closure, path.name)
    with pytest.raises(PublicDataV1ClosureVerificationError, match="no-registration attestation"):
        verify_public_data_v1_closure(repository=REPOSITORY, closure_root=closure)


def test_boreas_v1_r02_source_tamper_is_rejected(tmp_path: Path) -> None:
    repository = _copy_historical_repository(tmp_path / "repository")
    path = repository / "frozen_assets/real_data_boreas_stage1_v1/boreas_stage1_eligibility.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["R02"] = "PASS"
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(PublicDataV1ClosureVerificationError, match="historical SHA256"):
        verify_public_data_v1_closure(repository=repository, closure_root=FROZEN_CLOSURE)


def test_v1_protocol_tamper_is_rejected(tmp_path: Path) -> None:
    repository = _copy_historical_repository(tmp_path / "repository")
    path = repository / EXPECTED_PROTOCOLS[0]["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PublicDataV1ClosureVerificationError, match="v1 protocol SHA"):
        verify_public_data_v1_closure(repository=repository, closure_root=FROZEN_CLOSURE)


def test_extra_closure_file_is_rejected(tmp_path: Path) -> None:
    closure = tmp_path / "closure"
    shutil.copytree(FROZEN_CLOSURE, closure)
    (closure / "untracked.txt").write_text("not part of closure\n", encoding="utf-8")
    with pytest.raises(PublicDataV1ClosureVerificationError, match="closure file set mismatch"):
        verify_public_data_v1_closure(repository=REPOSITORY, closure_root=closure)
