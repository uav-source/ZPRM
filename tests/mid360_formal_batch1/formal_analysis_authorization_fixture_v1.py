"""Artificial metadata-only fixtures for authorization-control tests."""

from __future__ import annotations

from pathlib import Path

from experiments.mid360_formal_batch1.analysis_authorization.analysis_lock_v2_verify import (
    _fixture_lock,
    _fixture_repository,
    _fixture_request,
)
from experiments.mid360_formal_batch1.analysis_authorization.formal_analysis_authorization_producer_v1 import (
    prepare_authorization_candidate,
)
from experiments.mid360_formal_batch1.analysis_authorization.formal_analysis_authorization_verify_v1 import (
    verify_authorization_candidate,
)


CONTROL_COMMIT = "1" * 40
RELEASE_COMMIT = "2" * 40
NONCE = "a" * 32


def prepared_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = Path(__file__).resolve().parents[2]
    repository, lock_dir, output_root = _fixture_repository(source, tmp_path)
    _fixture_lock(source, repository, lock_dir, output_root, CONTROL_COMMIT)
    prepare_authorization_candidate(
        repository, lock_dir,
        analysis_control_commit=CONTROL_COMMIT,
        analysis_lock_release_commit=RELEASE_COMMIT,
        nonce=NONCE,
        confirm_explicit_user_analysis_authorization=True,
        output_root=output_root,
        issued_at_utc="2026-08-20T00:00:00Z",
        fixture_only=True,
    )
    return repository, lock_dir, output_root


def verified_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository, lock_dir, output_root = prepared_fixture(tmp_path)
    verify_authorization_candidate(
        repository, lock_dir,
        expected_analysis_control_commit=CONTROL_COMMIT,
        expected_lock_release_commit=RELEASE_COMMIT,
        expected_output_root=output_root,
        fixture_only=True,
    )
    return repository, lock_dir, output_root


def fixture_request(repository: Path, lock_dir: Path, output_root: Path):
    return _fixture_request(repository, lock_dir, output_root)
