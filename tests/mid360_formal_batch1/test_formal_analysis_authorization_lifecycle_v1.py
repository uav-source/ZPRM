from __future__ import annotations

from pathlib import Path

import pytest

from experiments.mid360_formal_batch1.analysis_authorization.authorized_analysis_runner_v1 import (
    run_fixture_lifecycle,
)
from experiments.mid360_formal_batch1.analysis_authorization.contract_v1 import (
    AuthorizationControlError,
)
from experiments.mid360_formal_batch1.analysis_authorization.formal_analysis_authorization_lifecycle_v1 import (
    invalidate_published_before_unblinding,
    mark_authorization_in_use,
    publish_verified_authorization,
)
from formal_analysis_authorization_fixture_v1 import fixture_request, verified_fixture


def _published(tmp_path: Path):
    repository, lock_dir, output_root = verified_fixture(tmp_path)
    publish_verified_authorization(
        lock_dir, published_at_utc="2026-08-20T00:01:00Z"
    )
    return repository, lock_dir, output_root


def test_success_is_consumed_and_cannot_be_reused(tmp_path: Path) -> None:
    repository, lock_dir, output_root = _published(tmp_path)
    request = fixture_request(repository, lock_dir, output_root)
    calls = []
    receipt = run_fixture_lifecycle(
        request, fixture_callback=lambda: calls.append("called") or "b" * 64,
        first_read_utc="2026-08-20T00:02:00Z",
        completed_utc="2026-08-20T00:03:00Z",
    )
    assert calls == ["called"]
    assert receipt["lifecycle_state"] == "CONSUMED"
    assert receipt["consumed"] is True
    with pytest.raises(AuthorizationControlError, match="already consumed"):
        run_fixture_lifecycle(
            request, fixture_callback=lambda: "c" * 64,
            first_read_utc="2026-08-20T00:04:00Z",
            completed_utc="2026-08-20T00:05:00Z",
        )


def test_post_unblinding_failure_consumes_authorization(tmp_path: Path) -> None:
    repository, lock_dir, output_root = _published(tmp_path)
    request = fixture_request(repository, lock_dir, output_root)
    with pytest.raises(RuntimeError, match="fixture crash"):
        run_fixture_lifecycle(
            request,
            fixture_callback=lambda: (_ for _ in ()).throw(RuntimeError("fixture crash")),
            first_read_utc="2026-08-20T00:02:00Z",
            completed_utc="2026-08-20T00:03:00Z",
        )
    import json
    receipt = json.loads(
        (lock_dir / "formal_analysis_authorization_consumption_receipt.json").read_text()
    )
    assert receipt["lifecycle_state"] == "CONSUMED_BY_FAILED_ANALYSIS_ATTEMPT"
    assert receipt["REAL_VALUES_ALREADY_UNBLINDED"] is True
    assert receipt["reusable"] is False


def test_published_authorization_can_be_voided_only_before_unblinding(tmp_path: Path) -> None:
    _, lock_dir, _ = _published(tmp_path)
    invalidation = invalidate_published_before_unblinding(
        lock_dir, reason="fixture preflight failure",
        invalidated_at_utc="2026-08-20T00:02:00Z",
    )
    assert invalidation["status"] == "VOID_BEFORE_REAL_VALUE_READ"
    assert invalidation["REAL_VALUES_ALREADY_UNBLINDED"] is False
    assert not (lock_dir / "formal_analysis_authorization_consumption_receipt.json").exists()


def test_in_use_authorization_cannot_be_voided_as_blinded(tmp_path: Path) -> None:
    _, lock_dir, _ = _published(tmp_path)
    mark_authorization_in_use(
        lock_dir, first_real_value_read_utc="2026-08-20T00:02:00Z"
    )
    with pytest.raises(AuthorizationControlError, match="cannot invalidate"):
        invalidate_published_before_unblinding(lock_dir, reason="too late")
