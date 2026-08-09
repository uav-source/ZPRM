from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from phase_a_harness.runtime_lifecycle_io import (
    EVENT_VERSION_V2,
    EventJournalError,
    ImmutableRunLockError,
    LeaseUnavailableError,
    RuntimeLifecycleIOError,
    SingleWriterLease,
    SymlinkPathError,
    append_event_v2,
    atomic_create_bytes,
    atomic_create_canonical_json,
    atomic_publish_directory,
    atomic_replace_bytes,
    build_immutable_run_lock,
    canonical_json_bytes,
    canonical_json_sha256,
    create_immutable_run_lock,
    read_canonical_json,
    read_event_journal_v2,
    resume_immutable_run_lock,
    strict_json_loads,
    write_once_immutable_run_lock,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _contract(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "implementation_sha256": SHA_A,
        "planned_snapshot_count": 3,
        "planned_trial_count": 6,
        "run_id": "fixture-lifecycle-v1",
    }
    value.update(updates)
    return value


def test_canonical_json_is_deterministic_compact_utf8_and_strict() -> None:
    left = {"z": [1, 2.5], "a": "测量"}
    right = {"a": "测量", "z": [1, 2.5]}
    payload = canonical_json_bytes(left)
    assert payload == canonical_json_bytes(right)
    assert payload == '{"a":"测量","z":[1,2.5]}\n'.encode()
    assert strict_json_loads(payload) == left
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


@pytest.mark.parametrize("value", [{1: "not-a-JSON-key"}, {"x": (1, 2)}])
def test_canonical_json_rejects_python_only_container_conversions(value: object) -> None:
    with pytest.raises(TypeError):
        canonical_json_bytes(value)


@pytest.mark.parametrize(
    "payload",
    [
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":1e999}',
    ],
)
def test_strict_json_rejects_duplicates_and_every_nonfinite_form(payload: str) -> None:
    with pytest.raises(ValueError):
        strict_json_loads(payload)


def test_read_canonical_json_rejects_valid_but_noncanonical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text('{"z": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeLifecycleIOError, match="canonical"):
        read_canonical_json(path)
    atomic_replace_bytes(path, canonical_json_bytes({"z": 1, "a": 2}))
    assert read_canonical_json(path) == {"a": 2, "z": 1}


def test_atomic_create_is_durable_no_clobber_and_cleans_temporary(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.bin"
    assert atomic_create_bytes(path, b"first") == path
    assert path.read_bytes() == b"first"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_create_bytes(path, b"second")
    assert path.read_bytes() == b"first"
    assert not list(path.parent.glob(".result.bin.tmp-*"))


def test_atomic_replace_replaces_regular_file_and_cleans_temporary(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_replace_bytes(path, b"one")
    atomic_replace_bytes(path, b"two")
    assert path.read_bytes() == b"two"
    assert not list(tmp_path.glob(".state.json.tmp-*"))


@pytest.mark.parametrize("writer", [atomic_create_bytes, atomic_replace_bytes])
def test_atomic_file_writers_reject_symlink_leaf(
    tmp_path: Path, writer: object
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"untouched")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(SymlinkPathError):
        writer(link, b"replacement")  # type: ignore[operator]
    assert target.read_bytes() == b"untouched"


def test_atomic_file_writer_rejects_symlink_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked-parent"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SymlinkPathError):
        atomic_create_bytes(link / "value", b"forbidden")
    assert not (real / "value").exists()


def test_atomic_directory_publication_is_complete_and_no_clobber(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    observed_during_population: list[bool] = []

    def populate(staging: Path) -> None:
        observed_during_population.append(destination.exists())
        (staging / "tables").mkdir()
        (staging / "tables" / "rows.csv").write_text("x\n1\n", encoding="utf-8")
        (staging / "decision.json").write_bytes(canonical_json_bytes({"pass": True}))
        observed_during_population.append(destination.exists())

    assert atomic_publish_directory(destination, populate) == destination
    assert observed_during_population == [False, False]
    assert (destination / "tables" / "rows.csv").read_text() == "x\n1\n"
    assert read_canonical_json(destination / "decision.json") == {"pass": True}
    with pytest.raises(FileExistsError):
        atomic_publish_directory(destination, lambda staging: None)
    assert not list(tmp_path.glob(".artifact.staging-*"))


def test_atomic_directory_publication_cleans_failed_staging(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"

    def fail(staging: Path) -> None:
        (staging / "partial").write_text("partial", encoding="utf-8")
        raise LookupError("injected population failure")

    with pytest.raises(LookupError, match="injected"):
        atomic_publish_directory(destination, fail)
    assert not destination.exists()
    assert not list(tmp_path.glob(".artifact.staging-*"))


def test_atomic_directory_publication_rejects_all_symlinks(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")

    def populate(staging: Path) -> None:
        (staging / "escape").symlink_to(outside)

    with pytest.raises(SymlinkPathError):
        atomic_publish_directory(destination, populate)
    assert not destination.exists()
    link_destination = tmp_path / "artifact-link"
    link_destination.symlink_to(outside)
    with pytest.raises(SymlinkPathError):
        atomic_publish_directory(link_destination, lambda staging: None)


def test_single_writer_lease_excludes_second_writer_and_is_reusable(tmp_path: Path) -> None:
    path = tmp_path / "control" / "run.lease"
    first = SingleWriterLease(path)
    second = SingleWriterLease(path)
    with first:
        assert first.acquired
        with pytest.raises(LeaseUnavailableError, match="already held"):
            second.acquire()
    assert not first.acquired
    with second:
        assert second.acquired
    assert path.is_file()


def test_single_writer_lease_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.lease"
    target.touch()
    link = tmp_path / "run.lease"
    link.symlink_to(target)
    with pytest.raises(SymlinkPathError):
        SingleWriterLease(link).acquire()


def test_immutable_run_lock_is_write_once_and_resume_exact(tmp_path: Path) -> None:
    path = tmp_path / "run-lock.json"
    contract = _contract()
    expected = build_immutable_run_lock(contract)
    created = create_immutable_run_lock(path, contract)
    assert created == expected
    assert path.read_bytes() == canonical_json_bytes(expected)
    assert resume_immutable_run_lock(path, dict(reversed(list(contract.items())))) == expected
    assert write_once_immutable_run_lock(path, contract, resume=True) == expected
    with pytest.raises(FileExistsError):
        write_once_immutable_run_lock(path, contract, resume=False)


def test_immutable_run_lock_rejects_tamper_and_resigned_wrong_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-lock.json"
    create_immutable_run_lock(path, _contract())
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["contract"]["planned_trial_count"] = 7
    atomic_replace_bytes(path, canonical_json_bytes(tampered))
    with pytest.raises(ImmutableRunLockError, match="payload SHA mismatch"):
        resume_immutable_run_lock(path, _contract())

    resigned = build_immutable_run_lock(_contract(planned_trial_count=7))
    atomic_replace_bytes(path, canonical_json_bytes(resigned))
    with pytest.raises(ImmutableRunLockError, match="different contract"):
        resume_immutable_run_lock(path, _contract())


def test_immutable_run_lock_rejects_noncanonical_and_symlink(tmp_path: Path) -> None:
    path = tmp_path / "run-lock.json"
    value = build_immutable_run_lock(_contract())
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ImmutableRunLockError, match="cannot read"):
        resume_immutable_run_lock(path, _contract())
    target = tmp_path / "actual-lock.json"
    atomic_create_canonical_json(target, value)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(SymlinkPathError):
        resume_immutable_run_lock(path, _contract())


def test_event_journal_v2_append_read_and_cross_invocation_chain(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    started = append_event_v2(
        path,
        run_id="run-a",
        invocation_id="invocation-1",
        event_type="STARTED",
        planned_trial_id="trial-1",
        backend="backend-a",
    )
    completed = append_event_v2(
        path,
        run_id="run-a",
        invocation_id="invocation-1",
        event_type="COMPLETED",
        planned_trial_id="trial-1",
        backend="backend-a",
        result_sha256=SHA_A,
        sequence=2,
        previous_event_sha256=started["event_sha256"],
    )
    resumed = append_event_v2(
        path,
        run_id="run-a",
        invocation_id="invocation-2",
        event_type="RESUMED",
        sequence=3,
        previous_event_sha256=completed["event_sha256"],
    )
    events = read_event_journal_v2(path, expected_run_id="run-a")
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert events[0]["event_version"] == EVENT_VERSION_V2
    assert events[1]["previous_event_sha256"] == events[0]["event_sha256"]
    assert events[2] == resumed
    assert path.read_bytes().endswith(b"\n")


def test_event_journal_v2_records_sigterm_without_result(tmp_path: Path) -> None:
    event = append_event_v2(
        tmp_path / "events.ndjson",
        run_id="run-a",
        invocation_id="invocation-1",
        event_type="INFRASTRUCTURE_INTERRUPTION",
        planned_trial_id="trial-1",
        backend="backend-a",
        signal_number=15,
    )
    assert event["signal_number"] == 15
    assert event["result_sha256"] is None


def test_event_journal_rejects_wrong_run_sequence_and_semantics(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    append_event_v2(
        path,
        run_id="run-a",
        invocation_id="invocation-1",
        event_type="STARTED",
        planned_trial_id="trial-1",
        backend="backend-a",
    )
    with pytest.raises(EventJournalError, match="different run"):
        append_event_v2(
            path,
            run_id="run-b",
            invocation_id="invocation-2",
            event_type="RESUMED",
        )
    with pytest.raises(EventJournalError, match="sequence"):
        append_event_v2(
            path,
            run_id="run-a",
            invocation_id="invocation-1",
            event_type="RESUMED",
            sequence=9,
        )
    invalid_path = tmp_path / "invalid-events.ndjson"
    with pytest.raises(EventJournalError, match="result SHA required"):
        append_event_v2(
            invalid_path,
            run_id="run-a",
            invocation_id="invocation-1",
            event_type="COMPLETED",
            planned_trial_id="trial-1",
            backend="backend-a",
        )
    assert not invalid_path.exists()


def test_event_journal_strict_reader_rejects_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    append_event_v2(
        path,
        run_id="run-a",
        invocation_id="invocation-1",
        event_type="STARTED",
        planned_trial_id="trial-1",
        backend="backend-a",
    )
    payload = path.read_bytes()
    path.write_bytes(payload[:-1])
    with pytest.raises(EventJournalError, match="torn final event line"):
        read_event_journal_v2(path)
    with pytest.raises(EventJournalError, match="torn final event line"):
        append_event_v2(
            path,
            run_id="run-a",
            invocation_id="invocation-2",
            event_type="RESUMED",
        )


def test_event_journal_strict_reader_rejects_tamper_and_symlink(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    append_event_v2(
        path,
        run_id="run-a",
        invocation_id="invocation-1",
        event_type="COMPLETED",
        planned_trial_id="trial-1",
        backend="backend-a",
        result_sha256=SHA_A,
    )
    event = strict_json_loads(path.read_bytes())
    event["result_sha256"] = SHA_B
    path.write_bytes(canonical_json_bytes(event))
    with pytest.raises(EventJournalError, match="event SHA mismatch"):
        read_event_journal_v2(path)

    target = tmp_path / "actual-events.ndjson"
    path.replace(target)
    path.symlink_to(target)
    with pytest.raises(SymlinkPathError):
        read_event_journal_v2(path)


def test_event_journal_reader_rejects_noncanonical_line_and_wrong_expected_run(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.ndjson"
    event = append_event_v2(
        path,
        run_id="run-a",
        invocation_id="invocation-1",
        event_type="RESUMED",
    )
    path.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(EventJournalError):
        read_event_journal_v2(path)

    path.unlink()
    append_event_v2(
        path,
        run_id="run-a",
        invocation_id="invocation-1",
        event_type="RESUMED",
    )
    with pytest.raises(EventJournalError, match="different run"):
        read_event_journal_v2(path, expected_run_id="run-b")
