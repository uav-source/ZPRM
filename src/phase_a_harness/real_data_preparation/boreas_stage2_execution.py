"""Production plumbing for gated Boreas Stage-2 map/query object streams.

This module owns lifecycle ordering only:

``download -> materialization gate -> injected processing -> checkpoint gate
-> injected checkpoint -> authenticated temporary deletion``.

It does not decode LiDAR, choose preprocessing/geometry parameters, select
snapshots, invoke a registration backend, or define a final verifier.  Those
scientific operations are explicit callbacks supplied under separately frozen
contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .boreas_stage2_remote import (
    AuthorizedRemoteObject,
    DownloadReceipt,
    StrictAllowlistDownloader,
    TemporaryDownloadedObject,
)
from .boreas_v2_stage2_authorization import (
    BoreasStage2AuthorizationError,
    VerifiedStage2Authorization,
)
from .io import canonical_json_bytes
from .stage2_disk_gate import Stage2DiskGate


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAP_STAGE = "MAP_INGEST"
QUERY_FIRST_PASS_STAGE = "QUERY_GEOMETRY_FIRST_PASS"
QUERY_SECOND_PASS_STAGE = "QUERY_CANONICAL_SECOND_PASS"
ALLOWED_STAGES = frozenset(
    {MAP_STAGE, QUERY_FIRST_PASS_STAGE, QUERY_SECOND_PASS_STAGE}
)


class BoreasStage2ExecutionError(RuntimeError):
    """The production lifecycle or an injected callback result was invalid."""


def _strict_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BoreasStage2ExecutionError(f"{field} must be a nonnegative integer")
    return value


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise BoreasStage2ExecutionError(
        f"processing callback metadata is not JSON-native: {type(value).__name__}"
    )


@dataclass(frozen=True)
class Stage2ObjectProcessingResult:
    """Minimal binding returned after one injected scientific operation."""

    result_sha256: str
    checkpoint_projected_bytes: int
    metadata: Mapping[str, Any]

    def validated(self) -> "Stage2ObjectProcessingResult":
        digest = str(self.result_sha256).lower()
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise BoreasStage2ExecutionError("result_sha256 must be lowercase SHA-256")
        projected = _strict_nonnegative_int(
            self.checkpoint_projected_bytes,
            field="checkpoint_projected_bytes",
        )
        metadata = _json_native(self.metadata)
        if not isinstance(metadata, dict):
            raise BoreasStage2ExecutionError("processing metadata must be a JSON object")
        # Round-trip to reject NaN/Infinity and mutable/noncanonical exotic values.
        try:
            canonical_json_bytes(metadata)
        except (TypeError, ValueError) as exc:
            raise BoreasStage2ExecutionError("processing metadata is not canonical JSON") from exc
        return Stage2ObjectProcessingResult(
            result_sha256=digest,
            checkpoint_projected_bytes=projected,
            metadata=json.loads(json.dumps(metadata, allow_nan=False)),
        )


@dataclass(frozen=True)
class Stage2ExecutionSummary:
    stage: str
    completed_object_count: int
    downloaded_bytes: int
    deleted_temporary_bytes: int
    object_result_sha256s: tuple[str, ...]
    receipt_sha256s: tuple[str, ...]

    @property
    def summary_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "completed_object_count": self.completed_object_count,
                    "deleted_temporary_bytes": self.deleted_temporary_bytes,
                    "downloaded_bytes": self.downloaded_bytes,
                    "object_result_sha256s": list(self.object_result_sha256s),
                    "receipt_sha256s": list(self.receipt_sha256s),
                    "stage": self.stage,
                }
            )
        ).hexdigest()


MaterializationProjection = Callable[[AuthorizedRemoteObject], int]
ObjectProcessor = Callable[
    [TemporaryDownloadedObject, AuthorizedRemoteObject],
    Stage2ObjectProcessingResult,
]
CheckpointWriter = Callable[
    [str, AuthorizedRemoteObject, DownloadReceipt, Stage2ObjectProcessingResult], Any
]


class BoreasStage2Execution:
    """Enforce production ordering around explicitly injected callbacks."""

    def __init__(
        self,
        *,
        disk_gate: Stage2DiskGate,
        downloader: StrictAllowlistDownloader,
        authorization: VerifiedStage2Authorization,
    ) -> None:
        if not isinstance(disk_gate, Stage2DiskGate):
            raise BoreasStage2ExecutionError("execution requires a Stage2DiskGate")
        if not isinstance(downloader, StrictAllowlistDownloader):
            raise BoreasStage2ExecutionError(
                "execution requires a StrictAllowlistDownloader"
            )
        if not isinstance(authorization, VerifiedStage2Authorization):
            raise BoreasStage2ExecutionError(
                "execution requires a VerifiedStage2Authorization capability"
            )
        if downloader.authorization is not authorization:
            raise BoreasStage2ExecutionError(
                "execution and downloader must share the exact authorization capability"
            )
        if downloader.disk_gate is not disk_gate:
            raise BoreasStage2ExecutionError(
                "downloader and execution must share the exact same disk gate"
            )
        try:
            authorization.assert_live()
            authorization.bind_disk_gate(disk_gate)
        except BoreasStage2AuthorizationError as exc:
            raise BoreasStage2ExecutionError(
                "execution authorization is not live"
            ) from exc
        self.disk_gate = disk_gate
        self.downloader = downloader
        self.authorization = authorization
        self._started = False

    def start(self, *, operation_id: str = "boreas-v2-stage2") -> dict[str, Any]:
        if self._started:
            raise BoreasStage2ExecutionError("execution start gate already passed")
        try:
            self.authorization.assert_live()
        except BoreasStage2AuthorizationError as exc:
            raise BoreasStage2ExecutionError(
                "execution authorization is no longer live"
            ) from exc
        event = self.disk_gate.assert_start(operation_id=operation_id)
        self._started = True
        return event

    def guard_bulk_materialization(
        self, projected_write_bytes: int, *, artifact_id: str
    ) -> dict[str, Any]:
        """Gate replay preallocation, target NPY, or another caller-owned write."""

        if not self._started:
            raise BoreasStage2ExecutionError("execution has not passed its live start gate")
        try:
            self.authorization.assert_operation_live()
        except BoreasStage2AuthorizationError as exc:
            raise BoreasStage2ExecutionError(
                "execution authorization is no longer live"
            ) from exc
        return self.disk_gate.before_materialization(
            projected_write_bytes, artifact_id=artifact_id
        )

    def guard_checkpoint(
        self, projected_write_bytes: int, *, checkpoint_id: str
    ) -> dict[str, Any]:
        """Gate a caller-owned aggregate/final checkpoint write."""

        if not self._started:
            raise BoreasStage2ExecutionError("execution has not passed its live start gate")
        try:
            self.authorization.assert_operation_live()
        except BoreasStage2AuthorizationError as exc:
            raise BoreasStage2ExecutionError(
                "execution authorization is no longer live"
            ) from exc
        return self.disk_gate.before_checkpoint(
            projected_write_bytes, checkpoint_id=checkpoint_id
        )

    @staticmethod
    def _validated_objects(
        values: Sequence[AuthorizedRemoteObject],
        *,
        expected_role: str,
        require_frozen_order: bool,
        authorized_by_key: Mapping[str, AuthorizedRemoteObject],
    ) -> tuple[AuthorizedRemoteObject, ...]:
        rows = tuple(values)
        if any(not isinstance(row, AuthorizedRemoteObject) for row in rows):
            raise BoreasStage2ExecutionError(
                "execution objects must be reconciled AuthorizedRemoteObject values"
            )
        keys = [row.key for row in rows]
        if len(keys) != len(set(keys)):
            raise BoreasStage2ExecutionError("execution object list contains duplicates")
        if any(row.selection_role != expected_role for row in rows):
            raise BoreasStage2ExecutionError(
                f"execution object role must be {expected_role}"
            )
        if any(authorized_by_key.get(row.key) != row for row in rows):
            raise BoreasStage2ExecutionError(
                "execution object is not in the downloader's reconciled allowlist"
            )
        if require_frozen_order:
            ordinals = [row.frozen.role_ordinal for row in rows]
            if ordinals != sorted(ordinals):
                raise BoreasStage2ExecutionError(
                    "stream objects are not in frozen allowlist order"
                )
        return rows

    def _run_stream(
        self,
        values: Sequence[AuthorizedRemoteObject],
        *,
        stage: str,
        expected_role: str,
        require_frozen_order: bool,
        materialization_projection: MaterializationProjection,
        processor: ObjectProcessor,
        checkpoint_writer: CheckpointWriter,
        expected_prior_receipts: Mapping[str, DownloadReceipt] | None = None,
    ) -> Stage2ExecutionSummary:
        if not self._started:
            raise BoreasStage2ExecutionError("execution has not passed its live start gate")
        try:
            self.authorization.assert_operation_live()
        except BoreasStage2AuthorizationError as exc:
            raise BoreasStage2ExecutionError(
                "execution authorization is no longer live"
            ) from exc
        if stage not in ALLOWED_STAGES:
            raise BoreasStage2ExecutionError(f"unknown Stage-2 stream stage: {stage}")
        if not callable(materialization_projection) or not callable(processor) or not callable(
            checkpoint_writer
        ):
            raise BoreasStage2ExecutionError("all production stream callbacks are mandatory")
        rows = self._validated_objects(
            values,
            expected_role=expected_role,
            require_frozen_order=require_frozen_order,
            authorized_by_key=self.downloader.inventory.by_key,
        )
        if expected_prior_receipts is not None:
            if any(
                not isinstance(value, DownloadReceipt)
                for value in expected_prior_receipts.values()
            ):
                raise BoreasStage2ExecutionError(
                    "prior query receipts must be DownloadReceipt values"
                )
            missing = [row.key for row in rows if row.key not in expected_prior_receipts]
            if missing:
                raise BoreasStage2ExecutionError(
                    f"selected query lacks its first-pass receipt: {missing[0]}"
                )
        downloaded_bytes = 0
        deleted_bytes = 0
        result_hashes: list[str] = []
        receipt_hashes: list[str] = []
        for item in rows:
            materialized = self.downloader.materialize(item)
            downloaded_bytes += item.size_bytes
            try:
                if expected_prior_receipts is not None:
                    prior = expected_prior_receipts[item.key]
                    if (
                        prior.selection_role != "QUERY"
                        or prior.payload_identity()
                        != materialized.receipt.payload_identity()
                    ):
                        raise BoreasStage2ExecutionError(
                            f"second-pass payload differs from first-pass receipt: {item.key}"
                        )
                projected = materialization_projection(item)
                projected = _strict_nonnegative_int(
                    projected, field="materialization projection"
                )
                self.disk_gate.before_materialization(
                    projected,
                    artifact_id=f"{stage}:{item.key}",
                )
                raw_result = processor(materialized, item)
                if not isinstance(raw_result, Stage2ObjectProcessingResult):
                    raise BoreasStage2ExecutionError(
                        "processor must return Stage2ObjectProcessingResult"
                    )
                result = raw_result.validated()
                self.disk_gate.before_checkpoint(
                    result.checkpoint_projected_bytes,
                    checkpoint_id=f"{stage}:{item.key}",
                )
                # ``stage`` is mandatory because a query object can have two
                # legitimate receipts (screening and selected-source pass).
                checkpoint_writer(stage, item, materialized.receipt, result)
                receipt_value = materialized.receipt.as_dict()
                result_hashes.append(result.result_sha256)
                receipt_hashes.append(str(receipt_value["receipt_sha256"]))
            finally:
                if materialized.path.exists():
                    deleted_bytes += self.downloader.release(materialized)
        return Stage2ExecutionSummary(
            stage=stage,
            completed_object_count=len(result_hashes),
            downloaded_bytes=downloaded_bytes,
            deleted_temporary_bytes=deleted_bytes,
            object_result_sha256s=tuple(result_hashes),
            receipt_sha256s=tuple(receipt_hashes),
        )

    def run_map(
        self,
        objects: Sequence[AuthorizedRemoteObject],
        *,
        materialization_projection: MaterializationProjection,
        processor: ObjectProcessor,
        checkpoint_writer: CheckpointWriter,
    ) -> Stage2ExecutionSummary:
        return self._run_stream(
            objects,
            stage=MAP_STAGE,
            expected_role="TARGET_MAP",
            require_frozen_order=True,
            materialization_projection=materialization_projection,
            processor=processor,
            checkpoint_writer=checkpoint_writer,
            expected_prior_receipts=None,
        )

    def run_query_first_pass(
        self,
        objects: Sequence[AuthorizedRemoteObject],
        *,
        materialization_projection: MaterializationProjection,
        processor: ObjectProcessor,
        checkpoint_writer: CheckpointWriter,
    ) -> Stage2ExecutionSummary:
        return self._run_stream(
            objects,
            stage=QUERY_FIRST_PASS_STAGE,
            expected_role="QUERY",
            require_frozen_order=True,
            materialization_projection=materialization_projection,
            processor=processor,
            checkpoint_writer=checkpoint_writer,
            expected_prior_receipts=None,
        )

    def run_query_second_pass(
        self,
        selected_objects: Sequence[AuthorizedRemoteObject],
        *,
        materialization_projection: MaterializationProjection,
        processor: ObjectProcessor,
        checkpoint_writer: CheckpointWriter,
        first_pass_receipts: Mapping[str, DownloadReceipt],
    ) -> Stage2ExecutionSummary:
        # Selection/order is an upstream frozen authority.  This lifecycle only
        # checks membership/role and never re-ranks the selected objects.
        return self._run_stream(
            selected_objects,
            stage=QUERY_SECOND_PASS_STAGE,
            expected_role="QUERY",
            require_frozen_order=False,
            materialization_projection=materialization_projection,
            processor=processor,
            checkpoint_writer=checkpoint_writer,
            expected_prior_receipts=first_pass_receipts,
        )


__all__ = [
    "BoreasStage2Execution",
    "BoreasStage2ExecutionError",
    "MAP_STAGE",
    "QUERY_FIRST_PASS_STAGE",
    "QUERY_SECOND_PASS_STAGE",
    "Stage2ExecutionSummary",
    "Stage2ObjectProcessingResult",
]
