#!/usr/bin/env python3
"""Record one FMB1 MAP/wait/QUERY pair; dry-run is side-effect free."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.protocol import (  # noqa: E402
    BATCH_ID,
    MAP_TARGET_DURATION_S,
    QUERY_TARGET_DURATION_S,
    REQUIRED_TOPICS,
    TARGET_MAP_QUERY_GAP_S,
    FormalBatchError,
    audit_bag,
    audit_pair,
    build_recording_plan,
    ensure_paths_do_not_exist,
    read_json_yaml,
    sha256_file,
    write_json,
)


def verify_live_topics() -> dict[str, str]:
    observed: dict[str, str] = {}
    for topic, expected_type in REQUIRED_TOPICS.items():
        process = subprocess.run(
            ["rostopic", "type", topic],
            check=False,
            capture_output=True,
            text=True,
        )
        actual = process.stdout.strip()
        if process.returncode != 0 or actual != expected_type:
            raise FormalBatchError(
                f"live topic check failed for {topic}: expected {expected_type}, got {actual or '<unavailable>'}"
            )
        observed[topic] = actual
    return observed


def verify_registered_station(scene_id: str, station_id: str) -> None:
    registry = read_json_yaml(
        REPOSITORY / "experiments/mid360_formal_batch1/station_registry.yaml"
    )
    keys = {
        (str(row["scene_id"]), str(row["station_id"]))
        for row in registry.get("stations", [])
    }
    if (scene_id, station_id) not in keys:
        raise FormalBatchError(
            "scene/station must be defined in station_registry.yaml before recording"
        )


def record_for(path: Path, duration_s: float) -> None:
    process = subprocess.Popen(
        ["rosbag", "record", "-O", str(path), *REQUIRED_TOPICS],
        stdin=subprocess.DEVNULL,
    )
    try:
        time.sleep(duration_s)
        process.send_signal(signal.SIGINT)
        return_code = process.wait(timeout=30)
    except BaseException:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
        raise
    if return_code != 0:
        raise FormalBatchError(f"rosbag record failed with exit code {return_code}: {path}")
    if not path.is_file():
        raise FormalBatchError(f"rosbag did not produce expected output: {path}")


def _failed_audit(path: Path, role: str, exc: BaseException) -> dict[str, Any]:
    return {
        "schema": "mid360_formal_batch1_bag_audit_v1",
        "role": role,
        "bag_path": str(path),
        "SHA256": sha256_file(path) if path.is_file() else None,
        "ACQUISITION_AUDIT_PASS": False,
        "REVIEW_REQUIRED": True,
        "failure_reasons": [f"AUDIT_EXCEPTION:{type(exc).__name__}:{exc}"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=BATCH_ID)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--station", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--timestamp", help="YYYYMMDD_HHMMSS; intended for dry-run/reproducibility")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = build_recording_plan(
        batch=args.batch,
        scene_id=args.scene,
        station_id=args.station,
        output_dir=args.output_dir.resolve(),
        timestamp=timestamp,
        attempt=args.attempt,
    )
    verify_registered_station(args.scene, args.station)
    if args.dry_run:
        print(
            json.dumps(
                {
                    **plan,
                    "dry_run": True,
                    "mechanically_supported_requirement": "OPERATOR_MUST_VERIFY_PER_PROTOCOL",
                    "same_pose_requirement": "OPERATOR_MUST_VERIFY_PER_PROTOCOL",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = Path(plan["map_path"])
    query_path = Path(plan["query_path"])
    metadata_path = Path(plan["pair_metadata_path"])
    ensure_paths_do_not_exist((map_path, query_path, metadata_path))
    live_topics = verify_live_topics()
    record_for(map_path, MAP_TARGET_DURATION_S)
    time.sleep(TARGET_MAP_QUERY_GAP_S)
    record_for(query_path, QUERY_TARGET_DURATION_S)

    config = json.loads(
        (REPOSITORY / "configs/mid360_pilot_config.json").read_text(encoding="utf-8")
    )
    try:
        map_audit = audit_bag(map_path, "MAP", config)
    except BaseException as exc:  # preserve and mark invalid even on parser failure
        map_audit = _failed_audit(map_path, "MAP", exc)
    try:
        query_audit = audit_bag(query_path, "QUERY", config)
    except BaseException as exc:  # preserve and mark invalid even on parser failure
        query_audit = _failed_audit(query_path, "QUERY", exc)
    try:
        pair_result = audit_pair(map_audit, query_audit)
    except (KeyError, TypeError, ValueError) as exc:
        pair_result = {
            "schema": "mid360_formal_batch1_pair_audit_v1",
            "FORMAL_PAIR_VALID": False,
            "exclusion_reasons": [f"PAIR_AUDIT_INCOMPLETE:{type(exc).__name__}:{exc}"],
            "raw_data_retained": True,
        }
    metadata = {
        "schema": "mid360_formal_batch1_pair_metadata_v1",
        "batch_id": args.batch,
        "scene_id": args.scene,
        "station_id": args.station,
        "attempt": args.attempt,
        "live_topics": live_topics,
        "same_mounting_attested": "UNKNOWN_OPERATOR_MUST_COMPLETE_METADATA",
        "mechanically_supported_attested": "UNKNOWN_OPERATOR_MUST_COMPLETE_METADATA",
        "map_audit": map_audit,
        "query_audit": query_audit,
        "pair_audit": pair_result,
        "raw_data_retained": True,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_MEASUREMENT_RESULT": False,
    }
    write_json(metadata_path, metadata, overwrite=False)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0 if pair_result["FORMAL_PAIR_VALID"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
