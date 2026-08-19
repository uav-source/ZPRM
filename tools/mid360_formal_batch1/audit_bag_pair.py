#!/usr/bin/env python3
"""Audit an already recorded FMB1 MAP/QUERY pair without modifying bags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.protocol import (  # noqa: E402
    audit_bag,
    audit_pair,
    parse_bag_filename,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-bag", type=Path, required=True)
    parser.add_argument("--query-bag", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    map_id = parse_bag_filename(args.map_bag.name)
    query_id = parse_bag_filename(args.query_bag.name)
    if (map_id["scene_id"], map_id["station_id"], map_id["role"]) != (
        query_id["scene_id"], query_id["station_id"], "MAP"
    ) or query_id["role"] != "QUERY":
        parser.error("MAP and QUERY names must bind the same FMB1 scene/station")
    config = json.loads(
        (REPOSITORY / "configs/mid360_pilot_config.json").read_text(encoding="utf-8")
    )
    map_audit = audit_bag(args.map_bag.resolve(strict=True), "MAP", config)
    query_audit = audit_bag(args.query_bag.resolve(strict=True), "QUERY", config)
    pair_result = audit_pair(map_audit, query_audit)
    payload = {
        "schema": "mid360_formal_batch1_pair_metadata_v1",
        "batch_id": "FMB1",
        "scene_id": map_id["scene_id"],
        "station_id": map_id["station_id"],
        "attempt": args.attempt,
        "map_audit": map_audit,
        "query_audit": query_audit,
        "pair_audit": pair_result,
        "raw_data_retained": True,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_MEASUREMENT_RESULT": False,
    }
    write_json(args.output, payload, overwrite=False)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if pair_result["FORMAL_PAIR_VALID"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
