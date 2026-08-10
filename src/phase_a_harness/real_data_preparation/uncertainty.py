"""Conservative uncertainty semantics; unknown values are never zero-filled."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def conservative_uncertainty(
    components: Iterable[Mapping[str, Any]], *, independence_and_one_sigma_proven: bool
) -> dict[str, Any]:
    rows = [dict(row) for row in components]
    unknown = [row["name"] for row in rows if row.get("value") in (None, "", "UNKNOWN")]
    for row in rows:
        value = row.get("value")
        if value == 0 and row.get("evidence_type") in (None, "UNKNOWN"):
            raise ValueError(f"unknown uncertainty cannot be encoded as zero: {row['name']}")
    if unknown:
        return {
            "combined": "UNKNOWN",
            "combination_rule": "NOT_COMPUTABLE_UNKNOWN_COMPONENT",
            "unknown_components": unknown,
        }
    values = [float(row["value"]) for row in rows]
    if independence_and_one_sigma_proven:
        return {
            "combined": sum(value * value for value in values) ** 0.5,
            "combination_rule": "RSS_INDEPENDENT_ONE_SIGMA",
            "unknown_components": [],
        }
    return {
        "combined": sum(values),
        "combination_rule": "CONSERVATIVE_LINEAR_SUM",
        "unknown_components": [],
    }
