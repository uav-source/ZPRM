# Zero-Perturbation v1.1 proposal difference report — R1

The retained proposal cannot be activated verbatim because its W04-replacement premise conflicts with the operator-confirmed and frozen W02 attempt lineage. The old JSON and Markdown remain byte-for-byte unchanged at SHA256 `4837f6bd...936e` and `836822c5...4ed`.

The material differences are:

1. The old proposal requires W04 acquisition/admission. The current final dataset retires W04 and excludes it completely.
2. The old alignment treated W02 as geometry-only ineligible. The corrected lineage says W02 attempt 1 was an invalid acquisition at the wrong scene; W02 attempt 2 is the correct reacquisition and is admitted as Weak.
3. The old proposal coupled activation to a separate registration authorization. R1 keeps protocol activation and formal lock distinct from a later authorization; no backend call is permitted by activation alone.
4. The old proposal points to the generic v1 result schema and does not fully freeze the required physical-reference limitation, exact scene-level inference, common reassociation analysis, systematic fraction, or model-transfer gate.

R1 resolves these issues through new versioned files. It does not rewrite the old proposal, the original preregistration, or the original capture-radius analysis. The correction occurred with zero formal trials and no registration evidence.

Current phase: `R1_CANDIDATE_CREATED_NOT_YET_ACTIVE`. Independent verification and an explicit activation record are still required.
