# FMB1 Zero-Perturbation locked-analysis determinacy audit v1

Status: `FAIL_CLOSED_UNDER_SPECIFIED_REQUIRED_RULES`

This audit was completed at `2026-08-20T10:07:14Z` from the active amendment, analysis contract/protocol, C1 missingness clarification, active pointer, and Post-run PASS report. No formal trial result file was opened, no real scientific value was read, and no backend, real aggregation, Weak/Rich comparison, p-value, or capture-radius computation was executed.

## Gate outcome

- Audited required rules: **18**
- Required rules classified `UNDER_SPECIFIED_REQUIRED_RULE`: **6**
- Distinct unresolved root definitions: **5**
- `ANALYSIS_CONTRACT_FULLY_DETERMINATE=false`
- `LOCKED_ANALYSIS_IMPLEMENTATION_FREEZE_READY=false`
- `READY_FOR_LOCKED_SCIENTIFIC_ANALYSIS=false`

The locked-analysis implementation and analysis lock were therefore not created. No reasonable definition was selected on the basis of real results.

## Under-specified required rules

1. **Scene-ordering agreement** — required, but the statistic, tie behavior, completeness rule, and exact output are absent. Kendall tau, Spearman, pairwise concordance, and exact rank equality are non-equivalent possibilities.
2. **Reassociation scene median turnover** — both `correspondence_turnover` and `accepted_source_turnover` are frozen, but the formal turnover endpoint or dual-endpoint hierarchy is not.
3. **Reassociation scene-level association** — Spearman and 6/6 completeness are fixed, but the turnover endpoint remains unresolved.
4. **Within-scene centered association** — centering and 180-pair completeness are fixed, but neither the turnover endpoint nor the association statistic is uniquely specified.
5. **Registered centered sensitivity permutation** — seed `20260820`, 10,000 within-scene permutations, and two-sidedness are fixed, but the test statistic and exact p-value comparison/counting rule are not.
6. **Systematic Weak/Rich comparison** — scene is the comparison unit and backends are separate, but the estimand, statistic, descriptive/inferential role, sidedness, and p-value rule are absent.

These six audit rows reduce to five root definitions because the unresolved turnover endpoint affects three reassociation outputs.

## Determined rules

The primary 30-snapshot translation summaries, Weak-minus-Rich estimand, exact 20-allocation one-sided permutation, secondary rotation cross-reference, scene/station Spearman completeness behavior, descriptive direction cosine statuses, station/scene systematic fraction, missingness gates, available-case restrictions, and solver/nonfinite accounting are uniquely determined or explicitly cross-referenced.

## Required corrective action

Before any formal scientific result is read for analysis, issue a new versioned pre-analysis clarification that freezes all five root definitions. It must not modify or conceal the current audit. After that clarification is independently verified, rerun the determinacy audit from the new frozen sources. Until then, analysis implementation, code-freeze tag, analysis lock, and real scientific analysis remain prohibited.
