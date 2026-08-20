# FMB1 Formal Analysis Authorization Contract v1

This is execution control, not a scientific-analysis definition.

Authorization requires both a published, independently verified
`formal_analysis_authorization.json` and the frozen CLI flag
`--confirm-read-frozen-formal-results`. A candidate or verifier report alone
has no authority. The canonical lifecycle is `ISSUED → VERIFIED_PUBLISHED →
IN_USE → CONSUMED` (or `CONSUMED_BY_FAILED_ANALYSIS_ATTEMPT` after unblinding).

The authorization permits exactly one frozen Zero-Perturbation locked-analysis
run. It forbids registration, capture-radius execution, input/code/statistics
changes, reuse, and a second issuance. The canonical execution wrapper must
check lifecycle artifacts before invoking the frozen analysis CLI.
