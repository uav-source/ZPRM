# FMB1 analysis-authorization contract audit v1

The frozen firewall can be satisfied by an external authorization-control
layer without changing the frozen statistical implementation. It looks only
for `analysis_lock_dir/formal_analysis_authorization.json` and requires four
exact values: authorization true, lock SHA equality, analysis-code commit
equality, and `consumed=false`.

The frozen firewall does not itself validate a producer/verifier receipt or
one-time consumption state. Analysis Lock v2 therefore freezes the
prepare/independent-verify/publish lifecycle and makes the one-time wrapper the
canonical real-analysis entry. Its compatibility view is a regular,
byte-identical `locked_analysis_lock_v1.json` beside the v2 lock. Candidate and
verification filenames cannot satisfy the firewall.

Both the published authorization and the frozen
`--confirm-read-frozen-formal-results` flag remain mandatory.

`EXTERNAL_AUTH_INFRA_CAN_SATISFY_FROZEN_FIREWALL=true`
`ANALYSIS_CODE_REFREEZE_REQUIRED=false`
