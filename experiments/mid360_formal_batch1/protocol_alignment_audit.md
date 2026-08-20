# FMB1 protocol-alignment audit

Audit timestamp: `2026-08-19T23:33:07+08:00` (`2026-08-19T15:33:07+00:00`), Asia/Shanghai.

Conclusion: W04 replacement planning is compatible with the active preregistration, but FMB1 is not backend-ready and no registration is authorized. The proposed zero-perturbation mainline is not the active preregistered capture-basin analysis.

This was a read-only, pre-backend audit. It did not edit the active preregistration, analysis protocol, backend parameter contract, frozen results, or registries, and it did not invoke a backend.

## Authorities checked

| Authority | SHA256 | Finding |
|---|---|---|
| `experiments/mid360_formal_batch1/preregistration.yaml` | `76ae548874d8c1584fcc033685db4a1e7cf104fd881cbd1c334eba0bfe1a9beb` | Active and unchanged |
| `experiments/mid360_formal_batch1/analysis_protocol.md` | `d453d12e713c546c5a054ceb1710b86eea12fa09e3f244705e46df9db255a879` | Active future-analysis description and unchanged |
| `frozen_assets/backend_parameter_contract.json` | `6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9` | Frozen Open3D/PCL parameters and unchanged |

## Current evidence

The frozen geometry summary has three admitted RICH scenes and two admitted WEAK scenes. `FMB1_W02` was a `WEAK_CANDIDATE`, but geometry-only analysis classified it as `RICH`; it is therefore `GEOMETRY_REJECTED` with `SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH`. The preregistration-compatible replacement category is `GEOMETRY_ONLY_INELIGIBLE`. This decision used no registration-derived field, and W02 remains preserved.

The current readiness artifact is false, the independent verification fails at W02's WEAK gate, `NO_ICP_ATTESTATION` passes, and actual registration-trial count remains zero. Those facts prohibit interpreting the W04 plan as an unlock.

## Alignment findings

1. W04 can be planned before ICP because the active replacement rule permits geometry-only ineligibility and requires rejected candidates to remain retained.
2. W04 cannot enter FMB1 merely by being called weak. It needs three acquisition-passing stations, ten fixed-quantile snapshots per station, 30 geometry-only rows, and all three WEAK median gates.
3. The active preregistration's future formal experiment is a capture-basin design: multiple translation magnitudes, weak/strong directions, both signs, bracket refinement, and capture-radius endpoints. A proposed 180 snapshots × 2 backends = 360 identity-initialization mainline is a different track; the active documents do not currently confer scientific authority on that track.
4. The backend contract calls the supplied initialization `T_reference`; current same-station MAP/QUERY canonical inputs use identity. A future result therefore must be bound to an explicit track and the exact `T0` stored in an authenticated formal lock. Capture-basin trials cannot silently receive identity.
5. Open3D `0.19.0+b012259` and PCL `1.15.1` remain the only backend identities under the unchanged parameter-contract SHA. No scene-specific tuning is aligned.
6. A validator or JSON schema is a data-integrity control, not an execution authorization. Fixture results may be schema-valid only with fixture flags and are never publishable.

## Resolution required before any backend

- Acquire W04 with an actual recorded location and new bag timestamps; do not infer or fabricate site metadata.
- Pass its three-station acquisition and geometry-only admission, then freeze and independently verify a replacement inventory with exactly 3 RICH + 3 WEAK scenes, 18 stations, and 180 snapshots.
- Decide the formal scientific track through a versioned authority. If the identity mainline is approved, activate an amendment explicitly; otherwise retain the preregistered capture-basin design unchanged.
- Generate a new formal lock and fingerprint that bind every trial ID, scene/station/snapshot, backend, source/target hashes, track, and exact `T0`.
- Issue separate explicit authorization only after these gates pass.

Until then: `FORMAL_AUTHORITY=false`, `FORMAL_REGISTRATION_AUTHORIZED=false`, and `MEASUREMENT_FINAL_RESULT=false`.
