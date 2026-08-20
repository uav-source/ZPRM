# FMB1 Zero-Perturbation Mainline Amendment v1.1-R1

Status: `ACTIVE`  
Amendment ID: `FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1`  
Activation effective: `true`  
Formal trial count at this correction: `0`

This is the active versioned pre-backend correction of the retained v1.1 proposal. Activation became effective only after the independent R1 protocol candidate verifier and final-dataset prelock reauthentication passed, with a zero-trial activation record and authenticated `ACTIVE_PROTOCOL.json` pointer. Formal-environment qualification and trial-plan/result-schema verification remain formal-lock gates rather than amendment-activation gates. Activation grants no backend authority: `FORMAL_AUTHORITY=false`, `FORMAL_ICP_UNLOCKED=false`, and `FORMAL_REGISTRATION_AUTHORIZED=false`.

## Why R1 exists

The original v1.1 proposal was written around a provisional `W02 -> W04 replacement` interpretation. The operator later confirmed that the first W02 acquisition had been recorded at the wrong scene. The scientifically correct lineage is therefore:

- `FMB1_W02 attempt 1`: `INVALID_ACQUISITION`, reason `WRONG_SCENE_LOCATION / OPERATOR_SCENE_SELECTION_ERROR`; its six raw bags remain in the invalid-acquisition archive and are excluded from geometry summaries, the 180-snapshot inventory, the trial plan, and all future formal results.
- `FMB1_W02 attempt 2`: the new 2026-08-20 acquisition; all three stations passed acquisition audit, 30 snapshots were frozen, geometry-only admission classified it `WEAK`, and it is the active final W02.
- `W04`: the provisional identifier is retired and is absent from the final dataset and future trial plan.

The correction occurred before any formal ICP and used no registration result: `correction_at_formal_trial_count=0`. The old proposal JSON and Markdown remain byte-for-byte preserved. A sidecar supersession record identifies their hashes and explains why they cannot be activated directly.

## Relationship to the original v1 protocol

The original files remain unchanged:

- `experiments/mid360_formal_batch1/preregistration.yaml`, SHA256 `76ae548874d8c1584fcc033685db4a1e7cf104fd881cbd1c334eba0bfe1a9beb`;
- `experiments/mid360_formal_batch1/analysis_protocol.md`, SHA256 `d453d12e713c546c5a054ceb1710b86eea12fa09e3f244705e46df9db255a879`.

They continue to define the capture-radius feasibility protocol, including H1/H2, translation magnitudes, weak/strong directions, both signs, boundary refinement, and capture-radius endpoints. R1 does not invalidate or retrospectively rewrite that work. Its status is `PRESERVED_SUPPLEMENTARY_NOT_EXECUTED`; this amendment does not authorize its execution.

## Primary real-data question

Under one mechanically supported station, independently recorded MAP and QUERY bags, no intentional platform motion, and an initial transform exactly equal to Identity, does scan-to-map registration produce a nonzero pose update, and does that update vary with scene geometry and correspondence reassociation?

Each future locked trial uses:

```text
T_reference_nominal = Identity
T0 = Identity
Delta_T = inverse(T0) @ T_est
```

Although `Delta_T = T_est` algebraically when `T0=I`, implementations must use the general inverse-composition formula. Translation and rotation perturbations are both zero.

## Physical-reference limitation

The mandatory result metadata value is:

```text
physical_reference_semantics =
NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT
```

MAP and QUERY are independent recordings at the same mechanically supported station. IMU evidence supports only `NO_OBVIOUS_MOTION`. There is no independent submillimeter external ground truth, and IMU data do not prove physical displacement below 1 mm. The permitted interpretation is a controlled-static nominal-identity registration update. It is not an independently metrology-validated estimate of absolute physical displacement. This limitation must appear in the protocol, analysis metadata, result metadata, and any paper or report.

## Final data and hierarchy

Only these scenes are eligible:

- Rich: `FMB1_R01`, `FMB1_R02`, `FMB1_R03`;
- Weak: `FMB1_W01`, `FMB1_W02`, `FMB1_W03`.

The hierarchy is `scene -> station -> snapshot -> backend`. Scene is the highest independent unit. Each scene contains three stations and 30 snapshots; each station contains ten snapshots. The batch contains 6 scenes, 18 stations, and 180 snapshots. The snapshots are nested repeats and must never be reported as 180 independent scenes.

Each of the 180 snapshots is planned once with `OPEN3D_POINT_TO_PLANE` and once with `PCL_POINT_TO_PLANE`, giving 180 + 180 = 360 planned trials and zero Native trials. A later authenticated plan must reject W04, W02 attempt 1, non-Identity T0, capture-radius perturbations, unapproved backends, duplicate rows, and missing rows.

## Integrity and exclusion policy

The frozen backend parameter contract remains unchanged at SHA256 `6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9`. Source and target hashes, exact trial IDs, track, scene/station/snapshot identity, backend, environment, execution-code commit, and Identity T0 must be authenticated before execution.

No result may be excluded because it is large, surprising, contrary to the Weak/Rich expectation, non-convergent but scientifically finite, directionally inconsistent, an outlier, or in disagreement across backends. Only an explicit infrastructure failure may be retried. A scientific result must never be rerun until it looks successful.

The detailed scene-first endpoints, exact 3-vs-3 inference, cross-backend summaries, common reassociation implementation, systematic fraction, and fail-closed model-transfer decision are frozen in `zero_perturbation_analysis_contract_v1_1_r1.json`.

## Active protocol and separate execution gate

The candidate bytes and their hashes remain preserved in the independent candidate-verification report and activation-transition record. R1 was activated while the formal trial count remained zero. Activation confers scientific protocol authority only; it does not confer execution authority. Formal lock issuance still must leave `FORMAL_ICP_UNLOCKED=false` and `FORMAL_REGISTRATION_AUTHORIZED=false`, and a separate future authorization is required before any real backend call.
