# Real-data Validation Protocol Framework v1

This deliverable is a design framework only. No dataset has been downloaded, selected, or executed.

- `REAL_DATA_PROTOCOL_FRAMEWORK_READY = true`
- `REAL_DATA_DATASET_ELIGIBILITY_COMPLETE = false`
- `REAL_DATA_RUN_AUTHORIZED = false`

## Formal comparison

Corridor/weak geometry is compared with geometry-rich environments using continuous translation and rotation displacement, cross-backend scene ranking, offline turnover/error association, and an uncertainty-normalized effect.

## Eligibility requirements

| ID | Mandatory requirement | Required evidence |
|---|---|---|
| R01 | At least two independent public data sources | dataset citations and immutable release identifiers |
| R02 | Independent high-accuracy 6DoF reference | reference-system specification and accuracy evidence |
| R03 | Evaluated scan is not used to build its target map | map-construction lineage |
| R04 | Target map and evaluated scan have independent source acquisition | scan/map source identifiers |
| R05 | Time synchronization and extrinsics are auditable | timestamps, calibration, and transform chain |
| R06 | Each dataset has at least 50 corridor or weak-geometry snapshots and 50 geometry-rich snapshots | frozen snapshot inventory |
| R07 | Scene labels are frozen before registration error is viewed | timestamped blinded labeling record |
| R08 | Open3D and PCL share identical inputs | per-backend input checksums |
| R09 | Frozen Open3D and PCL parameters are retained | parameter-lock SHA-256 |
| R10 | Reference trajectory, map, and interpolation uncertainty are recorded | uncertainty budget |
| R11 | Primary effect exceeds synthetic and real-reference uncertainty budgets | preregistered uncertainty-normalized comparison |
| R12 | Continuous errors are primary; an arbitrary success threshold is not the sole result | analysis contract |
| R13 | Common-association metrics use the frozen offline definition | common-analyzer implementation SHA-256 |
| R14 | Scene intervals cannot be reselected after results are inspected | immutable interval-selection manifest |

## Frozen analysis boundaries

Scene labels and intervals must be frozen before registration errors are viewed. Open3D and PCL share byte-identical inputs and retain frozen parameters. Common-association metrics retain the synthetic offline definition and are not treated as backend-internal correspondences or causal proof.

The primary scene effect must exceed both the synthetic uncertainty budget and the independent real-reference uncertainty budget. An arbitrary binary success threshold cannot replace the continuous primary outcomes.

Protocol payload SHA-256: `0ce3fb58a2d10395ab075f02ada9d5e732a11f12259c3409d182e46527e0d97b`

