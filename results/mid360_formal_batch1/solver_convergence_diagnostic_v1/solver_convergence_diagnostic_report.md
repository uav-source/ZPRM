# FMB1 solver-convergence diagnostic v1

Diagnosis: `ANALYSIS_ACCOUNTING_BUG_WRONG_STATUS_MAPPING + OPEN3D_NATIVE_CONVERGENCE_UNOBSERVABLE`.

## Reconciliation

| Diagnostic | Open3D | PCL | Total |
|---|---:|---:|---:|
| A: finite + formal `SOLVER_NON_CONVERGENCE` | 0 | 0 | 0 |
| B: frozen runner semantics | 0 (formal proxy) | 0 (native) | 0 |
| C: locked-analysis string expression | 180 | 180 | 360 |

All 360 rows are finite, `COMPLETED`, `OK`, and carry `solver_status=completed_finite_correspondences`. The 360 figure is therefore a status-token mapping defect, not evidence that 360 native solvers failed.

Open3D native convergence is unobservable from the frozen artifacts, so the true cross-backend native nonconverged total is `UNDETERMINABLE` (bounded 0..180). PCL native nonconverged count is determinably 0; PCL iteration counts were dropped.

## Scientific impact

The defect affects only convergence accounting/reporting. Translation, rotation, reassociation, cross-backend, and systematic numerical results use the preserved finite values and do not depend on this counter. Raw poses are unchanged and no registration rerun is required.

Recommended action: `VERSIONED_REPORTING_CORRECTION_WITHOUT_REGISTRATION_RERUN`.
