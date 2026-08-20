# FMB1 solver-status reporting correction v1

This is an authoritative versioned reporting correction, not a second scientific analysis.

- Original reported aggregate `solver_nonconverged_finite_n`: **360** (superseded reporting field).
- Corrected formal `COMPLETED`: **360**.
- Corrected formal `SOLVER_NON_CONVERGENCE`: **0**.
- PCL native `has_converged=true`: **180/180**.
- Open3D native convergence: **not retrospectively observable**.

Root cause: the old reporting allowlist accepted only `CONVERGED` and `SUCCESS`, while every valid formal row used `completed_finite_correspondences`.

No registration, scientific analysis, estimand, p-value, reassociation, systematic-component, or raw-pose recomputation was performed.

## Permitted paper wording

All 360 formal registration trials produced finite scientific outcomes and were recorded as COMPLETED under the frozen formal result protocol.

For PCL, native convergence was reported for all 180 trials.

The frozen Open3D wrapper did not preserve the native iteration count or native stopping criterion; therefore native Open3D convergence cannot be retrospectively determined.
