# FMB1 locked analysis — reporting-corrected view v1

Scientific values remain sourced byte-for-value from the original frozen analysis summary. This file changes solver-status reporting only.

## Corrected solver accounting

- Formal finite outcomes: 360/360.
- Formal `COMPLETED`: 360/360.
- Formal `SOLVER_NON_CONVERGENCE`: 0/360.
- PCL native `has_converged=true`: 180/180.
- Open3D native stopping convergence: not retrospectively observable.

## Original frozen scientific-result references

- Open3D primary estimand: `-2.6347781758088518e-05` m; exact p `0.7`.
- PCL primary estimand: `-1.0915494546904852e-05` m; exact p `0.7`.
- Open3D centered reassociation rho: `0.7376914925357778`; p `9.999000099990002e-05`.
- PCL centered reassociation rho: `0.7611716411000339`; p `9.999000099990002e-05`.

These values were copied from the original frozen `analysis_summary.json`; no scientific endpoint was recalculated.
