# Mid-360 translation capture-basin Pilot

`CAPTURE_BASIN_PILOT_READY=true`  
`CAPTURE_BASIN_PILOT_SUPPORTS_FORMAL_EXPANSION=true`  
`FORMAL_MEASUREMENT_RESULT=false`

This is a Pilot descriptive sensitivity analysis. It does not provide a formal
paper measurement or population inference.

The 50 mm level was reused from the byte-authenticated previous controlled
perturbation Pilot; it was not rerun. The base coarse grid contains 1120 rows,
of which 960 are new ICP executions. Automatic extension uses only the frozen
2.4/3.2/4.8/6.4 m sequence and never exceeds 6.4 m. This run produced
1664 coarse/extension rows and 776 boundary executions.

Weak/strong directions use the unchanged existing 3x3 translation geometry
matrix. Eigenvector sign canonicalization is used only for audit/reporting; both
physical signs are executed. `SPATIAL_AXIS_INTERPRETATION=UNRESOLVED` because no
reliable mounting/scene-axis transform for these Mid-360 acquisitions was found.

The shared pre-ICP overlap proxy is the fraction of source points with one target
neighbor within the existing 0.50 m association distance at `T_initial`.
`LOW_INITIAL_OVERLAP=true` means fraction <= 0.01 and is explanatory only; it
does not change registration. Recovery remains translation <= 5 mm and rotation
<= 0.2 degree.

Non-monotonic profiles are marked ambiguous and are not forced into bisection.
Monotonic success/failure brackets are refined to <=25 mm or at most eight
iterations. Results without failure through 6.4 m are right-censored lower
bounds, not capture radii.

10 snapshots from one scene/station are repeated within-sequence observations,
not 10 independent scenes.
