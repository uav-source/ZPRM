# Phase B — Cross-Backend Scene-Effect Survival Test

This is a small Development-only signal-survival experiment. It asks whether a correct initial transform still receives a nonzero point-to-plane correction after the map and scan are sampled independently, and whether weak corridor/wall geometry produces larger corrections than a geometry-rich room in both frozen backends.

## Frozen design

The two and only two conditions are `INDEPENDENT_NOISE_FREE` and `FULL_NOISE`. Both independently rule-sample the target map and source scan. The first has zero noise and dropout. The second uses scan/map Gaussian sigma `0.003/0.001 m` and scan/map dropout `0.01/0.00`. The initial transform is exactly the reference transform.

Seven synthetic scenes, Development geometry seeds `1850310744`, `1957656152`, and `1334931069`, measurement seed `217775206`, and repeat zero produce 42 shared snapshots. Frozen Open3D and PCL point-to-plane each consume the same 42 snapshots, for 84 trials. Native has zero trials.

The authoritative scene implementation and its import closure are copied byte-for-byte from source commit `89f46dda68e9ff5c71f078f6d13fc9050d58f0f5`. A local wrapper is separately hashed. Before Phase B generation, that export must reproduce the seven geometry-seed-zero Phase A anchors, including source, target, reference, parent-index, parent-point, and aggregate snapshot checksums.

The trial result remains the frozen 26-field `phase_a_trial_result_v1` contract. A narrowly scoped Phase B adapter authorizes only the two Phase B condition values; it validates every other field by passing a copied payload with condition temporarily normalized to `IDEAL_MATCHED` through the frozen Phase A strict validator. Stored results retain their true Phase B condition.

## Frozen signal gates

For each condition/backend/scene the three geometry-seed translation and rotation updates are reported with min, median, and max; no confidence interval is claimed for `n=3`. Cross-backend order uses `scipy.stats.spearmanr` over the seven translation medians. Both condition coefficients must be at least `0.50`, and one must be at least `0.70`; a nonfinite coefficient fails.

Ascending average-tie ranks must put `GEOMETRY_RICH_ROOM` at rank `<=2` and at least one of `LONG_CORRIDOR` or `PARALLEL_WALLS` at rank `>=5` for every condition/backend. For each condition, the first candidate in `[LONG_CORRIDOR, PARALLEL_WALLS]` that passes for both backends is selected. Each backend requires weak/rich ratio `>=2.0` and absolute difference `>=0.001 m`. A rich median `<=1e-12` gives a null ratio and fails. The selected weak scene must exceed the rich room for at least two of three geometry seeds in each backend.

Engineering completeness, cross-backend ranking, scene ranking, weak/rich effect, and seed consistency must all pass for `PHASE_B_SIGNAL_PASS=true`. That result may authorize only design of a later full synthetic Development protocol. A full synthetic run, Confirmatory work, real data, and paper-mainline use remain false regardless of the result.
