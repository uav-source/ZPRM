# Synthetic Confirmatory Protocol v1

This preregistration is design-only. It creates no point clouds and authorizes no run.

- Planned snapshots: **595**
- Planned trials: **1190**
- Backends: Open3D and PCL; Native is forbidden.
- Conditions: IDEAL_MATCHED, INDEPENDENT_NOISE_FREE, FULL_NOISE.
- `CONFIRMATORY_RUN_AUTHORIZED = false`

INDEPENDENT_NOISE_FREE has one input per scene × geometry seed and is not represented as repeated stochastic evidence. FULL_NOISE has 3 measurement seeds × 5 repeats.

## Frozen hypotheses

### H1_IDEAL_CONTROL

```json
{
  "backends": [
    "open3d_point_to_plane",
    "pcl_point_to_plane"
  ],
  "nonfinite_output_count_max": 0,
  "quantile_method": "linear",
  "rotation_q95_max_deg": 0.01,
  "rotation_q95_max_rad": 0.00017453292519943296,
  "solver_failure_count_max": 0,
  "translation_q95_max_m": 0.001
}
```

### H2_LONG_CORRIDOR_SCENE_EFFECT

```json
{
  "backends": [
    "open3d_point_to_plane",
    "pcl_point_to_plane"
  ],
  "comparison": "LONG_CORRIDOR_vs_GEOMETRY_RICH_ROOM",
  "conditions": [
    "INDEPENDENT_NOISE_FREE",
    "FULL_NOISE"
  ],
  "geometry_block_count": 5,
  "geometry_level_median_absolute_difference_min_m": 0.005,
  "geometry_level_median_ratio_min": 5.0,
  "minimum_long_greater_than_rich_blocks": 4
}
```

### H3_CROSS_BACKEND_SCENE_RANKING

```json
{
  "conditions": [
    "INDEPENDENT_NOISE_FREE",
    "FULL_NOISE"
  ],
  "scene_count": 7,
  "spearman_rho_min": 0.7
}
```

### H4_REASSOCIATION_MECHANISM

```json
{
  "backends": [
    "open3d_point_to_plane",
    "pcl_point_to_plane"
  ],
  "causal_claim_authorized": false,
  "direction_stability_rule": "ALL_FIVE_HELD_OUT_RHO_STRICTLY_POSITIVE",
  "leave_one_geometry_seed_out_fold_count": 5,
  "pooled_turnover_error_spearman_rho_min": 0.4,
  "scene_centered_spearman_rho_min": 0.2
}
```

### H5_FROZEN_MODEL_B_INCREMENTAL_VALUE

```json
{
  "alpha_change_authorized": false,
  "backends": [
    "open3d_point_to_plane",
    "pcl_point_to_plane"
  ],
  "feature_change_authorized": false,
  "mae_b_to_mae_a_ratio_max": 0.9,
  "refit_authorized": false,
  "restandardization_authorized": false
}
```

### H6_FULL_NOISE_SYSTEMATIC_OFFSET

```json
{
  "backends": [
    "open3d_point_to_plane",
    "pcl_point_to_plane"
  ],
  "condition": "FULL_NOISE",
  "effective_replicate_count_min": 12,
  "geometry_group_count": 5,
  "median_systematic_fraction_min": 0.7,
  "minimum_passing_geometry_groups": 4,
  "planned_replicates_per_geometry": 15,
  "scene_variant": "LONG_CORRIDOR",
  "systematic_fraction_min": 0.6,
  "systematic_translation_offset_min_m": 0.005
}
```

## Frozen models

Four Development-trained Ridge models use unique-input/condition-balanced weights. Scalers, coefficients, intercepts, feature order, alpha, data SHA, and code SHA are locked. Confirmatory refitting and restandardization are forbidden.

## Interpretation boundary

Model B is a post-registration explanatory model. Association findings are not causal claims.

