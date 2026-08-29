# Frozen source schema audit

All columns below were read directly from the frozen CSV headers. An empty unique-value list means that the named field is absent, not inferred.

## `results/mid360_formal_batch1/final_dataset_v1/final_geometry_manifest.csv`

- Role: Fig05 snapshot geometry
- Rows: 180
- Columns: `scene_id, station_id, selection_index, snapshot_id, initial_correspondence_count, initial_valid_normal_correspondence_count, lambda_min_trans, lambda_mid_trans, lambda_max_trans, normalized_lambda_min_trans, normalized_lambda_mid_trans, normalized_lambda_max_trans, condition_number_trans, spectral_entropy_trans, attempt`
- Scene IDs: `['FMB1_R01', 'FMB1_R02', 'FMB1_R03', 'FMB1_W01', 'FMB1_W02', 'FMB1_W03']`
- Station IDs: `['S01', 'S02', 'S03']`
- Backends: `[]`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `[]`
- attempt: `[1, 2]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/translation_scene_summaries.csv`

- Role: Fig11/Fig12 translation scene summaries
- Rows: 12
- Columns: `authoritative_scientific_outcome_n, available_case, backend, defined_station_summary_n, endpoint, endpoint_specific_undefined_n, finite_endpoint_n, formal_statistics, formal_summary_status, planned_n, resolved_infrastructure_attempt_n, scene_id, scene_statistics_are_direct_30_snapshot_statistics, scientific_undefined_nonfinite_n, solver_nonconverged_finite_n, station_medians, station_summaries, undefined_trial_ids, unresolved_infrastructure_failure_n`
- Scene IDs: `['FMB1_R01', 'FMB1_R02', 'FMB1_R03', 'FMB1_W01', 'FMB1_W02', 'FMB1_W03']`
- Station IDs: `[]`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `['translation_norm_m']`
- formal_summary_status: `['FORMAL_SCENE_SUMMARY_DEFINED_COMPLETE_30_OF_30']`
- status: `[]`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/translation_station_summaries.csv`

- Role: Fig11 translation station summaries
- Rows: 36
- Columns: `authoritative_scientific_outcome_n, available_case, backend, endpoint, endpoint_specific_undefined_n, finite_endpoint_n, formal_statistics, formal_summary_status, planned_n, resolved_infrastructure_attempt_n, scene_id, scientific_undefined_nonfinite_n, solver_nonconverged_finite_n, station_id, undefined_trial_ids, unresolved_infrastructure_failure_n`
- Scene IDs: `['FMB1_R01', 'FMB1_R02', 'FMB1_R03', 'FMB1_W01', 'FMB1_W02', 'FMB1_W03']`
- Station IDs: `['S01', 'S02', 'S03']`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `['translation_norm_m']`
- formal_summary_status: `['FORMAL_STATION_SUMMARY_DEFINED_COMPLETE_10_OF_10']`
- status: `[]`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/translation_exact_permutations.csv`

- Role: Fig12 frozen exact allocations
- Rows: 40
- Columns: `assigned_rich_scene_ids, assigned_weak_scene_ids, backend, greater_than_or_equal_observed, observed_allocation, statistic`
- Scene IDs: `[]`
- Station IDs: `[]`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `[]`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/cross_backend_scene_pairs.csv`

- Role: Fig13 frozen scene ordering pairs
- Rows: 15
- Columns: `classification, open3d_sign, pcl_sign, scene_i, scene_j`
- Scene IDs: `[]`
- Station IDs: `[]`
- Backends: `[]`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `[]`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/cross_backend_station_pairs.csv`

- Role: Fig13 station value pairs
- Rows: 18
- Columns: `complete, identifier, open3d_value, pcl_value`
- Scene IDs: `[]`
- Station IDs: `[]`
- Backends: `[]`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `[]`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/snapshot_direction_cosines.csv`

- Role: Fig13 snapshot direction cosines
- Rows: 180
- Columns: `cosine, snapshot_id, status`
- Scene IDs: `[]`
- Station IDs: `[]`
- Backends: `[]`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `['DIRECTION_COSINE_DEFINED']`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/reassociation_scene_summaries.csv`

- Role: Fig14 reassociation scene summaries
- Rows: 12
- Columns: `backend, finite_metric_n, formal_turnover_field, median, scene_id, status`
- Scene IDs: `['FMB1_R01', 'FMB1_R02', 'FMB1_R03', 'FMB1_W01', 'FMB1_W02', 'FMB1_W03']`
- Station IDs: `[]`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `['REASSOCIATION_SCENE_SUMMARY_DEFINED_COMPLETE_30_METRICS']`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/reassociation_centered_rows.csv`

- Role: Fig14 frozen centered rows
- Rows: 360
- Columns: `backend, centered_correspondence_turnover, centered_translation_norm_m, scene_id, scene_row_index`
- Scene IDs: `['FMB1_R01', 'FMB1_R02', 'FMB1_R03', 'FMB1_W01', 'FMB1_W02', 'FMB1_W03']`
- Station IDs: `[]`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `[]`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/reassociation_centered_permutation_draws.csv`

- Role: Fig14 frozen permutation draws
- Rows: 20000
- Columns: `abs_rho_permuted, backend, draw_index, greater_than_or_equal_abs_observed, rho_permuted`
- Scene IDs: `[]`
- Station IDs: `[]`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `[]`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/systematic_scene_values.csv`

- Role: Fig15 systematic scene values
- Rows: 12
- Columns: `backend, defined_station_n, scene_id, scene_systematic_fraction, status`
- Scene IDs: `['FMB1_R01', 'FMB1_R02', 'FMB1_R03', 'FMB1_W01', 'FMB1_W02', 'FMB1_W03']`
- Station IDs: `[]`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `['SCENE_SYSTEMATIC_FRACTION_DEFINED_COMPLETE_3_STATIONS']`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/systematic_station_values.csv`

- Role: Fig15 systematic station values
- Rows: 36
- Columns: `backend, denominator, finite_vector_n, mean_vector, planned_n, scene_id, station_id, status, systematic_fraction`
- Scene IDs: `['FMB1_R01', 'FMB1_R02', 'FMB1_R03', 'FMB1_W01', 'FMB1_W02', 'FMB1_W03']`
- Station IDs: `['S01', 'S02', 'S03']`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `[]`
- formal_summary_status: `[]`
- status: `['SYSTEMATIC_FRACTION_DEFINED_COMPLETE_10_VECTORS']`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/rotation_scene_summaries.csv`

- Role: FigS01 rotation scene summaries
- Rows: 24
- Columns: `authoritative_scientific_outcome_n, available_case, backend, defined_station_summary_n, endpoint, endpoint_specific_undefined_n, finite_endpoint_n, formal_statistics, formal_summary_status, planned_n, resolved_infrastructure_attempt_n, scene_id, scene_statistics_are_direct_30_snapshot_statistics, scientific_undefined_nonfinite_n, solver_nonconverged_finite_n, station_medians, station_summaries, undefined_trial_ids, unresolved_infrastructure_failure_n`
- Scene IDs: `['FMB1_R01', 'FMB1_R02', 'FMB1_R03', 'FMB1_W01', 'FMB1_W02', 'FMB1_W03']`
- Station IDs: `[]`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `['rotation_angle_rad', 'rotation_angle_deg']`
- formal_summary_status: `['FORMAL_SCENE_SUMMARY_DEFINED_COMPLETE_30_OF_30']`
- status: `[]`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/rotation_station_summaries.csv`

- Role: FigS01 rotation station summaries
- Rows: 72
- Columns: `authoritative_scientific_outcome_n, available_case, backend, endpoint, endpoint_specific_undefined_n, finite_endpoint_n, formal_statistics, formal_summary_status, planned_n, resolved_infrastructure_attempt_n, scene_id, scientific_undefined_nonfinite_n, solver_nonconverged_finite_n, station_id, undefined_trial_ids, unresolved_infrastructure_failure_n`
- Scene IDs: `['FMB1_R01', 'FMB1_R02', 'FMB1_R03', 'FMB1_W01', 'FMB1_W02', 'FMB1_W03']`
- Station IDs: `['S01', 'S02', 'S03']`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `['rotation_angle_rad', 'rotation_angle_deg']`
- formal_summary_status: `['FORMAL_STATION_SUMMARY_DEFINED_COMPLETE_10_OF_10']`
- status: `[]`
- attempt: `[]`

## `results/mid360_formal_batch1/zero_perturbation_locked_analysis_results_v1/rotation_exact_permutations.csv`

- Role: FigS01 frozen exact allocations
- Rows: 80
- Columns: `assigned_rich_scene_ids, assigned_weak_scene_ids, backend, endpoint, greater_than_or_equal_observed, observed_allocation, statistic`
- Scene IDs: `[]`
- Station IDs: `[]`
- Backends: `['OPEN3D_POINT_TO_PLANE', 'PCL_POINT_TO_PLANE']`
- Endpoints: `['rotation_angle_rad', 'rotation_angle_deg']`
- formal_summary_status: `[]`
- status: `[]`
- attempt: `[]`
