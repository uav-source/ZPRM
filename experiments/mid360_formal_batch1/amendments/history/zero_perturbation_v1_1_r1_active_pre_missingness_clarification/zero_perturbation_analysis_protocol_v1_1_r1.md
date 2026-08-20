# FMB1 Zero-Perturbation Analysis Protocol v1.1-R1

This protocol is `ACTIVE` under amendment `FMB1_ZERO_PERTURBATION_MAINLINE_V1_1_R1`. It was activated after independent candidate verification and final-dataset prelock reauthentication at `actual_formal_trials=0`. It has no backend execution authority, does not activate the capture-radius track, and leaves `FORMAL_ICP_UNLOCKED=false` and `FORMAL_REGISTRATION_AUTHORIZED=false`.

## Scope and physical semantics

The analysis concerns the pose update returned from independently recorded same-station MAP and QUERY data when the initial transform is exactly Identity. The common formula is `Delta_T = inverse(T0) @ T_est`, with both `T0` and the nominal reference equal to Identity. The primary translation value is `norm(Delta_T[:3,3])`; the secondary rotation value is the principal SO(3) angle of `Delta_T[:3,:3]`.

Every result must carry `physical_reference_semantics=NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT`. The platform had no intentional motion and was mechanically supported, but IMU screening establishes only `NO_OBVIOUS_MOTION`. There is no independent submillimeter external ground truth. These results are controlled-static nominal-identity registration updates, not metrology-validated absolute physical displacements. No paper or report may claim that the IMU proved physical displacement below 1 mm.

## Frozen dataset and hierarchy

The Rich scenes are R01, R02, and R03; the Weak scenes are W01, W02, and W03. W02 attempt 1 is an archived wrong-location invalid acquisition. W02 attempt 2 is the admitted Weak scene. W04 is retired and forbidden from the active dataset, trial plan, analysis, and results.

The hierarchy is `scene -> station -> snapshot -> backend`. Scene is the primary and highest independent unit. There are six scenes, three stations per scene, ten snapshots per station, and 30 snapshots per scene. The 180 snapshots are nested repeats, not 180 independent scenes. Snapshot-level naive p-values and any pooled analysis that presents 180 independent scenes are forbidden.

Open3D point-to-plane and PCL point-to-plane are analyzed separately under the unchanged backend contract SHA256 `6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9`.

## Primary translation analysis

For each backend and scene, report the 30-snapshot translation-norm median, q25, q75, and q95 using the linear quantile method, plus the three station medians. Report all six raw scene medians.

The primary estimand is:

```text
Delta_translation =
median(the three Weak scene medians)
- median(the three Rich scene medians)
```

Use an exact one-sided permutation test over every three-versus-three allocation of the six scene medians. There are exactly 20 allocations. The statistic for every allocation is the assigned-Weak median minus the assigned-Rich median, and the p-value is the number of permutation statistics greater than or equal to the observed statistic divided by 20. The alternative is Weak greater than Rich. No random seed or Monte Carlo correction applies to this exact enumeration. The conclusion is not forced to pass.

## Secondary rotation analysis

Use the same scene-first summaries and exact 20-allocation procedure for the principal rotation angle, reporting radians and degrees. Rotation remains secondary.

## Cross-backend analysis

Report the Open3D/PCL Spearman rho for the six paired scene medians and for the 18 paired station medians. Report per-snapshot translation-direction cosine only descriptively. If either vector is zero, retain the snapshot and report a null cosine with an explicit zero-vector status. Also report backend-specific solver/nonfinite counts and scene-ordering agreement. A single pooled 180-row correlation is not sufficient and may appear only as an auxiliary description.

## Reassociation analysis

Correspondence reassociation must reuse `src/phase_a_harness/common_association_analysis.py`, frozen at candidate SHA256 `458190c26d8640353004663474adb64305d1193d89ff0dc68f71f404de8f175d`. Mid-360-specific redefinitions are forbidden. Preserve initial/final correspondence and valid-normal counts, correspondence turnover, accepted-source turnover, correspondence-count change ratio, initial/final residual RMSE, residual RMSE change, and median/q95 normal-angle change.

For each backend report scene-level median turnover, the Spearman association between scene-median turnover and scene-median translation displacement, and an association after subtracting the corresponding scene median from each snapshot value. A pooled snapshot association is auxiliary only. The registered sensitivity permutation shuffles centered displacement values only within each scene stratum, uses seed `20260820`, runs exactly 10,000 permutations, and reports a two-sided secondary p-value; scene strata may never be broken.

## Systematic component

For every station and backend, retain all ten translation vectors and compute:

```text
systematic_fraction =
norm(mean(delta_translation_vectors))
/ mean(norm(delta_translation_vectors))
```

If the denominator is exactly zero, define the fraction as `0.0`, label it `ALL_ZERO_UPDATES_DEFINED_ZERO_SYSTEMATIC_FRACTION`, and retain the station. Report every station value, take the scene value as the median of its three station fractions, and compare Weak and Rich at scene level separately for Open3D and PCL. Directionally inconsistent snapshots are never removed.

## Synthetic model transfer

The frozen synthetic Model A/Model B transfer status is `MODEL_TRANSFER_NOT_COMPATIBLE`: complete equality of feature names, ordering, definitions, units, preprocessing, and scale with the real FMB1 result schema has not been demonstrated. No model inference, retraining, tuning, feature change, or temporary scale adaptation is permitted. This is not a formal-lock pass gate. A future secondary analysis would require a new pre-result versioned compatibility audit; it cannot be introduced after viewing real backend results.

## Failures, retries, and reporting

Retain solver non-convergence, finite scientific failures, large displacement, backend disagreement, outliers, unexpected direction, and outcomes contrary to the Weak/Rich expectation. Only process crash, file-read corruption, result-schema write failure, or PCL executable infrastructure failure permits a retry, and every attempt must remain in lineage. A scientific result cannot be rerun until successful.

Any final report must show all six scene summaries, the nested station summaries, backend-specific failures, null/zero-denominator statuses, and the physical-reference limitation. The original capture-radius v1 protocol remains `PRESERVED_SUPPLEMENTARY_NOT_EXECUTED`; R1 neither changes it nor authorizes it.
