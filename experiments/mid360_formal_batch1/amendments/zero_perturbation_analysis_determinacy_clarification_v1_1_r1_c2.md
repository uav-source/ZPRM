# FMB1 Zero-Perturbation analysis determinacy clarification v1.1-R1-C2

Status: `ACTIVE_BLINDED_POSTRUN_PRE_ANALYSIS_CLARIFICATION`

Phase: `BLINDED_POSTRUN_PRE_LOCKED_SCIENTIFIC_ANALYSIS`

This is an additive, versioned clarification triggered only by the preexisting determinacy audit at commit `21038ee1fd0b59b7edb5ab730d4683e4a9fe5170`. All 360 formal registrations and the independent Post-run integrity verification are complete. The Post-run verifier necessarily machine-read raw trial values for integrity checking, but C2 does not use those values, scene/Weak/Rich aggregates, effect directions, correlations, or p-values as inputs. No scientific aggregation, Weak/Rich comparison, or formal p-value was computed while selecting these definitions.

C2 is **not** preregistration and does not claim to precede formal registration. It is a blinded post-run clarification made before locked scientific analysis. It does not authorize analysis or backend execution.

## Preserved byte-frozen authority

C2 does not modify the C1 analysis contract, C1 analysis protocol, C1 missingness clarification, original determinacy audit, dataset, backend parameters, primary translation endpoint, primary exact 20-allocation inference, or secondary rotation endpoint.

## 1. Scene-ordering agreement

Use `PAIRWISE_SIGN_ORDER_AGREEMENT_ACROSS_SIX_SCENES` on the Open3D and PCL primary translation scene medians in fixed order R01, R02, R03, W01, W02, W03. Classify all 15 unordered pairs as concordant non-tie, discordant, both tied, or one-backend tied. Report all four counts, `(concordant_non_tie + both_tied)/15`, and a strict match that is true exactly when discordant and one-backend-tied counts are both zero. This is descriptive only, has no p-value, and does not replace six-scene Spearman.

## 2. Formal reassociation turnover

The formal primary turnover endpoint is `correspondence_turnover`, because it directly measures source-target correspondence-pair reassignment. Every unqualified use of turnover maps to this field. `accepted_source_turnover` remains required as `SECONDARY_DESCRIPTIVE_COMPANION`; it cannot replace the primary endpoint or be selected because it gives a larger correlation.

## 3. Within-scene centered association

For each backend, let `x=correspondence_turnover` and `y=translation_norm_m`. Within each scene, subtract the respective scene **median** from both x and y. Pool the resulting 30 × 6 centered pairs and report `scipy.stats.spearmanr` rho with average ranks as `FORMAL_CENTERED_ASSOCIATION_EFFECT_SIZE`. The snapshots are not claimed as independent scenes, and SciPy's asymptotic p-value is not formal inference. Incomplete 180-pair coverage or constant centered input yields the frozen null status and prevents the registered permutation.

## 4. Registered stratified sensitivity permutation

When centered Spearman is defined, use `abs(rho)` as the two-sided statistic. For each backend independently initialize `numpy.random.Generator(numpy.random.PCG64(20260820))`. For exactly 10,000 draws, hold centered turnover fixed and independently permute the 30 centered displacement values within each scene in fixed R01, R02, R03, W01, W02, W03 order. Never permute across scenes or deduplicate tied draws. Report `(1 + count(abs(rho_perm) >= abs(rho_obs))) / 10001` as `SECONDARY_STRATIFIED_MONTE_CARLO_PERMUTATION_SENSITIVITY`. This is distinct from the unchanged primary exact 20-allocation test.

## 5. Systematic Weak/Rich comparison

The station and scene systematic-fraction definitions remain unchanged. With all six scene fractions defined, separately by backend report the three Rich values, three Weak values, each group median, and `median(Weak)-median(Rich)`. Its role is `SECONDARY_DESCRIPTIVE_MECHANISTIC_COMPARISON`: no formal hypothesis test, p-value, directional pass/fail, permutation, Mann-Whitney test, or t-test is permitted. C2 deliberately does not add confirmatory inference that C1 failed to freeze before formal execution.

## Unchanged primary and missingness rules

The primary endpoint remains `translation_norm_m`; each scene uses 30 snapshots with median/q25/q75/q95 and linear quantiles; scene remains the independent unit; the estimand remains median of three Weak scene medians minus median of three Rich scene medians; and the p-value remains the exact one-sided count over all 20 three-versus-three scene allocations divided by 20 without a +1 correction. Rotation remains secondary under the same scene-first exact procedure. C1 completeness gates, null statuses, imputation/winsorization bans, and available-case restrictions remain unchanged.
