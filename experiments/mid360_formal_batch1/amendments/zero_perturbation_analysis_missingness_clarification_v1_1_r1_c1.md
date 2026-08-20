# FMB1 Zero-Perturbation v1.1-R1-C1 missingness clarification

Status: `ACTIVE_PRELOCK_CLARIFICATION`  
Clarification ID: `FMB1_ZERO_PERTURBATION_MISSINGNESS_CLARIFICATION_V1_1_R1_C1`  
Formal trial count at clarification: `0`  
Formal lock issued at clarification: `false`

This clarification was frozen before formal lock, ICP, or any real backend result. It resolves how scientific nonfinite/null endpoints and unresolved infrastructure failures affect nested summaries and inference. It does not change the dataset, Identity initialization, backend parameters, physical-reference semantics, or preserved capture-radius track.

## No silent exclusion

Every one of the 360 planned trial IDs remains in accounting. A row is classified endpoint-by-endpoint as a finite scientific value, a scientific undefined/nonfinite outcome, an unresolved infrastructure failure, or—for a secondary common metric—an endpoint-specific undefined value. Solver non-convergence remains a finite scientific value whenever infrastructure passed and the relevant endpoint is finite.

Nonfinite JSON tokens are forbidden. A scientific nonfinite outcome is represented by null endpoint fields plus an explicit nonfinite status. It is an authoritative scientific failure and cannot be rerun. Infrastructure attempts may be retried only under the existing infrastructure policy; every attempt remains in lineage, and the earliest schema-valid infrastructure-PASS scientific outcome is authoritative. If none exists, the planned trial remains an unresolved infrastructure failure.

Every station/scene/backend summary reports planned N, authoritative scientific N, finite endpoint N, scientific undefined/nonfinite N, unresolved infrastructure N, resolved infrastructure-attempt N, solver-nonconverged finite N, endpoint-specific undefined N, and a formal-summary status.

## Formal summaries use a complete-coverage gate

Formal station translation and rotation summaries require 10 of 10 finite endpoints, zero scientific nonfinite outcomes, and zero unresolved infrastructure failures. Formal scene summaries require 30 of 30 finite endpoints and all three defined station summaries. If the gate fails, formal median/q25/q75/q95 are null with an explicit incomplete-coverage status.

Finite available-case median and quantiles may be reported when at least one finite value exists, but only as `AVAILABLE_CASE_DESCRIPTIVE_NOT_PRIMARY_NOT_INFERENTIAL`, always with their N and missingness counts. They cannot replace a formal summary, enter the primary estimand, or rescue inference. No imputation, winsorization, or result-dependent threshold change is allowed.

## Permutation and Spearman behavior

The exact Weak-versus-Rich test runs only when all six formal scene summaries—three Rich and three Weak—are defined. Otherwise the estimand and p-value are null with `INFERENCE_UNDEFINED_INCOMPLETE_SIX_SCENE_COVERAGE`; partial-scene permutation, rebalancing, and imputation are forbidden.

Formal cross-backend scene Spearman requires six complete paired scene summaries. Formal station Spearman requires 18 complete paired station summaries. Incomplete pairs make formal rho null; available-pair correlation is descriptive only. Constant input makes rho and p-value null with `SPEARMAN_UNDEFINED_CONSTANT_INPUT`. Implement using SciPy `spearmanr` average ranks and always report complete-pair and undefined counts.

Per-snapshot direction cosine is descriptive. It is defined only when both backend translation vectors are finite and nonzero. A missing/nonfinite or zero vector produces a null cosine with its explicit status, and the snapshot remains counted.

## Reassociation and systematic fraction

Reassociation missingness is endpoint-specific: a missing common metric does not invalidate a finite pose endpoint. Every formal result must preserve `common_association_valid`, `common_association_invalid_reason`, and `common_association_invalid_detail`. Invalid reason is one of `NO_INITIAL_CORRESPONDENCE`, `NO_FINAL_CORRESPONDENCE`, `INSUFFICIENT_VALID_NORMALS`, `NONFINITE_COMMON_METRICS`, or `OTHER`; detail is bounded diagnostic text, required for `OTHER`. A valid row has null reason/detail. The result schema, runner, and verifier must retain these fields before formal lock.

A formal scene metric median requires all 30 metric values. Scene-level turnover/displacement Spearman requires all six paired scene summaries. The registered within-scene centered association requires 30 complete pairs in every scene and 180 total; otherwise the formal association and its stratified permutation are null. Available-case centered association may appear only descriptively with N and missingness counts.

A formal station systematic fraction requires all ten finite translation vectors. If any vector is undefined, the fraction is null with `SYSTEMATIC_FRACTION_UNDEFINED_INCOMPLETE_10_VECTORS`. If all ten vectors are finite and all have zero norm, retain the station and define the fraction as 0 with `ALL_ZERO_UPDATES_DEFINED_ZERO_SYSTEMATIC_FRACTION`. A scene requires all three station fractions; Weak/Rich comparison requires all six scene values.

## Reporting

Undefined formal results must be printed, not omitted. Reports list affected trial IDs, backend-specific scientific-nonfinite counts, infrastructure counts, and visibly separate formal complete-coverage summaries from available-case descriptions. This clarification preserves `NOMINAL_IDENTITY_NO_OBVIOUS_MOTION_NOT_SUBMILLIMETER_GT` and the capture-radius status `PRESERVED_SUPPLEMENTARY_NOT_EXECUTED`.
