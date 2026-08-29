# ZPRM FMB1 frozen paper data figures v1

This directory contains seven paper-candidate data figures generated only from frozen scientific analysis and final geometry sources. The figure-generation base commit is `81f4b566252d09ec5d055bc97c7daa964983ddc8`; the self commit is resolved by annotated tag `figures/fmb1-paper-data-figures-v1` after commit creation. Scientific source tags are `results/fmb1-zero-perturbation-locked-analysis-v1`, `correction/fmb1-solver-convergence-reporting-v1`, and `diagnostic/fmb1-solver-convergence-v1`.

No registration or ICP backend was run. No scientific analysis, p-value, Spearman rho, exact allocation, or 10,000-draw permutation was rerun. No new hypothesis test was created, no outlier or counter-directional observation was removed, and no frozen result was modified. Scene is the highest-level independent unit; stations and snapshots are nested repeated observations.

ZPRU means zero-perturbation registration update, where zero perturbation means zero intentionally imposed initialization perturbation. It must not be interpreted as a physical displacement measure. Reassociation is reported only as an association. Cross-backend panels report implementation-level agreement under identical inputs and initialization conditions.

Solver/accounting wording comes only from reporting correction v1:

- 360/360 formal COMPLETED.
- PCL native `has_converged = 180/180`.
- Open3D native stopping criterion is not retrospectively observable.

The frozen checksum roots passed before plotting: `True`. Run:

```bash
python3 paper/figures/zprm_fmb1_v1/scripts/generate_all_figures.py
python3 paper/figures/zprm_fmb1_v1/scripts/validate_paper_figures.py
```

Outputs are written as vector PDF/SVG and 600 dpi PNG/TIFF (LZW). The actual font is `Times New Roman` with PDF Type 42 settings. `audit/source_file_manifest.json`, `audit/source_schema_audit.json`, `audit/figure_provenance_manifest.json`, and the directory-level `SHA256SUMS` provide the audit trail.
