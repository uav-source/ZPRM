# Figure captions (English)

## Fig. 5. Rich/Weak geometry characterization

Panels show (a) normalized minimum translational eigenvalue, (b) translational condition number, and (c) spectral entropy for the six prespecified scenes. Small open points are the 30 frozen snapshot values per scene; diamonds are the frozen scene medians from the final scene registry. The panels document the prespecified coarse geometry classification used in the formal dataset. W02 is acquisition attempt 2. These descriptive geometry panels do not imply that Weak geometry necessarily produces a larger zero-perturbation registration update (ZPRU).

## Fig. 11. Translation ZPRU across six scenes

(a) Open3D and (b) PCL translation ZPRU. Thin intervals show frozen q25–q95 summaries, thick intervals show q25–q75 summaries, large filled markers show scene medians, and small open markers show the three nested station medians. Values are converted from metres to millimetres for display. Stations and snapshots are nested repeated observations; scene is the highest-level independent unit. The intervals are summary intervals, not Tukey boxplots, and the figure does not assert an ordering between Rich and Weak scenes.

## Fig. 12. Rich/Weak scene comparison and exact permutation distribution

(a, c) The six frozen scene medians are shown separately for Open3D and PCL; horizontal bars identify the median of the three scene medians in each prespecified group. (b, d) All 20 frozen exact allocation statistics are shown with the frozen observed statistic. Open3D: Weak − Rich = −0.026347782 mm, exact one-sided p = 0.7. PCL: Weak − Rich = −0.010915495 mm, exact one-sided p = 0.7. Scene is the highest-level independent unit. No significance threshold was prespecified; the exact one-sided p-value is reported descriptively.

## Fig. 13. Cross-backend agreement

(a) Six scene-level translation ZPRU pairs, (b) 18 station-level pairs, and (c) the ECDF of 180 snapshot-level update-direction cosines. Identity lines in (a, b) are absolute-agreement references; Spearman rho describes rank agreement and does not establish equality of magnitudes. The frozen scene and station rho values are 1.000 and 0.9917, respectively. In (c), the frozen median cosine is 0.997739005; all 180 values are defined, with zero zero-vector and zero missing/nonfinite cases. The two separately implemented backends used identical input point clouds and initialization conditions; this evaluates implementation-level agreement, not repeatability across independent measurement systems. Scene remains the highest-level independent unit.

## Fig. 14. Association with correspondence reassociation

(a, b) Scene-median correspondence turnover versus scene-median translation ZPRU for six scenes per backend, with frozen descriptive Spearman rho and p. (c, d) The 180 frozen within-scene median-centered rows per backend; insets show the ECDFs of the existing 10,000 stratified permutation |rho| draws and the frozen observed |rho|. No regression line or causal model is fitted. Both turnover and update magnitude depend on the estimated terminal pose; the observed relationship is an association and does not establish correspondence reassociation as an independent causal determinant. Scene is the highest-level independent unit, and the 180 centered rows are nested observations rather than independent scenes.

## Fig. 15. Systematic component

(a) Open3D and (b) PCL systematic fractions. Small open markers show three station values per scene, large filled markers show frozen scene values, and horizontal bars show frozen Rich/Weak group medians. Frozen Weak − Rich descriptive differences are −0.154058605 and −0.301378567, respectively. Secondary descriptive mechanistic comparison; no formal p-value was specified. Scene is the highest-level independent unit.

## Fig. S1. Secondary rotational ZPRU

(a) Open3D and (b) PCL rotational ZPRU in degrees. Glyphs match Fig. 11: frozen q25–q95 and q25–q75 summary intervals, scene medians, and three nested station medians. Open3D: Weak − Rich = −0.003238022 deg, exact p = 0.9. PCL: Weak − Rich = −0.001570443 deg, exact p = 0.8. This is a secondary endpoint and is not co-primary. Scene is the highest-level independent unit.
