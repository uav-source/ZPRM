# 图注（中文）

## Fig. 5. Rich/Weak 几何特征

各 panel 分别展示六个预设场景的 (a) 归一化最小平移特征值、(b) 平移条件数和 (c) 谱熵。小空心点为每个场景 30 个冻结 snapshot 值，菱形为 final scene registry 中冻结的场景中位数。本图记录正式数据集中预设的粗粒度几何分类；W02 为 acquisition attempt 2。本图是描述性几何展示，不表示 Weak geometry 必然对应更大的 zero-perturbation registration update（ZPRU）。

## Fig. 11. 六场景 Translation ZPRU

(a) Open3D，(b) PCL。细区间为冻结 q25–q95，粗区间为 q25–q75，大实心点为场景中位数，小空心点为三个嵌套 station 的中位数；数值仅为显示而由 m 换算为 mm。station 和 snapshot 是嵌套重复观测，scene 始终是最高层级独立单位。这些是 summary intervals，不是 Tukey boxplot；本图不宣称 Rich/Weak 的方向性排序。

## Fig. 12. Rich/Weak 场景比较与 exact permutation

(a, c) 分别显示 Open3D 和 PCL 的六个冻结场景中位数，横线标出各预设组中三个场景中位数的中位数。(b, d) 显示全部 20 个冻结 exact allocation statistics 与冻结 observed statistic。Open3D：Weak − Rich = −0.026347782 mm，exact one-sided p = 0.7；PCL：Weak − Rich = −0.010915495 mm，exact one-sided p = 0.7。scene 是最高层级独立单位。未预设显著性阈值，exact one-sided p-value 仅作描述性报告。

## Fig. 13. Cross-backend agreement

(a) 6 个 scene-level translation ZPRU 配对，(b) 18 个 station-level 配对，(c) 180 个 snapshot-level update-direction cosine 的 ECDF。(a, b) 的 identity line 仅是绝对一致性参考；Spearman rho 描述秩一致性，不能证明数值幅度相等。冻结的 scene/station rho 分别为 1.000 和 0.9917。(c) 的冻结中位数为 0.997739005；180 个值均已定义，zero-vector 与 missing/nonfinite 均为 0。两个独立实现的 backend 使用完全相同的输入点云和初始化条件，因此本图评估 implementation-level agreement，不评估独立测量系统之间的重复性。scene 仍是最高层级独立单位。

## Fig. 14. 与 correspondence reassociation 的关联

(a, b) 每个 backend 的 6 个 scene-median correspondence turnover 与 scene-median translation ZPRU，并标注冻结的描述性 Spearman rho 和 p。(c, d) 每个 backend 的 180 个冻结 within-scene median-centered rows；inset 展示既有 10,000 个分层 permutation |rho| draws 的 ECDF 和冻结 observed |rho|。图中不拟合回归线或因果模型。turnover 与 update magnitude 都依赖估计的 terminal pose；观察到的关系是 association，不能把 correspondence reassociation 确立为独立因果决定因素。scene 是最高层级独立单位，180 行是嵌套观测而不是独立场景。

## Fig. 15. Systematic component

(a) Open3D，(b) PCL。小空心点为每个 scene 的 3 个 station fraction，大实心点为冻结 scene fraction，横线为冻结 Rich/Weak 组中位数。冻结的 Weak − Rich 描述性差值分别为 −0.154058605 和 −0.301378567。该图为 secondary descriptive mechanistic comparison；未指定 formal p-value。scene 是最高层级独立单位。

## Fig. S1. Secondary rotational ZPRU

(a) Open3D，(b) PCL，单位为 deg。glyph 与 Fig. 11 一致：冻结 q25–q95、q25–q75 summary intervals、scene median 和 3 个嵌套 station medians。Open3D：Weak − Rich = −0.003238022 deg，exact p = 0.9；PCL：Weak − Rich = −0.001570443 deg，exact p = 0.8。这是 secondary endpoint，不提升为 co-primary。scene 是最高层级独立单位。
