# CAVERS 单数据集 Stage-1 总结

1. 实际物化 53.004087 MB（含论文、代码、GT/TF/时间索引与审计元数据）。
2. 否；没有下载任何完整 LiDAR、PCD 或 MCAP 大文件。
3. 是；Zenodo exact record=19367714/version=0.0.1/revision=18，GitHub commit=74ead2bf1cffa337a1bff1b03b990819cb7ff067。
4. 完整 OptiTrack 6DoF 的 DIABLO 序列：loc_diablo_1,loc_diablo_2,loc_diablo_3,loc_diablo_4,loc_diablo_5,loc_diablo_6,loc_diablo_7,loc_diablo_8。
5. 完整 OptiTrack 6DoF 的 HANDHELD 序列：loc_handheld_1,loc_handheld_2,loc_handheld_3,loc_handheld_4；5/6 因无独立 GT 排除。
6. 固定 world/map frame 判定：FAIL；同名且未归零的 map 已观察到，但官方未发布跨 recording 的共享 calibration/session 标识或不变性保证。
7. sequence-local reset：未检测到。
8. 公开 rig→VLP16 外参：已找到。
9. 外参来自各 sequence 的 TF_STATIC/data.csv（官方 /tf_static export），精确来源和矩阵见 extrinsic audit。
10. 时间链状态：PASS_WITH_DOCUMENTED_LIMITATION；共享记录时钟可审计，但无硬件级同步。
11. hardware sync、sensor/processing delay、GT/LiDAR timestamp uncertainty 仍为 UNKNOWN。
12. R02=PASS。
13. R03=BLOCKED_STAGE1。
14. R04=FAIL。
15. R05=PASS_WITH_DOCUMENTED_LIMITATION。
16. R10=PARTIAL。
17. GT-only 最佳 pair：无。
18. covered_duration_s=NOT_COMPUTABLE。
19. coverage_fraction=NOT_COMPUTABLE。
20. eligible 5s intervals=NOT_COMPUTABLE。
21. primary=无；reserve_1=无；reserve_2=无。
22. ICP/registration 调用严格为 0。
23. 否；没有任何 >500,000,000 bytes 文件被完整下载或物化。
24. CAVERS_STAGE1_READY=false。
25. 否；Stage-1 硬门未通过，不进入 VLP-16 下载阶段。
26. 失败硬门：COMMON_WORLD_FRAME、GT_ONLY_OVERLAP、R10_UNCERTAINTY_FEASIBILITY；UNKNOWN uncertainty 没有被伪装成 0。

## 最终结论

`CAVERS_STAGE1_READY=false`

CAVERS Stage-1 未通过；失败硬门已在 cavers_stage1_eligibility.json 中冻结，不下载 VLP-16。

`weak_snapshot_count=0`，`rich_snapshot_count=0`，`snapshot_count=0`，`planned_future_trials=0`，`registration_execution_count=0`。

`REAL_DATA_RUN_AUTHORIZED=false`，`MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false`。
