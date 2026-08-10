# Real-data Validation 预注册准备总结

1. IILABS 3D：已实际启动可续传下载，但截至冻结时未全部完成；GT 与标定已下载并校验。
2. GrandTour：参考/TF metadata 与 SPX-1/SPX-3 IE-TC、NavSatFix、prism、TF 已按固定 HF revision 选择性下载且 LFS SHA 全匹配；Hesai 大点云因 GT-only overlap 先失败而未下载。
3. 候选 acquisition 为 IILABS nav_a_diff→nav_a_omni，以及 GrandTour SPX-1→SPX-3 / SPX-3→SPX-1；没有 mission pair 被最终冻结。
4. IILABS 两条 TUM 均被后处理重置到零；没有官方固定跨序列变换，故未证明共享固定世界系。
5. GrandTour SPX-1/SPX-3 已取得公开独立 IE-TC 6DoF reference；SPX-2 reference 官方明确隐藏。
6. GT-only overlap 未通过：SPX-1→SPX-3 为 78.0 s/0.342105，反向为 27.0 s/0.150838；IILABS 不可计算。
7. 两个数据集均未冻结 50 weak + 50 rich；inventory 为零行，未伪造。
8. 若资格通过，盲选主指标固定为 normalized_lambda_min_trans；本次未执行标签选择。
9. 没有查看或生成任何 registration error；registration-derived field access count=0。
10. 尚无 canonical bundle，不能声称 Open3D/PCL 已绑定字节相同输入。
11. R01 PASS；R02 PASS；R03/R04/R05/R10 FAIL；R06/R07/R08 BLOCKED；R09 PASS。
12. R14 未完成，不可变成功冻结未通过；仅冻结了可验证的失败审计。
13. Open3D registration、PCL CLI、其他 registration 与真实 trial 调用次数全部为 0。
14. 数据绝对路径：/home/lj/zero_perturbation_data/real_data_v1；失败审计 manifest 绝对路径：/home/lj/zero_perturbation_runtime/real_data/preparation_v1。
15. 不具备单独授权正式 400-trial 运行的条件。

## 最终结论

`PREREGISTRATION_NOT_READY`

硬阻塞包括：IILABS 跨 acquisition 固定世界系尚未由官方证据证明；GrandTour SPX-1↔SPX-3 在 WGS84/ECEF 下的 GT-only overlap 双向均未达到冻结阈值；因此未下载 Hesai 大点云、未建图、未盲选、未生成伪造的 200-row inventory。

保持 `REAL_DATA_RUN_AUTHORIZED=false`、`MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false`、`registration_execution_count=0`。

