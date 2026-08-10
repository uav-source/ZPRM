# Boreas single-dataset Stage-1 总结

**BOREAS_STAGE1_READY=false**

1. 上一轮 CAVERS 删除 apparent=57517402 B，文件系统实测释放=57823232 B。
2. Boreas Stage-1 S3 allowlist 实际下载 158.774824 MB。
3. 否；lidar/*.bin 下载数为 0。
4. 实际审计 44 条原始 Boreas sequences（另将 bucket 中非原始 Boreas/RT prefixes 分开记录）。
5. 公开 GT 候选 31 条；13 条官方 test split 的 GT 隐藏。
6. 固定 ENU_ref=PASS；官方定义与实际全局数值范围一致。
7. sequence-local reset=未检测到。
8. 是；Applanix GT trajectory 来自 GNSS/IMU/轮速/RTX POSPac 后处理，不使用 LiDAR registration。
9. 是；T_applanix_lidar 的官方生成说明明确使用 LiDAR pointclouds 与后处理 GPS/IMU 批优化。
10. 冻结 R02 对最终 LiDAR-frame reference 的结论为 FAIL：query/calibration session 排除关系未公开，不能证明完全独立。
11. R02=FAIL。
12. R03=BLOCKED_STAGE1。
13. R04=PASS。
14. R05=PASS。
15. R09=PASS。
16. R10=PARTIAL。
17. 时间同步链=PASS_WITH_DOCUMENTED_LIMITATION；PPS/NMEA/UTC 链可审计但数值误差界 UNKNOWN。
18. 2–4 cm 的 uncertainty_type=RMSE（序列依赖，绝非 1SIGMA）。
19. orientation、Applanix/Velodyne time、外参平移/旋转、插值、deskew、map accumulation uncertainty 仍为 UNKNOWN。
20. GT-only 最优 pair=NOT_COMPUTABLE。
21. covered_duration_s=NOT_COMPUTABLE。
22. coverage_fraction=NOT_COMPUTABLE。
23. eligible_nonoverlapping_5s_intervals=NOT_COMPUTABLE。
24. reserve pair=无。
25. 否；没有执行任何 ICP/registration。
26. 是；weak/rich/snapshot/planned future trials 全部为 0。
27. BOREAS_STAGE1_READY=false。
28. 否；不进入 Stage-2，也不下载最小 LiDAR。
29. 主要硬门是 LiDAR-assisted proprietary extrinsic 使最终 LiDAR-frame reference 的 R02 independence 无法证明；R10 仍有关键 UNKNOWN，overlap 因 R02 被门控。

## 最终结论

BOREAS_STAGE1_READY=false；R02 independence 与 R10 硬门未通过，不进入 Stage-2。
