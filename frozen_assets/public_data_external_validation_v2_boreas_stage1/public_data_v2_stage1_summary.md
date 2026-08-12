# Boreas External Validation v2 Stage-1 总结

**BOREAS_EXTERNAL_V2_STAGE1_READY=true**
**READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD=true**

1. Boreas v1 的 R02=FAIL 属于已冻结的严格筛选结论；v2 使用新编号和新语义，因此不追溯修改 v1。
2. v1 原文要求 independent high-accuracy 6DoF reference；Stage-1 严格实现把 LiDAR-assisted static extrinsic 也视为独立性失败。
3. v2 要求 trajectory backbone 独立于待测 query-to-target registration，且不得由 ICP、scan matching、LiDAR odometry 或 SLAM 产生。
4. 允许历史 LiDAR-assisted fixed extrinsic，是因为它在实验前公开冻结、44 条 sequence 字节一致、未由当前 registration 估计。
5. 限制包括完整披露 provenance，date/session/query participation 与 uncertainty 保持 UNKNOWN，且只能用于 supplementary trend/generalization。
6. 外参、姿态、同步、插值、deskew 和 map accumulation 等不确定性仍含 UNKNOWN，故不能承担毫米级主真实证据。
7. 是；IILABS、GrandTour、RTS-GT、CAVERS、Boreas 五个 v1 候选已完整 closure，eligible=0。
8. 是；Public-data v1 registration count 与 ICP count 均保持 0。
9. 默认 source-only 明确 skip 并给出完整路径；strict external 模式因历史包缺失明确失败，不伪造也不报 PASS。
10. 是；source-only 为 0 failed/0 errors（811 collected, 801 passed, 10 skipped）。
11. strict external qualification=UNAVAILABLE；缺失外部历史包的单测按要求非零失败。
12. 是；84 个 Boreas v1 evidence 文件逐项重新计算 SHA，并绑定 v1 manifest、SHA256SUMS 与 76 个 receipt。
13. 否；已有 Stage-1 小文件重新下载数为 0，内容与 mtime 均未改变。
14. 否；本任务下载和物化的 lidar/*.bin 均为 0。
15. 是；31 条 public GT 中按原生 gap 排除 2 条，29 条 eligible reference sequences 均重新解析和 SHA 验证。
16. 是；GT-only overlap=PASS，全部 812 个有向 pair 中 812 个通过原门槛。
17. primary pair 为 boreas-2021-11-14-09-47 -> boreas-2021-01-26-11-22。
18. reserve pairs 为 boreas-2021-06-17-17-52 -> boreas-2021-01-26-11-22 与 boreas-2021-09-14-20-00 -> boreas-2021-01-26-11-22。
19. primary covered duration=1346.0 s。
20. primary coverage fraction=0.9525831564048125。
21. primary eligible 5 s intervals=246。
22. Stage-2 冻结 allowlist 预计下载 104.158637 GB（97.005290 GiB）。
23. 是；Open3D/PCL/其他 registration 调用与 registration execution count 全部为 0。
24. BOREAS_EXTERNAL_V2_STAGE1_READY=true。
25. 是；可以进入一个独立、另行授权的 Stage-2 LiDAR 下载任务，但本任务未授权或启动。
26. 是；MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false。

## 冻结零计数与授权边界

```text
weak_snapshot_count=0
rich_snapshot_count=0
snapshot_count=0
planned_trials=0
actual_trials=0
registration_execution_count=0
downloaded_lidar_payload_count=0
PUBLIC_DATA_V2_RUN_AUTHORIZED=false
REAL_DATA_MAIN_EXPERIMENT_AUTHORIZED=false
MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false
```

## 最终结论

BOREAS_EXTERNAL_V2_STAGE1_READY=true；READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD=true。Boreas 已在版本化的 supplementary external-validation 协议下通过 GT-only Stage-1。其轨迹 backbone 独立于待测 registration，固定外参的 LiDAR-assisted provenance 已公开保留，且所有 reference uncertainty 限制均未被隐藏。下一独立任务可以按冻结 allowlist 下载 primary pair 的最小必要 LiDAR，但本任务未执行 ICP。
