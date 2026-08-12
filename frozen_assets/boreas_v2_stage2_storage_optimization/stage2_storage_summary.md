# Boreas v2 Stage-2 存储优化审计总结

- `STAGE2_STORAGE_PLAN_READY=true`
- `CURRENT_DISK_SUFFICIENT=true`
- `STAGE2_DOWNLOAD_AUTHORIZED=false`
- `PUBLIC_DATA_V2_RUN_AUTHORIZED=false`
- `REAL_REGISTRATION_AUTHORIZED=false`
- `MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false`
- 本状态只表示存储与执行架构准备完成，不是下载或实验成功。

## 27 个必答问题

1. 旧公式把完整 raw、2×decoded、2×map、2×canonical 和一个完整 raw 大小的 temp 同时相加为 708,949,459,392 bytes，再乘 1.5 得 1,063,424,189,088 bytes （990.390954 GiB）。

2. 最大显式重复包络是 decoded 与 canonical，各为完整 remote payload 的 2 倍；随后又叠加 50% 总体余量。

3. 没有证据表明旧设计实际存在 100 份 target map；旧 JSON 没有单列这一项。新合同明确禁止并检测这种复制。

4. 没有证据表明旧设计实际持久保存 Open3D/PCL 双份输入；新合同让两个 backend 引用同一 canonical SHA。

5. 优化后保存 1 份 content-addressed 全局 target map；100 个未来 snapshot 仅引用其 SHA。

6. 可以。候选 query 第一遍只保留 authenticated receipt、checkpoint 和 geometry-only row，成功提交后删除 raw/decoded temp。

7. 是。第一遍 blind geometry screening 后冻结 100 个选择；第二遍只为这 100 个重取并生成 canonical source。

8. 每个 selected source 以 deterministic canonical NPY 内容寻址保存一次，并由 Open3D/PCL 共享；raw temp 随后删除。

9. 不必须。STREAMING_LOW_DISK 不永久保存 104,158,637,472 remote bytes；FULL_RAW_CACHE 只是可选操作模式。

10. THEORETICAL_MINIMUM 要求 70005149653 bytes（65.197376 GiB）。

11. RECOMMENDED_OPERATIONAL 要求 101563921871 bytes（94.588773 GiB）。

12. CONSERVATIVE_FULL_CACHE 要求 226554286838 bytes（210.995122 GiB）。

13. 当前 118671130624 bytes（110.521103 GiB）足够运行推荐的 STREAMING_LOW_DISK 模式；它不满足 FULL_RAW_CACHE。

14. 推荐流式模式额外需要 0.000000 GiB；full-cache 额外需要 100.474019 GiB。

15. 真实 target map 大小尚未知；不设 voxel 参数的保守持久上界是 41998817408 bytes。

16. 推荐模式最大 mutually-exclusive temporary phase 是 map checkpoint，上界 41998817408 bytes。

17. 推荐模式真正 simultaneously-live peak 是 84636601559 bytes。

18. 是。checkpoint 为 locked O_APPEND canonical JSONL hash chain，绑定 remote identity、科学合同和 state/row transition SHA。

19. 不需要。只重做未完成对象；认证完成项跳过，ETag/size/LastModified/contract 改变或 orphan state 均 fail closed。

20. 没有。未冻结的 voxel/range/normal/association 数值均标记 PREPROCESSING_PARAMETER_REQUIRES_STAGE2_PREREGISTRATION。

21. 没有。primary pair 仍为 boreas-2021-11-14-09-47 -> boreas-2021-01-26-11-22。

22. 没有。真实 lidar/*.bin 下载对象数和字节数均为 0。

23. 没有。Open3D/PCL/其他 registration 调用及 real result 均为 0。

24. 是；source-only 测试 934 collected, 924 passed, 10 skipped, 0 failed, 0 errors。

25. 是；producer 返回及 freeze 前均强制运行 `python3 scripts/verify_boreas_v2_stage2_storage_plan.py` 并要求 PASS，另有 11 个语义篡改用例必须全部被拒绝。

26. STAGE2_STORAGE_PLAN_READY=true。

27. 下一步只能由用户在单独任务中授权真正 Stage-2 下载；本 closure 仍保持 STAGE2_DOWNLOAD_AUTHORIZED=false。

## 最终边界

STAGE2_STORAGE_PLAN_READY=true 只表示 storage/lifecycle/resume/low-disk architecture 的当前审计结论；真实 Stage-2 runner 不在本任务范围，未来 runner 必须在启动和每次 projected write 前接入已测试的磁盘门禁。所有真实下载、geometry selection、Open3D/PCL 和 registration 仍未授权且未执行。下一步若要下载，必须由用户在单独任务中显式授权，并先独立冻结缺失的 preprocessing 科学参数。
