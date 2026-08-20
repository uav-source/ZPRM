# FMB1 当前失败闭包重新认证

状态：`PASS`。当前真实批次仍被 `MISSING_ADMITTED_WEAK_REPLACEMENT_W04` 正确阻塞。

- 原始 bag：36/36 已认证
- station acquisition：18/18 PASS
- provisional snapshots：180
- admitted scenes：5（Rich 3，Weak 2）
- W02：数据保留，geometry rejected，不属于未来正式集合
- W02 冻结替换原因：`GEOMETRY_ONLY_INELIGIBLE`
- 正式 lock、ICP unlock、registration authorization：均为 false
- 实际 Open3D/PCL/formal trials：0 / 0 / 0

该重新认证不采纳 W02，不生成 W04，也不产生任何 backend 结果。
