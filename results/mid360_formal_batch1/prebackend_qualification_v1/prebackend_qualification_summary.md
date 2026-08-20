# FMB1 W04 补采前 pre-backend 资格总结

## 结论

`FMB1_PREBACKEND_EXECUTION_PATH_QUALIFIED=true`，同时 `FMB1_CURRENT_REAL_BATCH_BLOCKED=true`。
真实批次唯一科学阻塞原因是 `MISSING_ADMITTED_WEAK_REPLACEMENT_W04`。

## 当前真实闭包

- 36/36 raw bags、18/18 stations、180 provisional snapshots 已重新认证。
- admitted scenes：Rich 3，Weak 2，合计 5；没有把 5 scenes 当成完整 batch。
- W02 数据和 rejection 均保留；其规则原因是 `GEOMETRY_ONLY_INELIGIBLE`，未来正式集合不包含 W02。
- W04 计划已在任何 ICP 前冻结；明天需 3 stations、6 个全新 timestamp bags。地点、方向、高度仍为 UNKNOWN。

## 软件执行链

- 真实 preflight 正确 fail-closed，没有发行 150-trial/任何 trial matrix、正式 lock 或 authorization。
- fixture-only 6×3×10×2 lifecycle、fresh/resume/interruption、worker invariance 与 SHA stability 全部通过。
- partial、orphan、checksum、manifest、duplicate、missing 均 fail-closed；fixture 不可发布或引用。
- schema 严格区分 ZERO_PERTURBATION_TRACK、CAPTURE_RADIUS_TRACK、FIXTURE_ONLY；zero track 强制 Identity T0。
- scene 保持最高独立分析单位，station/snapshot 是 nested repeated observations。

## 协议对齐

现行 active FMB1 primary endpoints 仍是 weak/strong direction capture radius，非论文的 Identity zero-perturbation 主线，且 active perturbation magnitudes 不含 0。
已生成 `PROPOSED_NOT_ACTIVE` 的 v1.1 提案：zero-perturbation 为 proposed primary，capture-radius 为 optional supplementary；提案没有 formal authority。

## 零执行与测试

- actual Open3D/PCL/formal trials = 0/0/0；NO_ICP_ATTESTATION_TONIGHT PASS。
- 改动前：1300 collected，17 skipped，0 failed。
- FMB1 专项：232 passed，0 skipped，0 failed。
- 全量：1397 passed，18 skipped，0 failed。

## 明天入口

W04 Weak admission PASS 后，目标集合为 R01/R02/R03/W01/W03/W04；仍须 final lock、独立 verifier、单独正式 registration authorization，不能在同一命令中自动运行 ICP。

`FORMAL_LOCK_ISSUED=false`  
`FORMAL_ICP_UNLOCKED=false`  
`FORMAL_REGISTRATION_AUTHORIZED=false`  
`actual_formal_trials=0`
