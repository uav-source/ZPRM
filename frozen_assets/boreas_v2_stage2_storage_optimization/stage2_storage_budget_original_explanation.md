# Boreas v2 Stage-2 原始磁盘预算逐项审计

## 结论

旧预算的 990.390954 GiB 是一个粗粒度的同时存活上界，不是不可避免的物理需求。它先把完整 raw、两倍 decoded、两倍 map、两倍 canonical 与一个完整 raw 大小的 temporary 全部相加，再统一增加 50% 容量。旧 JSON 没有对象生命周期，也没有证明这些大项会在同一时刻全部存在。

旧文件也没有单列 Open3D/PCL 副本或 100 份 per-snapshot target；因此本审计不能声称旧实现实际保存了这些副本。它们至多可能被 decoded/canonical/map 的粗粒度 envelope隐含覆盖。优化的依据是明确生命周期、单 target 内容寻址、共享 backend 输入与流式临时消费，而不是把未证明的旧副本当成事实。

## 原公式

```text
subtotal = download + decoded_working + target_map + canonical_bundle + temporary_processing
         = 708949459392 bytes
required = ceil(subtotal × 1.5)
         = 1063424189088 bytes = 990.390954 GiB
```

## 有非零显式贡献的项目

| component | peak bytes | legacy evidence |
|---|---:|---|
| raw lidar payload | 104158637472 | frozen disk budget: download_bytes |
| temporary download files | 104158637472 | frozen disk budget: temporary_processing_bytes |
| decoded arrays | 208317274944 | frozen disk budget: decoded_or_unpacked_working_bytes |
| accumulated target map | 83997634560 | frozen disk budget: target_map_bytes |
| canonical source arrays | 208317274944 | frozen disk budget: canonical_bundle_bytes |
| safety factor | 354474729696 | minimum_safety_multiplier=1.5 |

完整 CSV/JSON 还保留所有要求审计的零贡献行；零表示旧模型没有单独估计，不能解释成真实运行一定为零。
