# Boreas External Validation v2 Stage-2 preprocessing contract

状态：`FROZEN`。本合同在任何真实 Boreas LiDAR payload 下载、geometry metric、weak/rich 标签或 registration 产生之前冻结。

规范机器可读文件是 `protocols/boreas_v2_stage2_preprocessing_contract.json`：

- 文件 SHA-256：`6f5b3c1af4f329585d98404efc999b17b28ea9eb42114adde3b82f815fd14bc6`
- 去除 `contract_payload_sha256` 后，以 UTF-8、sorted keys、compact separators、`ensure_ascii=false`、`allow_nan=false` 编码的 payload SHA-256：`e1f89636946bad0d7d57af40a5a5dff4dfe37d92f0eea3c695c0672e2a0857b8`
- 生产预处理实现：`src/phase_a_harness/real_data_preparation/boreas_v2_stage2_preprocessing.py`
- 实现 SHA-256：`045b5b650bcd658f7abff401c6c95317f63dae8a07665daf6a3336683f88bce3`

## 冻结边界

只允许 frozen `PRIMARY_PAIR`：

- target-map sequence：`boreas-2021-11-14-09-47`
- query sequence：`boreas-2021-01-26-11-22`

本合同不授权下载、geometry screening 或 registration。冻结时 LiDAR payload 下载、geometry metric 和 registration execution count 均为 0。它不得被 storage planner、真实点云外观、weak/rich 排名或未来 backend 结果修改。

## Wire format 与时间

Boreas Velodyne Alpha-Prime 每点严格为 24 bytes、六个 little-endian float32：

```text
[x, y, z, intensity, laser_id, t]
```

`t` 是相对扫描时间中点的秒偏移；文件名是该扫描时间中点的 Unix UTC 微秒整数。微秒整数到 float seconds 严格复现 pinned pyboreas `micro_to_sec`：`float(np.round(timestamp_us / 1e6, 6))`，不得替换成 epoch-scale 下可能相差一个 float64 ULP 的 `float(timestamp_us) * 1e-6`。解码后立即转换为 little-endian float64。原始 binary record ordinal 是不可变的初始点序。

每个 allowlist object 必须在同 sequence 的 frozen `lidar_poses.csv` 中精确匹配同一整数微秒行。Stage-2 不做 nearest-pose、插值或外推；精确行不存在即 fail closed。`0.20 s` 是 frozen native-reference continuity ceiling，不是允许模糊 timestamp join 的容差；join 容差严格为 0 微秒。

`T_reference` 是官方 scan-middle `T_ENU_lidar`，方向为 source lidar frame 到 fixed `ENU_ref`：

```text
T_ENU_lidar(t) = T_ENU_applanix(t) @ T_applanix_lidar
```

不猜测或重命名 raw LiDAR 轴；原生 LiDAR XYZ 通过已审计的官方外参方向进入 Applanix，再进入 `ENU_ref`（x East、y North、z Up）。

## Deskew

Deskew 开启，并严格采用 pinned pyboreas `PointCloud.remove_motion` 的 sorted-timestamp 21-bin constant-body-twist 路径：

1. `t_ref` 显式为文件名的官方 scan temporal middle；
2. 从同 timestamp 的官方 `lidar_poses.csv` 行得到 Applanix-derived lidar-frame body twist；
3. retained point timestamps 必须 nondecreasing；
4. retained timestamp min/max 之间建立 21 个均匀 bin、20 个 interval；
5. 每个 interval 使用其左端点的 `Exp((t_bin - t_ref) * body_rate_lidar)`；
6. 最大 timestamp 属于第 19 个 interval，官方 sorted path 中第 21 个 transform 不被应用；
7. 非单调、非有限或退化 timestamp span 直接 fail closed。

Deskew 只使用 Applanix reference、frozen static extrinsic provenance 与 per-point time；严禁 KISS-ICP、FAST-LIO、LOAM、STEAM-ICP、scan matching 或其他 LiDAR odometry。

## Filter、downsampling 与 target map

处理顺序固定为：

```text
float32 decode → float64
→ finite XYZ/time filter
→ raw-lidar Euclidean range filter [1.0 m, 80.0 m]（两端 inclusive）
→ reference-based deskew
→ role-specific voxelization
```

Intensity 与 laser ID 不参与 geometry，也不作为 finite geometry gate。Dynamic-object policy 和 crop policy 均为 `NONE`。

Query source 在 deskew 后的 scan-middle native lidar frame 使用 `0.10 m` centroid voxel；origin 为 `[0,0,0]`，同 voxel 按 retained raw point ordinal 做 float64 sum/count，输出按 voxel key `(x,y,z)` lexicographic 排序。

Map 不先做 per-scan source voxel。全部 8,202 个 frozen target-map allowlist scans 按 allowlist ordinal、再按 retained raw point ordinal 转到 fixed `ENU_ref`，随后进入一个全局 `0.10 m` centroid voxel accumulator。target origin 为 ENU_ref `[0,0,0]`，最终点序为 voxel key `(x,y,z)` lexicographic。Query contribution 严格为 0；target map 只有一个 logical/physical copy。

`1–80 m`、source `0.10 m` 与 target `0.10 m` 来自在 Boreas geometry 之前已经存在的 repository-wide real-data v1 scientific prior。该 prior 的状态为 `FROZEN_BUT_NOT_EXECUTED_DUE_TO_ELIGIBILITY_FAILURE`，传感器原为 Hesai/Ouster；因此这里只能把它作为预先存在、无 Boreas outcome 调参的统一数值依据，不能声称是 Boreas 官方推荐。v1 的 `every_fifth_map_scan` 不继承，因为 Stage-2 已冻结全部 8,202 个 map allowlist objects。

## Geometry-only 常数与 backend 边界

Stage-2 screening 保持 frozen common-association 常数：one-NN、maximum distance `0.50 m`、`cKDTree workers=1`；target-only unoriented PCA normals 使用 `k=50`、minimum neighbors `10`、chunk `2048`、normal-norm epsilon `1e-12`。

Geometry-only screening 只能输出预注册的 correspondence count 与 translation Hessian spectrum。它不得为了复用已有 context 而计算或持久化 initial residual、gradient、final transform 或其他 registration-derived 字段。未来 Open3D/PCL normal 与 ICP 参数仍由 `frozen_assets/backend_parameter_contract.json` 单独约束，本任务不运行任何 backend。

Canonical source、target 均为 finite、C-contiguous、little-endian float64 `N×3` NPY v1.0；`T_reference` 为同规范的 float64 `4×4` NPY v1.0。未来两个 backend 必须引用相同 content-addressed source/target；禁止 backend-specific preprocessing copy。

## E02 limitation 与 uncertainty

Reference trajectory backbone 是 GNSS/IMU/wheel/RTX，未使用 LiDAR、ICP、scan matching 或 SLAM。但是 `T_applanix_lidar` 的官方历史 provenance 是 LiDAR-assisted proprietary batch calibration；calibration session、query sequence 是否参与、translation/rotation uncertainty 均为 `UNKNOWN`。因此 E02 只能是 `PASS_WITH_DOCUMENTED_LIMITATION`，Boreas 只能作为 supplementary external generalization。

保留以下数值状态：

- position reference：sequence-specific nominal `0.02–0.04 m RMSE`，并保留 documented urban-canyon `0.20–0.40 m residual RMSE`；
- orientation reference：`UNKNOWN`；
- time synchronization：`UNKNOWN`；
- extrinsic translation/rotation：`UNKNOWN`；
- upstream reference-pose materialization/interpolation：`UNKNOWN`；
- deskew：`UNKNOWN`；
- target-map accumulation：`UNKNOWN`；
- voxelization：`UNKNOWN`。

Stage-2 自身 pose interpolation operation 是 `NOT_APPLICABLE_EXACT_TIMESTAMP_JOIN`，但这不能把上游 reference-pose materialization uncertainty 改成 0。任何 `UNKNOWN → 0` 都违反本合同。
