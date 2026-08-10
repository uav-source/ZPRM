# RTS-GT 单数据集第二轮预注册总结

1. 第一次失败数据释放 73.676959744 GB（68.617 GiB）。
2. RTS-GT 当前本地占用 0.231616512 GB（含 partial-clone Git 对象；无 LiDAR payload）。
3. Tunnel 审计 20220717-1..5；GT-overlap 最优候选为 20220717-2 map → 20220717-4 query，exp1 参考文件为空。
4. Campus 审计 20220715-1..4；GT-overlap 最优候选为 20220715-1 map → 20220715-4 query。
5. 是；同一 deployment 的官方 GCP/control 文件逐字节一致，且没有做轨迹或点云拟合。
6. Tunnel GT-only overlap PASS。
7. Campus GT-only overlap PASS。
8. R02 PASS：同步、非共线的三棱镜 RTS 轨迹可独立构造 prism-rig 6DoF；但 LiDAR 变换链在 R05 失败。
9. 否；Stage 1 失败，weak=0、rich=0、snapshot=0。
10. R02=PASS，R03=PASS_CANDIDATE，R04=PASS，R05=FAIL，R06–R08=BLOCKED_STAGE1，R09=PASS，R10=FAIL。
11. R14 未完成（BLOCKED_STAGE1），没有 interval 可冻结。
12. 是；Open3D=0、PCL CLI=0、其他 registration process=0、registration execution=0。
13. 否；SINGLE_DATASET_PREREGISTRATION_READY=false。
14. 主要阻塞是 R05：官方 release 缺 sensor_positions/LiDAR 外参，并且 Tunnel 目录日期与参考轨迹时间戳冲突。
15. 暂不值得立即下载第二数据集；应先向 RTS-GT 发布方取得可认证外参与 Tunnel 时间/版本说明，否则当前 RTS-GT 无法完成 R05/R10。

## 判定

`SINGLE_DATASET_PREREGISTRATION_READY=false`

Stage 1 在 R05 失败后已停止：Campus 发布目录没有 LiDAR/robot `sensor_positions.csv` 或等价外参，Tunnel 的 `20220717-*` 目录内参考轨迹时间戳却落在 2022-05-23；官方脚本所需的传感器位置文件亦未随所审计 release 发布。未下载 LiDAR、未建图、未选择 weak/rich snapshot、未运行任何 registration。

`R01=PENDING_SECOND_DATASET`，`REAL_DATA_RUN_AUTHORIZED=false`，`MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false`。
