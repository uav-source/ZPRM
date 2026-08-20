# FMB1 W04 field checklist

Status: `NOT_ACQUIRED`. This checklist records a future weak-scene replacement candidate; it does not authorize registration and does not assert a W04 location.

Verbatim field gates: `3 genuinely distinct stations`; `MAP >=19 s`; `QUERY >=14 s`; `gap >=10 s`; `no overlap`; `new timestamp`; `failed attempts retained`.

## Before recording

- [ ] Record the actual physical location: `____________________________`.
- [ ] Record an environment description relevant to geometric weakness: `____________________________`.
- [ ] Record local date/time, timezone, operator, sensor serial number, mount/support, and weather/lighting if relevant.
- [ ] Confirm the scene identifier is `FMB1_W04` and the semantic label is only `WEAK_CANDIDATE`.
- [ ] Confirm the location was selected without inspecting any ICP/backend result.
- [ ] Confirm the Mid-360 is mechanically supported and not handheld.
- [ ] Confirm `/livox/lidar` is `sensor_msgs/PointCloud2` and `/livox/imu` is `sensor_msgs/Imu`.
- [ ] Confirm live LiDAR messages use `frame_id=livox_frame` and include `x,y,z,intensity,tag,line,timestamp`.
- [ ] Confirm clock state and available disk space. Do not rename, overwrite, or reuse any W02 bag.

Unknown field values must remain `UNKNOWN` or `null`; never infer a place name from geometry or filenames.

## Station record

Complete all three independent station blocks. Mark the physical point before MAP, and do not intentionally move or reorient the sensor until QUERY has finished.

### S01

- Physical mark / unambiguous station description: `____________________________`
- Support/mount and sensor height: `____________________________`
- Nominal heading/orientation: `____________________________`
- MAP raw filename and capture prefix: `____________________________`
- MAP first/last timestamp and measured duration: `____________________________`
- MAP SHA256 / bytes: `____________________________`
- Wait interval observed: `____________________________`
- QUERY raw filename and capture prefix: `____________________________`
- QUERY first/last timestamp and measured duration: `____________________________`
- QUERY SHA256 / bytes: `____________________________`
- Same mounting, physical station, and nominal orientation attested: `[ ]`
- No intentional platform motion attested: `[ ]`

### S02

- Physical mark / unambiguous station description: `____________________________`
- Support/mount and sensor height: `____________________________`
- Nominal heading/orientation: `____________________________`
- MAP raw filename and capture prefix: `____________________________`
- MAP first/last timestamp and measured duration: `____________________________`
- MAP SHA256 / bytes: `____________________________`
- Wait interval observed: `____________________________`
- QUERY raw filename and capture prefix: `____________________________`
- QUERY first/last timestamp and measured duration: `____________________________`
- QUERY SHA256 / bytes: `____________________________`
- Same mounting, physical station, and nominal orientation attested: `[ ]`
- No intentional platform motion attested: `[ ]`

### S03

- Physical mark / unambiguous station description: `____________________________`
- Support/mount and sensor height: `____________________________`
- Nominal heading/orientation: `____________________________`
- MAP raw filename and capture prefix: `____________________________`
- MAP first/last timestamp and measured duration: `____________________________`
- MAP SHA256 / bytes: `____________________________`
- Wait interval observed: `____________________________`
- QUERY raw filename and capture prefix: `____________________________`
- QUERY first/last timestamp and measured duration: `____________________________`
- QUERY SHA256 / bytes: `____________________________`
- Same mounting, physical station, and nominal orientation attested: `[ ]`
- No intentional platform motion attested: `[ ]`

## Immediate acquisition audit

For each of six new bags:

- [ ] Raw filename and path retained unchanged; SHA256 and byte size recorded.
- [ ] Required topics, types, frame, and PointCloud2 fields pass exactly.
- [ ] LiDAR rate is 9.5–10.5 Hz and IMU rate is 190–210 Hz.
- [ ] MAP duration is at least 19 s; QUERY duration is at least 14 s.
- [ ] IMU statistics were recorded with acceleration unit `UNKNOWN`; no 9.81 conversion or position integration was used.
- [ ] Motion screen is `NO_OBVIOUS_MOTION`.

For each station pair:

- [ ] `query_first_timestamp - map_last_timestamp >= 10 s`.
- [ ] MAP and QUERY do not overlap.
- [ ] Both roles share the intended station and use newly captured timestamps.

Any failed station keeps its raw files and invalidates W04 as a three-station candidate until a new attempt is recorded with a new timestamp.

## Pre-backend processing and admission

- [ ] All three stations are `ACQUISITION_PASS` before any W04 target or geometry admission is accepted.
- [ ] Each target uses only its MAP bag, direct stationary merge, frozen filters, and 0.05 m voxelization.
- [ ] Query contribution to every target is exactly zero; registration/odometry/scan matching calls are zero.
- [ ] Exactly ten QUERY snapshots per station use quantiles `0.05,0.15,...,0.95`; no manual frame choice occurred.
- [ ] Exactly 30 geometry-only rows exist for W04 and contain no registration-derived result.
- [ ] Scene medians satisfy all WEAK gates: normalized lambda-min ≤ 0.12, condition number ≥ 6.0, entropy ≤ 0.80.
- [ ] W04 is rejected if its geometry is `RICH` or `INTERMEDIATE`; no relabeling is permitted.
- [ ] A new frozen registry, manifests, hash anchor, formal lock, and fingerprint are independently verified before any later backend authorization.

Completion of this checklist does not authorize Open3D, PCL, ICP, GICP, NDT, or any other registration process.
