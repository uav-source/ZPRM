# Mid-360 Formal Batch-1 acquisition protocol

FMB1 is a formal-acquisition, blinded variance-estimation/feasibility batch. It is not a final sample-size claim and it is not yet a formal measurement result. The six scenes are the six highest-level independent experimental units. Three stations per scene and ten snapshots per station are nested repeated observations, not additional independent scenes.

Before recording, the operator must physically mark S01, S02, and S03 as three genuinely different locations for each scene. Each sensor/platform must be mechanically supported; hand-held recording is forbidden. Unknown height, heading, spacing, operator, or environment facts remain `null` or `UNKNOWN`.

For one station, verify that `/livox/lidar` is live as `sensor_msgs/PointCloud2` and `/livox/imu` is live as `sensor_msgs/Imu`. Record MAP for 20 s, stop, wait exactly the scripted 12 s target, then record QUERY for 15 s. The sensor mounting, physical station, and nominal orientation must not change between MAP and QUERY. Hard acceptance requires MAP duration at least 19 s, QUERY duration at least 14 s, actual gap at least 10 s, and no overlap.

Only the two required topics are recorded. Both must use `livox_frame`; mismatch is FAIL/REVIEW and is never auto-corrected. Required PointCloud2 fields are `x,y,z,intensity,tag,line,timestamp`. LiDAR must be 9.5–10.5 Hz and IMU 190–210 Hz. The IMU is used only for record integrity, rate audit, and the `NO_OBVIOUS_MOTION` screen. Its acceleration unit remains unknown; no 9.81 scaling, integration, displacement, or ground-truth claim is permitted. Point coordinates remain `ASSUMED_METERS_FROM_SCALE`.

After every station: compute both bag SHA256 values, audit each bag, run the obvious-motion screen, audit the gap/no-overlap rule, and write pair metadata. A failure is retained and marked invalid with its reason. A new attempt uses new timestamped files and never overwrites an earlier attempt.

Only after all three stations for a candidate pass acquisition audit may the registration-free target builder, fixed-quantile snapshot freezer, and geometry-only admission run. Replacement is allowed only for acquisition quality, bag integrity, or geometry-only eligibility, and the decision must precede any ICP result.

