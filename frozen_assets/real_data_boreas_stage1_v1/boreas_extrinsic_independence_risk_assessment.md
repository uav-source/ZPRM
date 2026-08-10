# Boreas extrinsic independence risk

The Applanix trajectory itself uses GNSS, IMU, wheel encoder, and RTX—not lidar. However, the published `T_applanix_lidar` was produced as a by-product of a proprietary batch optimization using lidar pointclouds and post-processed GPS/IMU. All audited sequence copies are byte-identical, but the calibration sequence/session and exclusion of future map/query sequences are not published. The frozen R02 requirement is therefore failed closed; no claim that this dependence is negligible is made.
