# R01–R10 Eligibility Audit

| ID | Global | IILABS | GrandTour | Reason |
|---|---|---|---|---|
| R01 | PASS | PASS | PASS | Two public sources and immutable release identifiers were recorded. |
| R02 | PASS | PASS | PASS | Downloaded IILABS MoCap TUM poses and GrandTour IE-TC/NavSatFix 6DoF reference files are independent of LiDAR registration. |
| R03 | FAIL | FAIL | FAIL | No target map was constructed; IILABS cross-acquisition world frame is unproven and GrandTour GT-only overlap failed. |
| R04 | FAIL | FAIL | FAIL | Acquisitions are distinct, but no eligible cross-acquisition map/query pair survived the common-frame and overlap hard gates. |
| R05 | FAIL | FAIL | FAIL | IILABS raw MoCap frame continuity is not proven and GrandTour raw Hesai per-point timing was intentionally not downloaded after overlap failure. |
| R06 | BLOCKED | BLOCKED | BLOCKED | Zero snapshots were selected because materialization stopped at hard gates. |
| R07 | BLOCKED | BLOCKED | BLOCKED | No labels exist to freeze; registration-result count is zero. |
| R08 | BLOCKED | BLOCKED | BLOCKED | No canonical bundles exist, so byte-identical backend inputs cannot be asserted. |
| R09 | PASS | PASS | PASS | Historical backend parameter file and internal canonical hashes match exactly. |
| R10 | FAIL | FAIL | FAIL | Extrinsic/map/interpolation numeric uncertainty evidence is incomplete; UNKNOWN values were not replaced with zero. |
