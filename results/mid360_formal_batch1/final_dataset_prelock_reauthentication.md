# FMB1 final dataset pre-lock reauthentication

Status: **PASS** for the corrected final dataset. This report does not activate a protocol, issue a formal lock, or authorize backend execution.

The active dataset is exactly 6 scenes, 18 stations, 18 MAP-only targets, and 180 query snapshots. Rich scenes are R01–R03; Weak scenes are W01–W03. The independent evidence rehashed 42 candidate raw bags (36 admitted plus 6 retained invalid), 18 targets, and 180 sources with zero missing files and zero mismatches. The final dataset `SHA256SUMS` passed all 13 entries.

W02 attempt 1 remains retained in `archive/invalid_acquisition/FMB1_W02_attempt1_wrong_location/`: all 6 bags exist and their hashes were recomputed successfully. It contributes zero active bags and zero final snapshots. Its status is `INVALID_ACQUISITION`, reason `WRONG_SCENE_LOCATION / OPERATOR_SCENE_SELECTION_ERROR`.

W02 attempt 2 is the active W02: 6 bags, 3 stations, 3 targets, 30 snapshots, final geometry class `WEAK`, and `GEOMETRY_ADMITTED`. W04 is retired and contributes zero rows to every active final manifest; its prior replacement interpretation is retained only as superseded history.

The correction is administrative and pre-backend: original W02 attempt 1 was an operator wrong-scene acquisition; the 2026-08-20 data are W02 attempt 2 reacquisition; W04 identity is retired. `correction_at_formal_trial_count=0`, and no registration evidence was used.

Protected v1 preregistration, capture-radius analysis protocol, and backend parameter contract hashes remain unchanged. Actual Open3D, PCL, and total formal trial counts are all zero. `FORMAL_ICP_UNLOCKED=false`, `FORMAL_REGISTRATION_AUTHORIZED=false`, and `MEASUREMENT_FINAL_RESULT=false`.

Pre-ICP data freeze: commit `b84c81855d0cdce2958cdc51ea3831ea4fb17f5a`, annotated tag `freeze/fmb1-final-dataset-pre-icp-v1`.
