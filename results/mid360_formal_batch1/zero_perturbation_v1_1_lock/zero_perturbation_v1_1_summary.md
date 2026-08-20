# FMB1 Zero-Perturbation v1.1 R1 Locked Summary

The corrected final dataset is reauthenticated and frozen as `R01/R02/R03 + W01/W02/W03`, with W02 attempt 2 as the active Weak scene. W02 attempt 1 remains archived as an invalid wrong-scene acquisition; W04 is retired and absent from the final data and trial plan. The original W04-based proposal remains byte-unchanged and is separately registered as `SUPERSEDED_PROPOSAL`.

R1-C1 is ACTIVE as a scientific protocol, but grants no execution authority. It freezes scene-level independence, forbids snapshot pseudo-replication, discloses that nominal Identity plus no obvious motion is not sub-millimeter ground truth, preserves the capture-radius track as supplementary/not executed, and records synthetic Model A/B transfer as not compatible.

The locked plan contains exactly 360 rows: 180 Open3D and 180 PCL, all initialized at Identity. Independent lock verification, preflight, dry-run, tamper tests, 461 FMB1 tests, and the 1644-test full repository suite all passed with zero failures or errors. The 18 skips are documented external/historical/no-registration cases.

Final state at report generation:

```text
FMB1_ZERO_PERTURBATION_V1_1_LOCK_READY=true
READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION=true
FORMAL_ICP_UNLOCKED=false
FORMAL_REGISTRATION_AUTHORIZED=false
actual_open3d_trials=0
actual_pcl_trials=0
actual_formal_trials=0
```

Lock fingerprint: `9fcd01bd439e1c4e8c2bd0a220a3af1d16c394177bb17197850c4d769109cbba`.
