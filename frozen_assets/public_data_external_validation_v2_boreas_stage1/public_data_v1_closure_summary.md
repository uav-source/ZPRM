# Public-data v1 screening closure

Public-data v1 is closed as a failed eligibility screening, not as a successful real-data qualification.
All five candidates were excluded before any real registration result existed.

| Candidate | Historical status | Frozen failure reason |
|---|---|---|
| IILABS | `R03=FAIL;R04=FAIL;R05=FAIL;R10=FAIL` | cross-acquisition fixed world frame not proven |
| GrandTour | `R03=FAIL;R04=FAIL;R05=FAIL;R10=FAIL` | GT-only overlap insufficient |
| RTS-GT | `R05=FAIL;R10=FAIL;single_dataset_preregistration_ready=false` | LiDAR extrinsic/release lineage and uncertainty failure |
| CAVERS | `R04=FAIL;R10=PARTIAL;CAVERS_STAGE1_READY=false` | cross-recording fixed world frame not proven |
| Boreas v1 | `R02=FAIL;R10=PARTIAL;BOREAS_STAGE1_READY=false` | R02 failed under the stricter Stage-1 independence interpretation; R10 partial |

Boreas v1 remains `R02=FAIL`, `R10=PARTIAL`, and `BOREAS_STAGE1_READY=false`; this closure does not reinterpret it.

```text
PUBLIC_DATA_V1_SCREENING_COMPLETE=true
PUBLIC_DATA_V1_CANDIDATE_COUNT=5
PUBLIC_DATA_V1_ELIGIBLE_DATASET_COUNT=0
PUBLIC_DATA_V1_REAL_REGISTRATION_COUNT=0
PUBLIC_DATA_V1_ICP_COUNT=0
PUBLIC_DATA_V1_RUN_AUTHORIZED=false
PUBLIC_DATA_V1_CLOSED=true
```

Verified source bundles: 4; source files including their root SHA256SUMS: 124.
