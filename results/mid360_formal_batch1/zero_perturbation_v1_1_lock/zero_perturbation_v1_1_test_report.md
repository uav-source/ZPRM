# FMB1 Zero-Perturbation v1.1 R1 Test Report

Status: **PASS**  
Generated: `2026-08-20T03:52:06Z`  
Execution-code commit: `e162bf1e53e683424562a0488fa19097d3151d2f`

Both required suites ran under the FMB1 and real-data no-registration guards.

| Suite | Collected | Passed | Skipped | Failed | Errors |
|---|---:|---:|---:|---:|---:|
| `tests/mid360_formal_batch1` | 461 | 461 | 0 | 0 | 0 |
| Full repository | 1644 | 1626 | 18 | 0 | 0 |

The 18 full-suite skips are fully accounted for: four tests intentionally execute real Open3D/PCL backends and were skipped by `NO_FORMAL_REGISTRATION=true`; the remainder require external or historical assets absent from the consolidated source workspace. Tamper tests passed. No backend was imported or called, no trial result was written, and authorization remained false.
