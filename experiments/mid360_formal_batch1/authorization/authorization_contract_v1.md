# FMB1 Formal Registration Authorization Contract v1

This contract controls one future execution of the frozen FMB1 Zero-Perturbation v1.1 R1 360-row plan. It changes no scientific input, scene assignment, backend parameter, transform initialization, or analysis rule.

The authorization is an immutable canonical JSON document. Issuance requires an explicit operator confirmation flag and binds the exec-r2 lock fingerprint and file SHA, lock-release commit, execution-code commit, frozen plan/analysis/backend/environment/PCL identities, exactly 180 Open3D plus 180 PCL trials, Identity-only T0, two workers, and the canonical runtime root.

Authorization lifecycle state is external to the immutable authorization:

```text
ISSUED -> VERIFIED -> IN_USE -> CONSUMED
                              -> CONSUMED_BY_FAILED_ATTEMPT
IN_USE -> RESUME_SAME_ATTEMPT -> CONSUMED
```

`authorization_in_use.json` binds the selected authorization to exactly one runtime before the first backend loader is reached. `authorization_consumption_receipt.json` permanently consumes it after a complete or failed formal attempt. Neither file may overwrite or mutate the original authorization. A consumed authorization cannot start or resume another run. An in-use authorization may only resume the same runtime, lock, plan, environment, and execution commit.

The authorization never permits capture-radius execution, a third backend, parameter changes, trial reselection, non-Identity T0, or a second fresh run. Fixture lifecycle qualification must have zero real Open3D/PCL calls.
