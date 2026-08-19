# Mid-360 controlled perturbation Pilot

`CONTROLLED_PERTURBATION_PILOT_READY=true`  
`CONTROLLED_PERTURBATION_PILOT_SUPPORTS_EXPANSION=false`  
`FORMAL_MEASUREMENT_RESULT=false`

This is a nonformal, descriptive sensitivity/debug experiment over the frozen
Rich and Weak Mid-360 Pilot inputs. It contains 480 new ICP runs and
does not include the preserved zero-perturbation runs in that count.

The geometry matrix is the existing translation-only matrix
`H_trans=(N.T@N)/valid_normal_count`, using the existing frozen 0.50 m
association and target PCA-normal context. Weak and strong directions are the
unit eigenvectors for `lambda_min` and `lambda_max`. Eigenvector signs are made
deterministic by requiring the largest-absolute component to be nonnegative;
both positive and negative perturbations are then tested.

Transforms map source/query points into target/map coordinates. The initial
pose is `T_initial = Delta_T @ T_star`, so the perturbation vector is expressed
in the target frame. Translation errors use `T_est @ inv(T_star)`. Recovery was
pre-registered as final translation error <= 0.005 m and final rotation error
<= 0.2 degree. The reduction ratio is `1 - final/initial`.

The Open3D/PCL algorithms and the backend parameter contract were not changed.
The backend contract SHA256 is `6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9`. Open3D iteration
count is blank because its API does not expose the executed iteration count.

Current zero-perturbation baseline retained in the report:

- Rich translation median/q95: Open3D 0.000535890/0.000789956 m; PCL 0.000580512/0.000834444 m.
- Weak translation median/q95: Open3D 0.000588254/0.000779236 m; PCL 0.000655950/0.000739324 m.
- Weak/Rich translation median ratio: Open3D 1.097713; PCL 1.129950.
- Rotation was opposite to a general degradation claim (Open3D Rich/Weak median 0.016014/0.009592 deg; PCL 0.018061/0.010979 deg), and turnover also did not support it.

Therefore: geometry separation is confirmed; zero-perturbation translation
showed a weak directionally consistent pilot signal; rotation and reassociation
did not support a general degradation claim.

`NO_OBVIOUS_MOTION` is retained and is not a claim of perfect, sub-millimeter,
or ground-truth static conditions. `ACCELERATION_UNIT_UNKNOWN=true`; no 9.81
conversion, IMU position integration, or IMU displacement ground truth was
used. Point coordinates remain `ASSUMED_METERS_FROM_SCALE`.

Snapshot-level observations within the same scene/station are not independent scene-level replicates.
All summaries are pilot descriptive / sensitivity analysis only and must not be
used for paper-level population inference.

Verification: `PASS`. See
`verification_report.txt`, `manifest.json`, and `decision_gate.json` for the
authenticated inputs and pre-registered decision rule.
