# Static solver semantics audit

Status: `PASS_DIAGNOSTIC_DEFECT_IDENTIFIED`.

Open3D `solver_converged` is frozen as `finite and correspondence_count > 0` (`open3d_backend.py:81-86`). It is not the native Open3D stopping criterion. The wrapper stores `iteration_count=-1` and constructs `termination_reason` as `completed_finite_correspondences` or `open3d_invalid_result` (`:87-99`). Native iteration count, max-iteration contact, and relative-fitness/RMSE stop reason are unobservable.

PCL's adapter requires `has_converged_raw`, checks it against the compatibility field, and maps it directly to `PclBackendResult.has_converged` (`pcl_backend.py:179-241`). It also receives `iteration_count` (`:255`), but the formal result schema preserves neither field.

The runner maps backend convergence to internal `solver_success` and then to `scientific_status` (`zero_perturbation_v1_1_r1_runner.py:506-534,630-638`). The final rows omit the bool but preserve `scientific_status` and `solver_status`.

The defect is at `descriptive_summary_v1.py:82-83`: every finite status other than exact uppercase `CONVERGED` or `SUCCESS` is counted as nonconverged. All frozen rows instead contain the valid success token `completed_finite_correspondences`.
