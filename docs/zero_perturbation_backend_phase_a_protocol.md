# Zero-Perturbation Dual-Backend Phase A Protocol v1

Status: prospectively frozen before any Phase A snapshot, RNG construction, or
backend trial. Protocol type:
`dual_independent_backend_phase_a_qualification`, version 1.

This lock round defines a future IDEAL_MATCHED qualification of two independent
implementations: Open3D point-to-plane and PCL ICP-with-normals point-to-plane.
It authorizes no scientific claim and executes neither Phase A nor Phase B.
Native full/frozen are excluded because `native_full` failed the ideal matched
control; neither Native implementation may appear in the plan or be executed.

## Matrix and non-execution boundary

The future matrix contains exactly seven frozen scenes, three Development
geometry seeds, two Development measurement seeds, five repeats, and only
IDEAL_MATCHED: 210 planned snapshots. Each snapshot is paired with exactly two
backends, producing 420 planned trials. The CSV plans enumerate identifiers and
integers only. Plan construction must not import/call the scene generator,
construct an RNG, create coordinates, write PCD, or invoke a backend.

This round fixes all execution counters at zero. A planned row is not an
executed snapshot or trial. `NOT_EVALUATED` means the required observations do
not exist; it must never be converted to a numeric zero or zero failures.

## Source and target coordinate relationship

The target/map is produced in a future authorized run by the unchanged frozen
scene generator. Source is an exact deterministic subset of target coordinates,
in target order, selected by the existing inclusive 0.30–20.0 m reference-range
rule. There is no independent resampling, scan/map noise, or dropout.
Measurement seed and repeat may control only subset selection; they may not
change map geometry, reference pose, or scene parameters.

Immediately before either backend, source and target become C-contiguous
little-endian float32 arrays. Their SHA-256 digests cover raw bytes. The
reference pose is a C-contiguous little-endian float64 4×4 matrix whose digest
also covers raw bytes. Open3D receives float64 values converted only from those
canonical float32 coordinates; PCL receives the exact same float32 values.
Backend-specific normals and correspondences are allowed, but preprocessing,
clipping, downsampling, noise, dropout, coordinate changes, initial-transform
changes, and correspondence-distance changes are forbidden.

## Transform and update semantics

`T_reference` maps source frame to target/map frame. Each `T_estimated` has the
same direction. The update is
`T_delta = inverse(T_reference) @ T_estimated`. Translation update is
`norm(T_delta[0:3, 3])` metres.

Rotation update uses the v3-validated reflection-safe nearest-SO(3) SVD
projection followed by the `atan2` geodesic angle. Raw trace/acos is diagnostic
only and cannot gate. Before projection, each backend must provide a finite,
positive-determinant raw rotation with Frobenius orthogonality defect,
determinant error, and projection correction each within the frozen `1e-5`
limits. Projection cannot rescue a matrix that fails raw quality.

## Frozen backend parameters

Open3D is exactly `0.19.0+b012259`, using
`TransformationEstimationPointToPlane`, maximum correspondence distance 0.50 m,
target-only normals with radius 0.40 m/max_nn 50, and convergence
relative-fitness/relative-RMSE `1e-8` with 50 iterations. Initial transform is
`T_reference`; scene-specific switching and PCL-result access are forbidden.

PCL is exactly 1.15.1, using
`IterativeClosestPointWithNormals` and
`TransformationEstimationPointToPlaneLLS`. PCL independently estimates source
and target normals with `NormalEstimationOMP`, KSearch=50. ICP uses 0.50 m,
50 iterations, transformation/fitness epsilons `1e-10`, no reciprocal or
symmetric objective, and same-direction normals. Initial transform is
`T_reference`; scene-specific switching and Open3D-result access are forbidden.

## Solver failure definitions

An Open3D trial fails on Python/C++ exception, nonfinite transform/fitness/RMSE,
zero correspondences, raw rotation quality failure, missing required output, or
snapshot/input checksum mismatch.

A PCL trial fails on CLI exception/nonzero exit, raw non-convergence, nonfinite
transform/fitness, zero correspondences, NaN/zero source or target normals, raw
rotation quality failure, missing output, or snapshot/input checksum mismatch.
An unexecuted trial is NOT_EVALUATED and cannot be counted as zero failures.

## Quantiles, diversity, and Gate aggregation

Median is `numpy.median`. q95 is
`numpy.quantile(values, q=0.95, method="linear")`; post-result changes are
forbidden. At the per-backend all-210-trial level, solver and nonfinite counts
must be zero, q95 translation at most 0.001 m, q95 rotation at most
0.00017453292519943296 rad, and at least 95% of translations at most 0.001 m.
At each scene/backend 30-trial level, median translation must be at most
0.001 m. Completeness, pairing, input checksums, GT isolation, and prohibited
execution/RNG counters are global gates. Every gate for both backends must pass.

Snapshot diversity is aggregated per scene over its 30 snapshots. Each scene
must have at least ten unique source checksums. Unique target, unique source,
and duplicate source counts are reported. Repeated target checksums are allowed
because a geometry seed may repeat the same target map.

## Lock and future runner

The future runner requires `--protocol-lock <path>`, verifies the locked file
and SHA before any RNG/snapshot/backend boundary, and has no ignore-lock,
parameter-override, or threshold-change option. This round must not call it.
Passing this lock authorizes only a future Phase A run. It does not qualify two
backends scientifically, complete Day 1, authorize Phase B/full Development,
or authorize Confirmatory, real data, vision, or the measurement-paper
mainline.
