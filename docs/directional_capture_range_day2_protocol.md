# Directional Capture Range MVP — Day 2 Frozen Protocol

## Status

This protocol is prospectively frozen **before any Day 2 development or test run**.
It is not a retroactive reconstruction of an already executed experiment. No Day 2
test seed may be consumed until the protocol, direction inventory, source tree and
test lock have been committed and hashed.

Required Day 1 baseline:

- branch: `feature/directional-capture-range-mvp`
- commit: `91f2d07f0b112a339702ea72f4055404cd356fdf`
- tag: `archive/directional-capture-range-day1-pass`
- Day 1 Engineering Gate: PASS

## Scientific question

Day 2 asks whether the algorithm-conditioned empirical directional capture range:

1. identifies known translation-weak directions in controlled geometry;
2. separates weak and strong directions by a meaningful margin;
3. exhibits behavior under full reassociation that a frozen-Jacobian approximation cannot express;
4. is sufficiently repeatable to justify Day 3.

Day 2 does **not** establish novelty, real-data validity or a Measurement paper.

## Frozen split

Development geometry seeds: `1101, 1103, 1107`

Development measurement seeds: `2101, 2111`

Test geometry seeds: `1201, 1213, 1223`

Test measurement seeds: `2203, 2213`

Development and Test sets must remain disjoint. Test seeds may not be evaluated,
visualized or used for debugging before `day2_test_lock.json` is committed.

## Frozen scenes

The world-frame X axis is the corridor direction, Y is lateral and Z is vertical.

The seven executable scene variants are:

1. `GEOMETRY_RICH_ROOM`
2. `LONG_CORRIDOR`
3. `PARALLEL_WALLS`
4. `END_FACE_TRANSITION/PRESENT`
5. `END_FACE_TRANSITION/WEAK`
6. `END_FACE_TRANSITION/ABSENT`
7. `REPEATED_STRUCTURE`

Exact geometric dimensions and sampling parameters are specified in
`configs/capture_range/day2_synthetic_locked.yaml`.

`LONG_CORRIDOR`, `PARALLEL_WALLS`, `END_FACE_TRANSITION/WEAK`,
`END_FACE_TRANSITION/ABSENT` and `REPEATED_STRUCTURE` use the world X axis as
their theoretical weak translation axis. `GEOMETRY_RICH_ROOM` and
`END_FACE_TRANSITION/PRESENT` have no assumed weak direction for the direction
accuracy gate.

## Frozen directions

The directed set is the union of:

- ±X, ±Y, ±Z;
- the scene weak axis and its antipode, where defined;
- two scene strong axes and their antipodes;
- twelve normalized icosahedron vertices.

Vectors are normalized and exact oriented duplicates are removed. Antipodal
directions are never merged for measurement. Axis sign is ignored only when
computing the final angular error against a theoretical weak axis.

A complete `direction_inventory_lock.json` must be generated and committed before
Development begins.

## Frozen amplitudes

Translation, metres:

`0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80`

Rotation, degrees:

`0.00, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00`

Translation is the Day 2 primary scientific gate. Rotation must still run and be
reported, but does not determine Day 2 survival unless it reveals an engineering
error.

## Repeat model

Development uses five repeats per direction and amplitude. Test uses ten.

Each repeat applies the prospectively frozen measurement perturbation model:

- scan point Gaussian noise: 0.003 m;
- map point Gaussian noise: 0.001 m;
- scan dropout: 1%;
- map dropout: 0%;
- no reference-pose noise.

The same realization must be shared by full reassociation and frozen-Jacobian
methods. Method names must not enter random seeds.

## Weak-direction extraction

The empirical weak translation direction is extracted from full-reassociation
`d50` only.

Right-censored `d50` values remain `null` in stored data and are treated as
positive infinity only for ordering. They may never be replaced by the largest
sampled amplitude as if they were exact.

If several minimum-`d50` directions are antipodal or collinear, they represent one
axis. If non-collinear directions tie at the minimum, the weak direction is marked
ambiguous and receives a 90-degree error in the primary direction gate. The
pipeline may not choose the tied direction closest to ground truth.

## Strong/weak separation with censoring

The weak-axis capture radius is the conservative minimum of the positive and
negative weak-axis sides.

For each strong axis:

- if an exact side exists, use the conservative exact side;
- if both sides are right-censored, the largest sampled amplitude is used only as a lower bound, never as an exact `d50`.

The strong radius is the median of the two strong-axis lower bounds.

The gated quantity is therefore a conservative lower bound:

```text
s_lower = 1 - d50_weak / d50_strong_lower
```

The gate requires the weak radius itself to be uncensored.

## Repeatability with censoring

Repeatability is estimated with 2,000 bootstrap resamples of repeat indices,
recomputing the recovery probability, isotonic curve and `d50` on every resample.

A scene-direction pair is eligible for a coefficient of variation only when at
least 80% of bootstrap `d50` estimates are uncensored. At least 70% of evaluated
scene-direction pairs must be eligible. Right-censored bootstrap values are not
silently converted into finite values.

## Geometry-rich control

A stable pseudo-weak direction is flagged only if all conditions hold:

1. the same Cartesian axis is the minimum in at least five of six Test blocks;
2. its median separation ratio from the other axes is at least 0.30;
3. its median minimum-axis `d50` is at most 0.20 m;
4. the bootstrap lower confidence bound of separation exceeds 0.20.

A flagged pseudo-weak direction fails the geometry-rich control. No post-hoc
geometric explanation may override this rule.

## Full reassociation versus frozen Jacobian

The primary comparison is evaluated in `END_FACE_TRANSITION` and
`REPEATED_STRUCTURE`.

A numerical relative `d50` difference is computed only when both radii are exact:

```text
abs(d50_full - d50_frozen) / max(min(d50_full, d50_frozen), 1e-6)
```

The value gate passes if either:

- median relative exact-pair difference is at least 20%; or
- maximum absolute recovery-probability curve gap is at least 0.25.

The effect must have a stable directional sign in at least 67% of Test blocks.

## Candidate non-equivalence search

Final candidates are searched only in the locked Test pool. Development candidates
are diagnostic only.

Pairs must use the same canonical translation direction and satisfy all three
linear-similarity conditions:

- normalized-eigenvalue cosine similarity ≥ 0.98;
- absolute log condition-number ratio ≤ 0.15;
- absolute log minimum-eigenvalue ratio ≤ 0.15.

Both `d50` values must be exact, and their relative difference must be at least
30%.

Automatic selection does not establish scientific validity. Every candidate must
receive one manual status:

- `VALID`
- `REJECT_NUMERICAL`
- `REJECT_GEOMETRY_NOT_COMPARABLE`
- `REJECT_NO_LOCAL_MINIMUM_EVIDENCE`
- `INCONCLUSIVE`

The record must preserve recovery curves, correspondence checksum changes and
evidence for an alternate local minimum.

## Frozen gates

### Engineering

All tests pass; split isolation and locks pass; no ground truth enters registration;
full reassociation really executes; frozen Jacobian stays isolated; no trial is
missing; all outputs are finite; paired inputs share checksums; final worktree is
clean.

### Directionality

- `LONG_CORRIDOR` median weak-axis angle error ≤ 10°;
- `PARALLEL_WALLS` median weak-axis angle error ≤ 10°;
- at least two degraded scenes have conservative separation ≥ 0.30.

### Geometry-rich control

The pseudo-weak flag must be false.

### Full reassociation value

The full-versus-frozen gate above must pass.

### Preliminary repeatability

Median eligible `d50` CV ≤ 0.20 and eligible fraction ≥ 0.70.

### Day 3 authorization

Day 3 requires:

- Engineering PASS;
- Directionality PASS;
- Full-reassociation value observed.

Repeatability and candidate non-equivalence are reported on Day 2 but are not
individually required for Day 3 authorization. They become hard gates later.

## Prohibitions

No Day 1 definition may be modified. No threshold, scene, seed, direction,
amplitude or repeat count may be changed after viewing Development or Test
results. Failed seeds and non-monotonic curves must remain. No real data, visual
data, ODI modification, FAST-LIO2 estimator modification, weak-direction update
or novelty claim is permitted. No push is permitted.
