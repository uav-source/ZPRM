# Directional Capture Range Day 2 Protocol Amendment v1.1

## Purpose and status

This amendment resolves implementation ambiguities discovered after the Day 2
v1.0 protocol was byte-for-byte archived, but **before any Day 2 Development or
Test execution and before any Development or Test seed was consumed**.

The v1.0 protocol remains immutable:

- YAML SHA-256:
  `3b2f007c1d68d2399493ce5e15775120c4936495e4c08ef3fd0df1a88c35db65`
- Markdown SHA-256:
  `765f3eb755db18559fa0a83a1729512f2123b9ce658169542491faecc59e7bd4`
- lock commit:
  `04a20dd9fc4174558bba1f631358c6bd33ae870d`
- lock tag:
  `archive/directional-capture-range-day2-protocol-lock`

The effective Day 2 protocol is:

> v1.0 base protocol + this v1.1 amendment.

This amendment does not change Day 1 measurement definitions, seeds, success
thresholds, direction amplitudes, repeat counts or scientific gates. It only makes
previously ambiguous implementation clauses unique and executable.

## 1. Direction inventory

All scene-declared strong axes are preserved as **role declarations**.

The final inventory has two sections:

1. `declarations`: every base, weak-axis, strong-axis and icosahedron declaration;
2. `retained_directions`: unique oriented vectors used by the experiment.

Source precedence for a retained vector is:

1. base directed axis;
2. scene weak axis;
3. scene strong axis;
4. supplemental icosahedron direction.

Later duplicate declarations become aliases and do not change the retained
direction source. Antipodal vectors are never merged.

`GEOMETRY_RICH_ROOM` declares X, Y and Z as three control axes. No arbitrary pair
is selected. The rich-room scene does not participate in the ordinary two-strong-
axis separation gate. Every degraded scene still has exactly two strong axes:
world Y and world Z.

After oriented deduplication, every scene must have exactly 18 retained
directions: six directed Cartesian axes and twelve normalized icosahedron
vertices.

## 2. Synthetic scene generation

All geometry is generated from deterministic axis-aligned planar rectangles and
cuboid faces.

Planar grids use:

```text
coordinate = minimum + seeded_phase + k * spacing
```

with phase in `[0, spacing)`. Phases are derived from geometry seed, scene,
variant, primitive, map/scan role and surface axis. Samples outside the primitive
bounds are clipped. Coordinates are quantized to `1e-9 m` for checksums.

Only samples with equal quantized position **and equal normal** are deduplicated.
Coincident edge samples with different normals remain separate.

Map and scan are independently sampled from the same noiseless scene geometry.
The scan is transformed into the reference sensor frame and range-filtered to
`0.30–20.0 m`. Day 2 uses a full-360-degree synthetic field of view and does not
claim ray-occlusion or physical LiDAR scan-pattern realism.

Geometry seed affects grid phases, weak end-face retention and repeated-rib
longitudinal phase. Measurement seed affects only scan noise, map noise and scan
dropout. Geometry dimensions, reference pose and theoretical axes never depend on
a seed.

The weak end face retains exactly the points whose deterministic SHA-256-derived
uniform value lies below `0.10`.

Repeated ribs are thin cuboids attached symmetrically to both side walls. Their
longitudinal phase is deterministic in `[0, period)`.

Every snapshot must preserve the primitive inventory, map checksum, noiseless
scan checksum, noisy scan checksum and normal checksum.

## 3. Test blocks and scene aggregation

A Test block is:

```text
scene variant × geometry seed × measurement seed
```

There are six blocks per scene variant.

The five degraded scene variants used by the separation gate are:

- `LONG_CORRIDOR`
- `PARALLEL_WALLS`
- `END_FACE_TRANSITION_WEAK`
- `END_FACE_TRANSITION_ABSENT`
- `REPEATED_STRUCTURE`

For each block, the pipeline reports weak-direction angle, weak radius, strong
lower bound and separation lower bound.

A scene-variant separation passes only when:

- the median of the six separation lower bounds is at least `0.30`;
- at least four of six blocks individually reach `0.30`;
- at least four of six blocks have an uncensored weak radius.

The global gate requires at least two of the five degraded scene variants to
pass.

The Long Corridor and Parallel Walls angle gates use the median over all six Test
blocks. A non-collinear minimum-`d50` tie receives the frozen 90-degree penalty.

## 4. Geometry-rich bootstrap control

For each rich-room Test block, conservative axis radii are computed for X, Y and
Z. A block contributes a unique minimum axis only when the minimum is not tied
within `1e-12 m`.

The per-block rich-room separation is:

```text
1 - minimum_axis_radius / median(other_two_axis_lower_bounds)
```

The scene statistic is the median over all six blocks.

The 95% interval is a percentile bootstrap over the six Test blocks, with 2,000
resamples and seed `161803`.

The pseudo-weak flag is true only if the same unique Cartesian axis is the minimum
in at least five blocks, median separation is at least `0.30`, median minimum-axis
`d50` is at most `0.20 m`, and the bootstrap lower bound exceeds `0.20`.

If the required axis radii cannot be resolved because of censoring, the rich-room
control is inconclusive and does not pass.

## 5. Full reassociation versus frozen Jacobian

The predeclared comparison strata are:

- `END_FACE_TRANSITION_WEAK`, directions `+X` and `-X`;
- `END_FACE_TRANSITION_ABSENT`, directions `+X` and `-X`;
- `REPEATED_STRUCTURE`, directions `+X` and `-X`.

For exact `d50` pairs, the signed effect is:

```text
d50_full - d50_frozen
```

A sign is stable when the same non-zero sign appears in at least five of six Test
blocks. At least four blocks must have exact pairs. The magnitude statistic is the
median absolute relative difference; the threshold remains `20%`.

For curve-based comparison, the sign is the trapezoidal integral of:

```text
P_full(amplitude) - P_frozen(amplitude)
```

The same non-zero sign must appear in at least five of six blocks. At least five
blocks must be eligible. The magnitude statistic is the median blockwise maximum
absolute probability gap; the threshold remains `0.25`.

The Day 2 value gate passes when any predeclared stratum passes either exact-d50
or curve-based criteria. Directions may not be selected after viewing results.

## 6. CV and censoring

A bootstrap `d50` CV is eligible only when:

- at least 80% of bootstrap estimates are uncensored;
- the finite bootstrap mean is greater than `1e-12 m`.

A zero-mean estimate has `CV = null`, reason `ZERO_MEAN_D50`, and is excluded from
the CV median. Its standard deviation is still reported.

Right-censored values are never imputed. The aggregate denominator is all
translation scene-direction pairs with at least one exact observed `d50`.
At least 70% must be CV-eligible, and the median eligible CV must not exceed
`0.20`.

## 7. Linear matrix used by candidate search

Candidate search uses the **translation Schur information matrix** produced by the
existing production detector at the reference pose.

The candidate-search code must call the existing detector function. It may not
reimplement the matrix.

The matrix uses the detector's frozen whitening, robust weights and rotation-block
regularization, followed by symmetric averaging. Eigenvalues are sorted
ascending. Tiny negative values down to `-1e-10` may be clipped; more negative
values are an engineering failure.

For candidate comparison:

```text
lambda_min = smallest clipped eigenvalue
condition number = lambda_max / max(lambda_min, 1e-12)
normalized spectrum = eigenvalues / max(sum(eigenvalues), 1e-12)
```

One reference matrix is computed per snapshot before any initial-pose
perturbation. The same matrix applies to all directions of that snapshot.

## 8. Candidate manual review

Automatic candidates remain only candidates.

A human review record must include both raw and fitted curves, final-pose clusters,
correspondence checksum changes, final-error histograms and an explanation of any
alternate minimum or correspondence switch.

Codex may populate the review form, but may not mark a candidate `VALID` unless
`reviewed_by=human_principal_investigator` is supplied by the human researcher.

## 9. Interpretation of the previous block

The previous:

```text
DAY2_PROTOCOL_IMPLEMENTATION_BLOCKED = true
```

means only that the v1.0 specification was ambiguous. It is not a scientific
negative result and does not consume a seed.

Development remains forbidden until this amendment is byte-locked, schema-tested,
committed and tagged.
