# Cleanup audit

This package was produced from the uploaded `零扰动位移_最新实验代码.zip`.

## Removed

- old v1 direct Confirmatory execution scripts;
- old v2 direct Confirmatory execution/freezing/qualification scripts;
- pre-repair inventory script;
- bootstrap-repair, bootstrap-repair-r2, and bootstrap-repair-r3 scripts;
- repair-specific artifact-verifier modules and tests;
- superseded r1/r2 repair manifests and execution profiles;
- data, results, historical artifacts, caches, `.git`, Degen-LIO/ODI/AIS,
  weak-direction update, FAST-LIO2/MUN-FRL legacy pilot, and Native backend
  were already absent from the uploaded source package.

## Retained compatibility dependencies

Some paths still contain `v1`, `v2`, or `capture_range` in their names. They are
not failed result packages:

- v1/v2 scientific protocols and selected v2 analysis/publisher/verifier code
  are retained because the current v3 and version-agnostic lifecycle use them
  for frozen-science equivalence and fixture publication;
- `phase_b_generator_frozen/capture_range/` and its three YAML files are retained
  because the current zero-perturbation scene generator imports this frozen
  geometry-generation dependency;
- `qualification_json_native.py` refers to JSON-native Python values, not the
  removed Native registration backend.

## Important boundary

This is a source-code package. Formal v3 execution still requires separately
created and authenticated runtime locks/artifacts. Those historical or mutable
runtime artifacts are deliberately not bundled here.
