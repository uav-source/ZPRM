# Mid-360 unified workspace

The canonical working repository is `/home/lj/ZPRM` on branch
`formal/mid360-batch1-unified-v1`.

All Mid-360 source code, frozen configuration, protocols, acquisition tools,
tests, Pilot reports, and Formal Batch-1 preacquisition artifacts are available
inside this repository. The raw single bag is `mid360.bag`; the two-scene
MAP/QUERY bags are under `bag/`.

The consolidation copied 76 files (11,818,133 bytes). Source and target file
manifests have the identical SHA256
`1323a00f2b90abf73536afda66861080f2b8c90429883e2d3edf900dfd42ca6b`.

The retained runtime cache is now workspace-local at
`/home/lj/ZPRM/zero_perturbation_runtime`. It is explicitly ignored by Git, so
mutable state remains separate from commits while the whole active workspace
lives under one folder. Historical result files keep their original absolute
paths as provenance records.

The former `/home/lj/zero_perturbation_data` Boreas dataset (397 MB) was moved
to the system trash at the user's request on 2026-08-19 and has since been
permanently cleared. Boreas preparation must redownload it into
`/home/lj/ZPRM/zero_perturbation_data`.

Also on 2026-08-19, the remaining Boreas runtime directories and the 677 MB
synthetic confirmatory snapshot cache were permanently deleted. The runtime
was reduced from 769 MB to 43 MB. Both Mid-360 Pilot runtime directories and
the non-cache confirmatory result records were retained.

On 2026-08-19, the user explicitly authorized permanent deletion of the
41,998,817,280-byte incomplete Boreas Stage-2 replay intermediate
`map/transformed_xyz.f64le`. It was unrelated to Mid-360/FMB1. Resuming that
Boreas preparation requires redownloading and rebuilding the replay file.

## Repository layout

```text
configs/mid360_pilot_config.json
experiments/mid360_capture_basin/
experiments/mid360_controlled_perturbation/
experiments/mid360_formal_batch1/
results/mid360_capture_basin_pilot/
results/mid360_controlled_perturbation_pilot/
results/mid360_formal_batch1/
src/phase_a_harness/mid360_pilot/
src/phase_a_harness/mid360_two_scene_pilot/
tests/mid360_formal_batch1/
tools/mid360_formal_batch1/
zero_perturbation_runtime/       # local mutable state; Git-ignored
```

The three older Git worktrees were removed after their copied content was
verified byte-for-byte. Their Git branches and commit history remain available.
New Mid-360 work should use `/home/lj/ZPRM` only.
