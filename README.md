# ZPRM

**Zero-Perturbation Registration Measurement**

ZPRM is the experiment repository for the paper project:

> **Measuring Zero-Perturbation Displacement and Reassociation Effects in LiDAR Scan-to-Map Registration**

The repository measures whether scan-to-map registration produces a nonzero pose update when the initial pose is already equal to the reference pose, and studies how this displacement changes with scene geometry, sampling/noise conditions, backend implementation, and correspondence reassociation.

## Current scientific scope

Included:

- Open3D point-to-plane registration;
- PCL point-to-plane registration and the frozen CLI;
- Phase A ideal-matched controls;
- Phase B signal exploration;
- full Synthetic Development;
- correspondence turnover, residual-change, and normal-change diagnostics;
- H1–H6 confirmatory analysis and independent verification;
- v3 compatibility entry points;
- version-agnostic formal lifecycle code;
- frozen protocols, models, schemas, and tests.

Excluded:

- Degen-LIO, ODI/AIS, weak-direction update, and capture-range paper routes;
- FAST-LIO2/MUN-FRL legacy pilots;
- Native registration backend;
- historical failed run outputs and failure archives;
- generated `data/`, `results/`, and mutable runtime artifacts.

Some files with `v2` or `capture_range` in their names remain because the current v3/formal lifecycle and frozen scene generator import them for compatibility and scientific-equivalence checks. They are dependencies, not revived paper routes.

## Repository layout

```text
bin/            Frozen PCL executables
configs/        Active and dependency configurations
frozen_assets/  Frozen protocols, schemas, models, and manifests
protocols/      Human-readable scientific protocols
scripts/        Development, qualification, and formal lifecycle entry points
src/            ZPRM implementation (`phase_a_harness` Python package)
tests/          Mathematical, lifecycle, and scientific-contract tests
```

## Environment

The formal environment is frozen in `ENVIRONMENT_LOCK.txt`:

```text
Python 3.11.15
Open3D 0.19.0+b012259
PCL 1.15.1
```

`ENVIRONMENT_LOCK.txt` is an environment record, **not** a pip requirements file.

For portable development and static/unit testing:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pip.txt
python -m pytest -q
```

The public PyPI Open3D 0.19.0 package is suitable for development checks but is not guaranteed to be bitwise identical to the frozen `0.19.0+b012259` build used by the formal protocol.

## Important execution boundary

This ZIP is a clean source snapshot. It intentionally does not include mutable formal runtime directories, generated snapshots, trial results, or machine-specific authenticated locks. Some formal scripts retain frozen absolute-path and Git-identity contracts; they must be rebound or executed in the originally qualified environment before a formal Confirmatory run.

Development findings must not be presented as Confirmatory conclusions until the frozen independent H1–H6 run is complete.
