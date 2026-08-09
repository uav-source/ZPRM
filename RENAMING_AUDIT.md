# ZPRM Renaming and Packaging Audit

This package was derived from the uploaded `ZPRM.zip`.

Changes limited to repository presentation and packaging:

- top-level directory renamed to `ZPRM/`;
- `pyproject.toml` project name changed to `zprm-lidar`;
- README rewritten around the ZPRM identity and current paper scope;
- the non-pip-compatible `requirements-lock.txt` renamed to `ENVIRONMENT_LOCK.txt`;
- added `requirements-pip.txt` for portable development;
- removed Python bytecode and cache directories;
- regenerated `FILE_MANIFEST.txt` and `SHA256SUMS`.

Scientific source modules, frozen protocols, backend binaries, models, schemas, and experiment logic were not rewritten by this packaging step.
