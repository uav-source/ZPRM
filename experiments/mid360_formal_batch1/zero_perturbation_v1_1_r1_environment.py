"""Environment evidence for the FMB1 zero-perturbation v1.1-R1 track.

The collector performs metadata/version and ELF dependency probes only.  It
never imports Open3D and never invokes the PCL registration executable.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "mid360_fmb1_zero_perturbation_environment_v1_1_r1"
EXPECTED_VERSIONS = {
    "python": "3.11.15",
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "open3d": "0.19.0+b012259",
    "pcl": "1.15.1",
}
BACKEND_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
PCL_ENVIRONMENT_PREFIX = Path(
    "/home/lj/.local/share/degen-lio-micromamba/envs/degen-lio-pcl-backend"
)
PCL_PKG_CONFIG_SHA256 = (
    "7590aa30f53362ced6ba582bd8b34cfe3fca67e0f84fb6b01a0542d3c1bc4e03"
)
PCL_COMMON_PC_SHA256 = (
    "7af03303621644e94bcb6c1ffdd54e01ee94e203ce54cf5e54ff5d1eb4bef73b"
)
OPEN3D_INIT_PATH = Path(
    "/home/lj/.local/share/degen-lio-micromamba/envs/degen-lio-zprm-py311/"
    "lib/python3.11/site-packages/open3d/__init__.py"
)
OPEN3D_INIT_SHA256 = (
    "0bcb34895ce3d438683e6c7a7913aafee42b42b3dee948ddfacd63ce35bd1946"
)


class R1EnvironmentError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise R1EnvironmentError(f"FMB1_R1_ENVIRONMENT_FAIL: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _metadata_version(distribution: str) -> str:
    candidates = (distribution, "open3d-cpu") if distribution == "open3d" else (distribution,)
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "NOT_INSTALLED"


def _static_open3d_version(path: Path = OPEN3D_INIT_PATH) -> str:
    lexical = Path(path)
    if lexical.is_symlink():
        _fail("Open3D version source must not be a symlink")
    resolved = lexical.resolve(strict=True)
    spec = importlib.util.find_spec("open3d")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != resolved:
        _fail("current interpreter does not resolve Open3D to the frozen version source")
    if sha256_file(resolved) != OPEN3D_INIT_SHA256:
        _fail("Open3D version source SHA changed")
    tree = ast.parse(resolved.read_text(encoding="utf-8"), filename=str(resolved))
    values = []
    for statement in tree.body:
        if (isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__"
                        for target in statement.targets)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)):
            values.append(statement.value.value)
    if values != [EXPECTED_VERSIONS["open3d"]]:
        _fail(f"Open3D static version literal differs: {values}")
    return values[0]


def _probe_command(arguments: list[str], *, environment: Mapping[str, str] | None = None) -> str:
    completed = subprocess.run(
        arguments,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        env=None if environment is None else dict(environment),
    )
    return completed.stdout.strip()


def collect_environment_manifest(
    repository: Path,
    *,
    pcl_executable: Path | None = None,
    observed_versions: Mapping[str, str] | None = None,
    pcl_version_text: str | None = None,
    ldd_text: str | None = None,
    pcl_environment_prefix: Path = PCL_ENVIRONMENT_PREFIX,
) -> dict[str, Any]:
    """Collect a deterministic environment manifest without a backend call.

    Tests and offline audits may inject probe values.  The live collector uses
    package metadata, ``pkg-config --modversion pcl_common`` and ``ldd``; it
    deliberately does not execute ``pcl_point_to_plane_cli`` even with a
    version flag.
    """

    root = Path(repository).resolve(strict=True)
    pcl_lexical = pcl_executable or root / "bin/pcl_point_to_plane_cli"
    if Path(pcl_lexical).is_symlink():
        _fail("PCL CLI must not be a symlink")
    pcl_path = Path(pcl_lexical).resolve(strict=True)
    if not pcl_path.is_file():
        _fail("PCL CLI must be a regular non-symlink file")
    prefix = Path(pcl_environment_prefix).resolve(strict=True)
    pkg_config = prefix / "bin/pkg-config"
    pcl_common_pc = prefix / "lib/pkgconfig/pcl_common.pc"
    if (not pkg_config.is_file() or pkg_config.is_symlink()
            or not pcl_common_pc.is_file() or pcl_common_pc.is_symlink()):
        _fail("frozen PCL pkg-config evidence is missing/symlinked")
    if sha256_file(pkg_config) != PCL_PKG_CONFIG_SHA256:
        _fail("frozen PCL pkg-config executable SHA changed")
    if sha256_file(pcl_common_pc) != PCL_COMMON_PC_SHA256:
        _fail("frozen pcl_common.pc SHA changed")
    probe_environment = {
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
        "PKG_CONFIG_PATH": str(prefix / "lib/pkgconfig"),
        "LD_LIBRARY_PATH": str(prefix / "lib"),
        "LANG": "C",
        "LC_ALL": "C",
    }
    versions = dict(
        observed_versions
        or {
            "python": platform.python_version(),
            "numpy": _metadata_version("numpy"),
            "scipy": _metadata_version("scipy"),
            "open3d": _static_open3d_version(),
        }
    )
    if pcl_version_text is None:
        pcl_version_text = _probe_command(
            [str(pkg_config), "--modversion", "pcl_common"],
            environment=probe_environment,
        )
    versions["pcl"] = pcl_version_text.strip()
    if ldd_text is None:
        ldd_text = _probe_command(["/usr/bin/ldd", str(pcl_path)], environment=probe_environment)
    ldd_lines = [line.rstrip() for line in ldd_text.splitlines() if line.strip()]
    missing = [line for line in ldd_lines if "not found" in line.lower()]

    contract_path = (root / "frozen_assets/backend_parameter_contract.json").resolve(
        strict=True
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, Mapping):
        _fail("backend parameter contract must be a mapping")
    open3d = contract.get("open3d")
    pcl = contract.get("pcl")
    if not isinstance(open3d, Mapping) or not isinstance(pcl, Mapping):
        _fail("backend contract lacks Open3D/PCL sections")
    open3d_parameters = open3d.get("parameters")
    pcl_parameters = pcl.get("parameters")
    canonical_pass = bool(
        isinstance(open3d_parameters, Mapping)
        and isinstance(pcl_parameters, Mapping)
        and canonical_sha256(open3d_parameters) == open3d.get("canonical_sha256")
        and canonical_sha256(pcl_parameters) == pcl.get("canonical_sha256")
    )
    version_checks = {
        key: versions.get(key) == expected
        for key, expected in EXPECTED_VERSIONS.items()
    }
    contract_sha = sha256_file(contract_path)
    payload = {
        "schema": SCHEMA,
        "execution_scope": "REAL_FORMAL_ENVIRONMENT_QUALIFICATION",
        "versions": versions,
        "expected_versions": dict(EXPECTED_VERSIONS),
        "version_checks": version_checks,
        "pcl_cli": {
            "path": str(pcl_path),
            "sha256": sha256_file(pcl_path),
            "bytes": pcl_path.stat().st_size,
            "executable": bool(pcl_path.stat().st_mode & 0o111),
            "version_probe": "PKG_CONFIG_PCL_COMMON_NO_CLI_EXECUTION",
            "ldd_output": ldd_lines,
            "ldd_output_sha256": hashlib.sha256(
                ("\n".join(ldd_lines) + "\n").encode("utf-8")
            ).hexdigest(),
            "ldd_missing_dependencies": missing,
            "ldd_audit_pass": not missing and bool(ldd_lines),
            "registration_executable_invocation_count": 0,
            "frozen_environment_prefix": str(prefix),
            "pkg_config_executable": str(pkg_config),
            "pkg_config_executable_sha256": PCL_PKG_CONFIG_SHA256,
            "pcl_common_pc": str(pcl_common_pc),
            "pcl_common_pc_sha256": PCL_COMMON_PC_SHA256,
            "probe_path": probe_environment["PATH"],
            "probe_pkg_config_path": probe_environment["PKG_CONFIG_PATH"],
            "probe_ld_library_path": probe_environment["LD_LIBRARY_PATH"],
        },
        "backend_contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
            "expected_sha256": BACKEND_CONTRACT_SHA256,
            "canonical_open3d_sha256": open3d.get("canonical_sha256"),
            "canonical_pcl_sha256": pcl.get("canonical_sha256"),
            "canonical_parameter_hashes_verified": canonical_pass,
        },
        "open3d_import_count": 0,
        "open3d_version_evidence": {
            "probe": "STATIC_AST_LITERAL_NO_IMPORT",
            "path": str(OPEN3D_INIT_PATH),
            "sha256": OPEN3D_INIT_SHA256,
            "version": versions["open3d"],
        },
        "pcl_registration_call_count": 0,
    }
    payload["qualification_pass"] = bool(
        all(version_checks.values())
        and payload["pcl_cli"]["executable"]
        and payload["pcl_cli"]["ldd_audit_pass"]
        and contract_sha == BACKEND_CONTRACT_SHA256
        and canonical_pass
    )
    return payload


def verify_environment_manifest(
    payload: Mapping[str, Any], repository: Path, *, remeasure_versions: bool = True
) -> dict[str, Any]:
    """Verify frozen environment evidence and current immutable file hashes."""

    if payload.get("schema") != SCHEMA or payload.get("qualification_pass") is not True:
        _fail("environment schema/status is not qualified")
    versions = payload.get("versions")
    if not isinstance(versions, Mapping) or dict(versions) != EXPECTED_VERSIONS:
        _fail("environment versions differ from the exact R1 contract")
    if remeasure_versions:
        observed = {
            "python": platform.python_version(),
            "numpy": _metadata_version("numpy"),
            "scipy": _metadata_version("scipy"),
            "open3d": _static_open3d_version(),
        }
        for key, value in observed.items():
            if value != EXPECTED_VERSIONS[key]:
                _fail(f"current {key} version changed: {value}")
    root = Path(repository).resolve(strict=True)
    pcl = payload.get("pcl_cli")
    contract = payload.get("backend_contract")
    if not isinstance(pcl, Mapping) or not isinstance(contract, Mapping):
        _fail("environment file bindings are missing")
    open3d_evidence = payload.get("open3d_version_evidence")
    if not isinstance(open3d_evidence, Mapping):
        _fail("Open3D static version evidence is missing")
    if (open3d_evidence.get("probe") != "STATIC_AST_LITERAL_NO_IMPORT"
            or open3d_evidence.get("path") != str(OPEN3D_INIT_PATH)
            or open3d_evidence.get("sha256") != OPEN3D_INIT_SHA256
            or open3d_evidence.get("version") != EXPECTED_VERSIONS["open3d"]
            or _static_open3d_version() != EXPECTED_VERSIONS["open3d"]):
        _fail("Open3D static version evidence differs")
    pcl_lexical = Path(str(pcl.get("path")))
    contract_lexical = Path(str(contract.get("path")))
    if pcl_lexical.is_symlink() or contract_lexical.is_symlink():
        _fail("environment file binding uses a symlink")
    pcl_path = pcl_lexical.resolve(strict=True)
    contract_path = contract_lexical.resolve(strict=True)
    for path, label in ((pcl_path, "PCL CLI"), (contract_path, "backend contract")):
        try:
            path.relative_to(root)
        except ValueError:
            _fail(f"{label} escapes repository")
    if sha256_file(pcl_path) != pcl.get("sha256"):
        _fail("PCL CLI SHA changed")
    if not bool(pcl_path.stat().st_mode & 0o111):
        _fail("PCL CLI executable bit changed")
    if pcl.get("ldd_audit_pass") is not True or pcl.get("ldd_missing_dependencies") != []:
        _fail("PCL ldd audit does not pass")
    if pcl.get("registration_executable_invocation_count") != 0:
        _fail("environment collector invoked the registration executable")
    prefix = Path(str(pcl.get("frozen_environment_prefix"))).resolve(strict=True)
    pkg_config = Path(str(pcl.get("pkg_config_executable")))
    common_pc = Path(str(pcl.get("pcl_common_pc")))
    if pkg_config.is_symlink() or common_pc.is_symlink():
        _fail("frozen PCL metadata evidence uses a symlink")
    if pkg_config.resolve(strict=True) != prefix / "bin/pkg-config":
        _fail("frozen PCL pkg-config path changed")
    if common_pc.resolve(strict=True) != prefix / "lib/pkgconfig/pcl_common.pc":
        _fail("frozen pcl_common.pc path changed")
    if sha256_file(pkg_config) != PCL_PKG_CONFIG_SHA256 or pcl.get("pkg_config_executable_sha256") != PCL_PKG_CONFIG_SHA256:
        _fail("frozen PCL pkg-config SHA changed")
    if sha256_file(common_pc) != PCL_COMMON_PC_SHA256 or pcl.get("pcl_common_pc_sha256") != PCL_COMMON_PC_SHA256:
        _fail("frozen pcl_common.pc SHA changed")
    expected_probe = {
        "probe_path": f"{prefix / 'bin'}:/usr/bin:/bin",
        "probe_pkg_config_path": str(prefix / "lib/pkgconfig"),
        "probe_ld_library_path": str(prefix / "lib"),
    }
    if any(pcl.get(key) != value for key, value in expected_probe.items()):
        _fail("frozen PCL probe environment changed")
    if remeasure_versions:
        live_environment = {
            "PATH": expected_probe["probe_path"],
            "PKG_CONFIG_PATH": expected_probe["probe_pkg_config_path"],
            "LD_LIBRARY_PATH": expected_probe["probe_ld_library_path"],
            "LANG": "C", "LC_ALL": "C",
        }
        live_ldd = _probe_command(
            ["/usr/bin/ldd", str(pcl_path)], environment=live_environment
        )
        live_lines = [line for line in live_ldd.splitlines() if line.strip()]
        if not live_lines or any("not found" in line.lower() for line in live_lines):
            _fail("live PCL ldd dependency audit failed")
    if sha256_file(contract_path) != BACKEND_CONTRACT_SHA256:
        _fail("backend contract SHA changed")
    if contract.get("sha256") != BACKEND_CONTRACT_SHA256:
        _fail("environment backend contract binding changed")
    if contract.get("canonical_parameter_hashes_verified") is not True:
        _fail("backend canonical parameter hashes were not verified")
    return {
        "schema": "mid360_fmb1_zero_perturbation_environment_verification_v1_1_r1",
        "pass": True,
        "exact_version_match": True,
        "pcl_cli_sha256": pcl.get("sha256"),
        "backend_parameter_contract_sha256": BACKEND_CONTRACT_SHA256,
        "backend_calls": 0,
    }
