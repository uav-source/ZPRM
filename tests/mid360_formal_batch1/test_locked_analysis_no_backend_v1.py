import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_scientific_analysis_modules_have_no_backend_import_or_call():
    scientific = ROOT / "experiments/mid360_formal_batch1/locked_analysis"
    denied_imports = ("open3d", "phase_a_harness.open3d_backend",
                      "phase_a_harness.pcl_backend")
    denied_calls = {"registration_icp", "run_open3d_point_to_plane",
                    "run_pcl_point_to_plane"}
    scanned = 0
    for path in sorted(scientific.glob("*.py")):
        if path.name in {"lock_v1.py", "lock_verify_v1.py"}:
            continue
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            assert not any(name == denied or name.startswith(denied + ".")
                           for name in names for denied in denied_imports), path
            if isinstance(node, ast.Call):
                function = node.func
                name = function.id if isinstance(function, ast.Name) else (
                    function.attr if isinstance(function, ast.Attribute) else ""
                )
                assert name not in denied_calls, path
    assert scanned >= 10
