from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
