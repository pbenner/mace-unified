from __future__ import annotations

import sys
from pathlib import Path


def _prepend(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent

_prepend(ROOT / "src")
_prepend(WORKSPACE / "mace")
_prepend(WORKSPACE / "mace-jax")
