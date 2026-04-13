"""Project startup tweaks for the local Python environment.

This removes stray user-level Python paths that can shadow the project venv
and break scientific imports like SciPy / scikit-fuzzy.
"""

from __future__ import annotations

import os
import sys


def _normalize(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


_BAD_PATH_MARKERS = (
    _normalize(r"C:\Users\asus\Downloads"),
)

sys.path[:] = [
    path for path in sys.path
    if not any(marker in _normalize(path) for marker in _BAD_PATH_MARKERS)
]