"""Make the repository root importable from any working directory, so the tests can
use ``from iv_model import IVModel`` exactly as the notebook does (also under a bare
``pytest tests``, which puts only ``tests/`` on ``sys.path``)."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
