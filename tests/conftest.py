import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os

import pytest

import openramen.runtime as _runtime


def _stress(on: bool) -> None:
    """Make the scheduler hand every node to another thread, however cheap."""
    _runtime._EVERYTHING_HEAVY = on
    _runtime._CHEAP = -1.0 if on else 200e-6


if os.environ.get("OPENRAMEN_STRESS"):
    # Run the whole suite with every node dispatched to the thread pool.
    _stress(True)


@pytest.fixture
def stress():
    """Within one test: every node goes to the thread pool."""
    before = (_runtime._EVERYTHING_HEAVY, _runtime._CHEAP)
    _stress(True)
    yield
    _runtime._EVERYTHING_HEAVY, _runtime._CHEAP = before
