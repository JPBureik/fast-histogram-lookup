"""
fast-histogram-lookup: Cython-accelerated 3D histogram bin lookups.

This package demonstrates a ~50-100x speedup for 3D histogram bin lookups
by moving the hot loop from Python to Cython with typed memoryviews.

Example
-------
>>> import numpy as np
>>> from histlookup import lookup_cy, lookup_py
>>>
>>> # Create a 3D histogram (e.g., 50x50x50 bins)
>>> hist = np.random.rand(50, 50, 50)
>>>
>>> # Cython version (fast)
>>> result_cy = lookup_cy.lookup_all_voxels(hist, 50)
>>>
>>> # Python version (baseline)
>>> result_py = lookup_py.lookup_all_voxels(hist, 50)
"""

from __future__ import annotations

import types

from . import lookup_py

__version__ = "0.1.0"

# Import submodules - Cython import may fail if not compiled
lookup_cy: types.ModuleType | None
try:
    from . import lookup_cy as _lookup_cy

    lookup_cy = _lookup_cy
except ImportError:
    lookup_cy = None

__all__ = ["lookup_cy", "lookup_py"]
