"""
fast-histogram-lookup: Cython-accelerated 3D histogram bin lookups.

This package provides ~40x speedup for 3D histogram voxel lookups by moving
the hot loop from Python to Cython with typed memoryviews.

Quick Start
-----------
>>> import numpy as np
>>> from histlookup import lookup_all_voxels, multi_shot_lookup
>>>
>>> # Single histogram - just pass the array
>>> hist = np.random.rand(50, 50, 50).astype(np.float64)
>>> values = lookup_all_voxels(hist)
>>>
>>> # Batch processing - multiple histograms
>>> histograms = np.random.rand(100, 30, 30, 30).astype(np.float64)
>>> all_values = multi_shot_lookup(histograms)

The high-level API automatically selects the Cython implementation if
available, with pure Python fallback.

Low-Level Access
----------------
For direct access to implementations:

>>> from histlookup import lookup_cy, lookup_py  # Cython and Python modules
>>> from histlookup import is_cython_available   # Check if Cython is compiled
"""

from __future__ import annotations

import types

from . import lookup_py
from .api import (
    get_value,
    is_cython_available,
    lookup_all_voxels,
    multi_shot_lookup,
)

__version__ = "0.1.0"

# Import submodules - Cython import may fail if not compiled
lookup_cy: types.ModuleType | None
try:
    from . import lookup_cy as _lookup_cy

    lookup_cy = _lookup_cy
except ImportError:
    lookup_cy = None

__all__ = [
    # High-level API (recommended)
    "lookup_all_voxels",
    "multi_shot_lookup",
    "get_value",
    "is_cython_available",
    # Low-level modules
    "lookup_cy",
    "lookup_py",
]
