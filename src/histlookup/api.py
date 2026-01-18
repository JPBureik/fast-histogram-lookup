"""
High-level API for fast 3D histogram lookups.

This module provides a clean, user-friendly interface that automatically
selects the fastest available implementation (Cython if compiled, otherwise
pure Python fallback).

Example
-------
>>> import numpy as np
>>> from histlookup import lookup_all_voxels, multi_shot_lookup
>>>
>>> # Single histogram
>>> hist = np.random.rand(50, 50, 50)
>>> values = lookup_all_voxels(hist)
>>>
>>> # Multiple histograms (batch processing)
>>> histograms = np.random.rand(100, 30, 30, 30)
>>> all_values = multi_shot_lookup(histograms)
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from . import lookup_py as _py

# Try to import Cython implementation, fall back to pure Python
_cy: types.ModuleType | None
try:
    from . import lookup_cy as _cy_module

    _cy = _cy_module
    _HAS_CYTHON = True
except ImportError:
    _cy = None
    _HAS_CYTHON = False


def _get_cy() -> Any:
    """Get Cython module, raising if unavailable."""
    if _cy is None:
        raise RuntimeError(
            "Cython implementation requested but not available. "
            "Install with: pip install -e ."
        )
    return _cy


def lookup_all_voxels(
    hist: NDArray[np.float64],
    *,
    use_cython: bool | None = None,
) -> NDArray[np.float64]:
    """
    Extract all voxel values from a 3D histogram in flat (row-major) order.

    Parameters
    ----------
    hist : ndarray of shape (N, N, N)
        3D histogram array. Must be cubic (equal dimensions) and float64.
    use_cython : bool, optional
        Force Cython (True) or Python (False) implementation.
        If None (default), uses Cython if available.

    Returns
    -------
    ndarray of shape (N³,)
        All histogram values in flat row-major order.

    Raises
    ------
    ValueError
        If histogram is not 3D or not cubic.
    TypeError
        If histogram is not float64.

    Example
    -------
    >>> hist = np.random.rand(30, 30, 30).astype(np.float64)
    >>> values = lookup_all_voxels(hist)
    >>> values.shape
    (27000,)
    """
    hist = _validate_histogram(hist)
    axis_len = hist.shape[0]

    if _should_use_cython(use_cython):
        return np.asarray(_get_cy().lookup_all_voxels(hist, axis_len))
    return _py.lookup_all_voxels(hist, axis_len)


def multi_shot_lookup(
    histograms: NDArray[np.float64],
    *,
    use_cython: bool | None = None,
) -> NDArray[np.float64]:
    """
    Extract all voxel values from multiple 3D histograms (batch processing).

    This is the primary use case for performance-critical workloads: processing
    many independent histogram "shots" efficiently.

    Parameters
    ----------
    histograms : ndarray of shape (n_shots, N, N, N)
        4D array of cubic histograms. Must be float64.
    use_cython : bool, optional
        Force Cython (True) or Python (False) implementation.
        If None (default), uses Cython if available.

    Returns
    -------
    ndarray of shape (n_shots, N³)
        All histogram values, one row per shot.

    Raises
    ------
    ValueError
        If histograms array is not 4D or histograms are not cubic.
    TypeError
        If array is not float64.

    Example
    -------
    >>> histograms = np.random.rand(100, 30, 30, 30).astype(np.float64)
    >>> values = multi_shot_lookup(histograms)
    >>> values.shape
    (100, 27000)
    """
    histograms = _validate_histograms_4d(histograms)
    n_shots = histograms.shape[0]
    axis_len = histograms.shape[1]

    if _should_use_cython(use_cython):
        out = np.empty((n_shots, axis_len**3), dtype=np.float64)
        _get_cy().multi_shot_lookup(histograms, n_shots, axis_len, out)
        return out
    return _py.multi_shot_lookup(histograms, n_shots, axis_len)


def get_value(
    hist: NDArray[np.float64],
    i: int,
    j: int,
    k: int,
    *,
    use_cython: bool | None = None,
) -> float:
    """
    Get a single value from a 3D histogram at the specified indices.

    Parameters
    ----------
    hist : ndarray of shape (N, N, N)
        3D histogram array.
    i, j, k : int
        Indices along each axis.
    use_cython : bool, optional
        Force Cython (True) or Python (False) implementation.
        If None (default), uses Cython if available.

    Returns
    -------
    float
        Value at hist[i, j, k].

    Example
    -------
    >>> hist = np.random.rand(10, 10, 10).astype(np.float64)
    >>> value = get_value(hist, 5, 5, 5)
    """
    if _should_use_cython(use_cython):
        return float(_get_cy().get_hist_value(hist, i, j, k))
    return _py.get_hist_value(hist, i, j, k)


def is_cython_available() -> bool:
    """
    Check if the Cython implementation is available.

    Returns
    -------
    bool
        True if Cython module is compiled and importable.

    Example
    -------
    >>> if is_cython_available():
    ...     print("Using fast Cython implementation")
    """
    return _HAS_CYTHON


def _should_use_cython(use_cython: bool | None) -> bool:
    """Determine whether to use Cython based on preference and availability."""
    if use_cython is True:
        if not _HAS_CYTHON:
            raise RuntimeError(
                "Cython implementation requested but not available. "
                "Install with: pip install -e ."
            )
        return True
    if use_cython is False:
        return False
    return _HAS_CYTHON


def _validate_histogram(hist: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate a 3D histogram array."""
    if hist.ndim != 3:
        raise ValueError(f"Histogram must be 3D, got {hist.ndim}D")
    if not (hist.shape[0] == hist.shape[1] == hist.shape[2]):
        raise ValueError(f"Histogram must be cubic, got shape {hist.shape}")
    if hist.dtype != np.float64:
        hist = hist.astype(np.float64)
    return hist


def _validate_histograms_4d(histograms: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate a 4D array of histograms."""
    if histograms.ndim != 4:
        raise ValueError(f"Histograms must be 4D, got {histograms.ndim}D")
    if not (histograms.shape[1] == histograms.shape[2] == histograms.shape[3]):
        raise ValueError(f"Each histogram must be cubic, got shape {histograms.shape}")
    if histograms.dtype != np.float64:
        histograms = histograms.astype(np.float64)
    return histograms
