"""
Pure Python 3D histogram lookup functions (baseline for benchmarking).

This module provides the same functionality as lookup_cy.pyx but using
pure Python/NumPy operations. It serves as the baseline for measuring
the Cython speedup.

These implementations represent typical "first-pass" scientific Python code
before optimization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def get_hist_value(hist: NDArray[np.float64], ix: int, iy: int, iz: int) -> float:
    """
    Get value from 3D histogram at specified indices.

    Parameters
    ----------
    hist : ndarray
        3D histogram array
    ix, iy, iz : int
        Indices along each axis

    Returns
    -------
    float
        Value at hist[ix, iy, iz]
    """
    return float(hist[ix, iy, iz])


def flat_to_3d(flat_idx: int, axis_len: int) -> tuple[int, int, int]:
    """
    Convert flat (linear) index to 3D coordinates.

    Parameters
    ----------
    flat_idx : int
        Linear index into flattened array
    axis_len : int
        Length of each axis (assumes cubic array)

    Returns
    -------
    tuple[int, int, int]
        3D coordinates (i, j, k)
    """
    ax_sq = axis_len * axis_len
    i = flat_idx // ax_sq
    j = (flat_idx - i * ax_sq) // axis_len
    k = flat_idx - i * ax_sq - j * axis_len
    return (i, j, k)


def batch_lookup(
    hist: NDArray[np.float64],
    flat_indices: NDArray[np.int32],
    axis_len: int,
) -> NDArray[np.float64]:
    """
    Perform batch lookup of histogram values for multiple flat indices.

    Parameters
    ----------
    hist : ndarray
        3D histogram array
    flat_indices : ndarray
        Array of flat indices to look up
    axis_len : int
        Length of each histogram axis

    Returns
    -------
    ndarray
        Array of looked-up values
    """
    out = np.empty(len(flat_indices), dtype=np.float64)
    for idx, flat_idx in enumerate(flat_indices):
        i, j, k = flat_to_3d(flat_idx, axis_len)
        out[idx] = hist[i, j, k]
    return out


def lookup_all_voxels(
    hist: NDArray[np.float64],
    axis_len: int,
) -> NDArray[np.float64]:
    """
    Look up all voxel values in a 3D histogram.

    Iterates over all N^3 voxels and returns their values in flat order.

    Parameters
    ----------
    hist : ndarray
        3D histogram array
    axis_len : int
        Length of each histogram axis

    Returns
    -------
    ndarray
        All histogram values in flat (row-major) order
    """
    total = axis_len**3
    result = np.empty(total, dtype=np.float64)

    for idx in range(total):
        i, j, k = flat_to_3d(idx, axis_len)
        result[idx] = hist[i, j, k]

    return result


def multi_shot_lookup(
    histograms: NDArray[np.float64],
    n_shots: int,
    axis_len: int,
) -> NDArray[np.float64]:
    """
    Process multiple histogram 'shots' - the realistic workload.

    Parameters
    ----------
    histograms : ndarray
        4D array of shape (n_shots, axis_len, axis_len, axis_len)
    n_shots : int
        Number of histogram shots to process
    axis_len : int
        Length of each histogram axis

    Returns
    -------
    ndarray
        Output array of shape (n_shots, axis_len^3)
    """
    total = axis_len**3
    out = np.empty((n_shots, total), dtype=np.float64)

    for shot in range(n_shots):
        for idx in range(total):
            i, j, k = flat_to_3d(idx, axis_len)
            out[shot, idx] = histograms[shot, i, j, k]

    return out
