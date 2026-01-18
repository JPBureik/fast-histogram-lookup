# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Cython-accelerated 3D histogram lookup functions.

This module provides optimized functions for accessing elements in 3D
histograms using typed memoryviews, bypassing Python's interpreter overhead
for significant speedup in tight loops.

Key optimizations:
- Typed memoryviews (double[:, :, :]) for direct C-level array access
- Bounds checking disabled via compiler directive
- Negative index wrapping disabled
- Pure C integer arithmetic for index conversion
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport floor

cnp.import_array()


cpdef double get_hist_value(double[:, :, :] hist, int ix, int iy, int iz) noexcept:
    """
    Get value from 3D histogram at specified indices.

    Uses typed memoryview for O(1) direct memory access without Python
    overhead or bounds checking.

    Parameters
    ----------
    hist : memoryview (double[:, :, :])
        3D histogram data as a typed memoryview
    ix, iy, iz : int
        Indices along each axis

    Returns
    -------
    double
        Value at hist[ix, iy, iz]
    """
    return hist[ix, iy, iz]


cpdef tuple flat_to_3d(int flat_idx, int axis_len):
    """
    Convert flat (linear) index to 3D coordinates.

    For a 3D array of shape (N, N, N) stored in row-major (C) order,
    converts a flat index to (i, j, k) coordinates.

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
    cdef int i, j, k
    cdef int ax_sq = axis_len * axis_len

    i = <int>floor(flat_idx / ax_sq)
    j = <int>floor((flat_idx - i * ax_sq) / axis_len)
    k = flat_idx - i * ax_sq - j * axis_len

    return (i, j, k)


cpdef void batch_lookup(
    double[:, :, :] hist,
    int[:] flat_indices,
    int axis_len,
    double[:] out
) noexcept:
    """
    Perform batch lookup of histogram values for multiple flat indices.

    This is the primary hot loop optimization: iterates over many indices
    and performs fast histogram lookups without returning to Python.

    Parameters
    ----------
    hist : memoryview (double[:, :, :])
        3D histogram data
    flat_indices : memoryview (int[:])
        Array of flat indices to look up
    axis_len : int
        Length of each histogram axis
    out : memoryview (double[:])
        Pre-allocated output array for results
    """
    cdef int n = flat_indices.shape[0]
    cdef int idx, i, j, k
    cdef int ax_sq = axis_len * axis_len

    for idx in range(n):
        flat_idx = flat_indices[idx]
        i = <int>floor(flat_idx / ax_sq)
        j = <int>floor((flat_idx - i * ax_sq) / axis_len)
        k = flat_idx - i * ax_sq - j * axis_len
        out[idx] = hist[i, j, k]


cpdef double[:] lookup_all_voxels(double[:, :, :] hist, int axis_len):
    """
    Look up all voxel values in a 3D histogram.

    Iterates over all N^3 voxels and returns their values in flat order.
    This simulates the workload of processing all bins in a large histogram.

    Parameters
    ----------
    hist : memoryview (double[:, :, :])
        3D histogram data
    axis_len : int
        Length of each histogram axis

    Returns
    -------
    memoryview (double[:])
        All histogram values in flat (row-major) order
    """
    cdef int total = axis_len * axis_len * axis_len
    cdef double[:] result = np.empty(total, dtype=np.float64)
    cdef int idx, i, j, k
    cdef int ax_sq = axis_len * axis_len

    for idx in range(total):
        i = <int>floor(idx / ax_sq)
        j = <int>floor((idx - i * ax_sq) / axis_len)
        k = idx - i * ax_sq - j * axis_len
        result[idx] = hist[i, j, k]

    return result


cpdef void multi_shot_lookup(
    double[:, :, :, :] histograms,
    int n_shots,
    int axis_len,
    double[:, :] out
) noexcept:
    """
    Process multiple histogram 'shots' - the realistic workload.

    In scientific applications, we often have many independent histograms
    (e.g., repeated measurements) and need to extract all voxel values
    from each. This function processes all shots efficiently.

    Parameters
    ----------
    histograms : memoryview (double[:, :, :, :])
        4D array of shape (n_shots, axis_len, axis_len, axis_len)
    n_shots : int
        Number of histogram shots to process
    axis_len : int
        Length of each histogram axis
    out : memoryview (double[:, :])
        Pre-allocated output array of shape (n_shots, axis_len^3)
    """
    cdef int shot, idx, i, j, k
    cdef int total = axis_len * axis_len * axis_len
    cdef int ax_sq = axis_len * axis_len

    for shot in range(n_shots):
        for idx in range(total):
            i = <int>floor(idx / ax_sq)
            j = <int>floor((idx - i * ax_sq) / axis_len)
            k = idx - i * ax_sq - j * axis_len
            out[shot, idx] = histograms[shot, i, j, k]
