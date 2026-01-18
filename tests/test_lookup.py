"""Tests for histogram lookup functions."""

import numpy as np
import pytest

from histlookup import (
    get_value,
    is_cython_available,
    lookup_all_voxels,
    lookup_py,
    multi_shot_lookup,
)

# Cython module may not be compiled yet
try:
    from histlookup import lookup_cy

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


class TestFlatTo3D:
    """Tests for flat index to 3D coordinate conversion."""

    def test_first_element(self):
        """Index 0 should map to (0, 0, 0)."""
        assert lookup_py.flat_to_3d(0, 10) == (0, 0, 0)

    def test_last_element(self):
        """Last index should map to (n-1, n-1, n-1)."""
        n = 10
        assert lookup_py.flat_to_3d(n**3 - 1, n) == (n - 1, n - 1, n - 1)

    def test_middle_element(self):
        """Test a middle element."""
        # For axis_len=10: index 123 = 1*100 + 2*10 + 3 -> (1, 2, 3)
        assert lookup_py.flat_to_3d(123, 10) == (1, 2, 3)

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython module not compiled")
    def test_cython_matches_python(self):
        """Cython implementation should match Python."""
        for axis_len in [5, 10, 20]:
            for flat_idx in [0, 1, axis_len, axis_len**2, axis_len**3 - 1]:
                py_result = lookup_py.flat_to_3d(flat_idx, axis_len)
                cy_result = lookup_cy.flat_to_3d(flat_idx, axis_len)
                assert py_result == cy_result


class TestGetHistValue:
    """Tests for histogram value retrieval."""

    @pytest.fixture
    def hist_3d(self):
        """Create a test 3D histogram."""
        np.random.seed(42)
        return np.random.rand(10, 10, 10)

    def test_known_value(self, hist_3d):
        """Test retrieval of known value."""
        hist_3d[1, 2, 3] = 42.0
        assert lookup_py.get_hist_value(hist_3d, 1, 2, 3) == 42.0

    def test_corners(self, hist_3d):
        """Test retrieval at corners."""
        hist_3d[0, 0, 0] = 1.0
        hist_3d[9, 9, 9] = 2.0
        assert lookup_py.get_hist_value(hist_3d, 0, 0, 0) == 1.0
        assert lookup_py.get_hist_value(hist_3d, 9, 9, 9) == 2.0

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython module not compiled")
    def test_cython_matches_python(self, hist_3d):
        """Cython implementation should match Python."""
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    py_val = lookup_py.get_hist_value(hist_3d, i, j, k)
                    cy_val = lookup_cy.get_hist_value(hist_3d, i, j, k)
                    assert py_val == cy_val


class TestLookupAllVoxels:
    """Tests for full histogram lookup."""

    @pytest.fixture
    def hist_3d(self):
        """Create a test 3D histogram."""
        np.random.seed(42)
        return np.random.rand(10, 10, 10).astype(np.float64)

    def test_output_length(self, hist_3d):
        """Output should have n^3 elements."""
        result = lookup_py.lookup_all_voxels(hist_3d, 10)
        assert len(result) == 1000

    def test_matches_flatten(self, hist_3d):
        """Result should match numpy flatten (C order)."""
        result = lookup_py.lookup_all_voxels(hist_3d, 10)
        expected = hist_3d.flatten()
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython module not compiled")
    def test_cython_matches_python(self, hist_3d):
        """Cython implementation should match Python."""
        py_result = lookup_py.lookup_all_voxels(hist_3d, 10)
        cy_result = np.asarray(lookup_cy.lookup_all_voxels(hist_3d, 10))
        np.testing.assert_array_almost_equal(py_result, cy_result)


class TestMultiShotLookup:
    """Tests for multi-shot histogram processing."""

    @pytest.fixture
    def histograms(self):
        """Create test 4D histogram array (shots, x, y, z)."""
        np.random.seed(42)
        return np.random.rand(5, 10, 10, 10).astype(np.float64)

    def test_output_shape(self, histograms):
        """Output should have shape (n_shots, axis_len^3)."""
        result = lookup_py.multi_shot_lookup(histograms, 5, 10)
        assert result.shape == (5, 1000)

    def test_matches_per_shot_flatten(self, histograms):
        """Each row should match corresponding histogram flattened."""
        result = lookup_py.multi_shot_lookup(histograms, 5, 10)
        for shot in range(5):
            expected = histograms[shot].flatten()
            np.testing.assert_array_almost_equal(result[shot], expected)

    @pytest.mark.skipif(not HAS_CYTHON, reason="Cython module not compiled")
    def test_cython_matches_python(self, histograms):
        """Cython implementation should match Python."""
        py_result = lookup_py.multi_shot_lookup(histograms, 5, 10)
        cy_out = np.empty((5, 1000), dtype=np.float64)
        lookup_cy.multi_shot_lookup(histograms, 5, 10, cy_out)
        np.testing.assert_array_almost_equal(py_result, cy_out)


class TestHighLevelAPI:
    """Tests for the high-level user-facing API."""

    def test_is_cython_available(self):
        """is_cython_available should return a boolean."""
        result = is_cython_available()
        assert isinstance(result, bool)

    def test_lookup_all_voxels_basic(self):
        """High-level lookup_all_voxels should work with just an array."""
        hist = np.random.rand(10, 10, 10).astype(np.float64)
        result = lookup_all_voxels(hist)
        assert result.shape == (1000,)
        assert isinstance(result, np.ndarray)

    def test_lookup_all_voxels_matches_flatten(self):
        """High-level API should produce same result as flatten."""
        hist = np.random.rand(15, 15, 15).astype(np.float64)
        result = lookup_all_voxels(hist)
        np.testing.assert_array_almost_equal(result, hist.flatten())

    def test_lookup_all_voxels_force_python(self):
        """Should be able to force Python implementation."""
        hist = np.random.rand(10, 10, 10).astype(np.float64)
        result = lookup_all_voxels(hist, use_cython=False)
        np.testing.assert_array_almost_equal(result, hist.flatten())

    def test_lookup_all_voxels_auto_converts_dtype(self):
        """Should auto-convert to float64 if needed."""
        hist = np.random.rand(10, 10, 10).astype(np.float32)
        result = lookup_all_voxels(hist)
        assert result.dtype == np.float64

    def test_lookup_all_voxels_rejects_non_cubic(self):
        """Should reject non-cubic histograms."""
        hist = np.random.rand(10, 10, 20).astype(np.float64)
        with pytest.raises(ValueError, match="cubic"):
            lookup_all_voxels(hist)

    def test_lookup_all_voxels_rejects_non_3d(self):
        """Should reject non-3D arrays."""
        hist = np.random.rand(10, 10).astype(np.float64)
        with pytest.raises(ValueError, match="3D"):
            lookup_all_voxels(hist)

    def test_multi_shot_lookup_basic(self):
        """High-level multi_shot_lookup should work with just an array."""
        histograms = np.random.rand(5, 10, 10, 10).astype(np.float64)
        result = multi_shot_lookup(histograms)
        assert result.shape == (5, 1000)
        assert isinstance(result, np.ndarray)

    def test_multi_shot_lookup_matches_per_shot(self):
        """Each row should match corresponding histogram flattened."""
        histograms = np.random.rand(3, 8, 8, 8).astype(np.float64)
        result = multi_shot_lookup(histograms)
        for i in range(3):
            np.testing.assert_array_almost_equal(result[i], histograms[i].flatten())

    def test_multi_shot_lookup_force_python(self):
        """Should be able to force Python implementation."""
        histograms = np.random.rand(3, 8, 8, 8).astype(np.float64)
        result = multi_shot_lookup(histograms, use_cython=False)
        for i in range(3):
            np.testing.assert_array_almost_equal(result[i], histograms[i].flatten())

    def test_get_value_basic(self):
        """High-level get_value should return correct value."""
        hist = np.zeros((10, 10, 10), dtype=np.float64)
        hist[3, 4, 5] = 42.0
        assert get_value(hist, 3, 4, 5) == 42.0

    def test_get_value_force_python(self):
        """Should be able to force Python implementation."""
        hist = np.zeros((10, 10, 10), dtype=np.float64)
        hist[3, 4, 5] = 42.0
        assert get_value(hist, 3, 4, 5, use_cython=False) == 42.0
