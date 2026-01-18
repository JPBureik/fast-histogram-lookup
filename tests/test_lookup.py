"""Tests for histogram lookup functions."""

import numpy as np
import pytest

from histlookup import lookup_py

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
