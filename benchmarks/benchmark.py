#!/usr/bin/env python3
"""
Benchmark script comparing Python vs Cython histogram lookup performance.

This script measures the speedup achieved by the Cython implementation
for various histogram sizes and workloads.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np

from histlookup import lookup_cy, lookup_py


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    axis_len: int
    n_voxels: int
    n_shots: int
    python_time: float
    cython_time: float

    @property
    def speedup(self) -> float:
        return self.python_time / self.cython_time


def generate_synthetic_histogram(
    axis_len: int, n_shots: int = 1, seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic 3D histogram data.

    Parameters
    ----------
    axis_len : int
        Length of each axis (creates axis_len^3 voxels)
    n_shots : int
        Number of independent histograms (shots)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of shape (n_shots, axis_len, axis_len, axis_len) or
        (axis_len, axis_len, axis_len) if n_shots == 1
    """
    rng = np.random.default_rng(seed)
    if n_shots == 1:
        return rng.random((axis_len, axis_len, axis_len), dtype=np.float64)
    return rng.random((n_shots, axis_len, axis_len, axis_len), dtype=np.float64)


def benchmark_lookup_all_voxels(
    axis_len: int, n_iterations: int = 5
) -> BenchmarkResult:
    """
    Benchmark lookup_all_voxels for a single histogram.

    Parameters
    ----------
    axis_len : int
        Length of each histogram axis
    n_iterations : int
        Number of iterations for timing

    Returns
    -------
    BenchmarkResult
        Benchmark results with timing information
    """
    hist = generate_synthetic_histogram(axis_len)
    n_voxels = axis_len**3

    # Warmup
    _ = lookup_py.lookup_all_voxels(hist, axis_len)
    _ = np.asarray(lookup_cy.lookup_all_voxels(hist, axis_len))

    # Benchmark Python
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = lookup_py.lookup_all_voxels(hist, axis_len)
    python_time = (time.perf_counter() - start) / n_iterations

    # Benchmark Cython
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = np.asarray(lookup_cy.lookup_all_voxels(hist, axis_len))
    cython_time = (time.perf_counter() - start) / n_iterations

    return BenchmarkResult(
        name="lookup_all_voxels",
        axis_len=axis_len,
        n_voxels=n_voxels,
        n_shots=1,
        python_time=python_time,
        cython_time=cython_time,
    )


def benchmark_multi_shot(
    axis_len: int, n_shots: int, n_iterations: int = 3
) -> BenchmarkResult:
    """
    Benchmark multi_shot_lookup for multiple histograms.

    Parameters
    ----------
    axis_len : int
        Length of each histogram axis
    n_shots : int
        Number of histogram shots to process
    n_iterations : int
        Number of iterations for timing

    Returns
    -------
    BenchmarkResult
        Benchmark results with timing information
    """
    histograms = generate_synthetic_histogram(axis_len, n_shots)
    n_voxels = axis_len**3

    # Pre-allocate output for Cython
    cy_out = np.empty((n_shots, n_voxels), dtype=np.float64)

    # Warmup
    _ = lookup_py.multi_shot_lookup(histograms, n_shots, axis_len)
    lookup_cy.multi_shot_lookup(histograms, n_shots, axis_len, cy_out)

    # Benchmark Python
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = lookup_py.multi_shot_lookup(histograms, n_shots, axis_len)
    python_time = (time.perf_counter() - start) / n_iterations

    # Benchmark Cython
    start = time.perf_counter()
    for _ in range(n_iterations):
        lookup_cy.multi_shot_lookup(histograms, n_shots, axis_len, cy_out)
    cython_time = (time.perf_counter() - start) / n_iterations

    return BenchmarkResult(
        name="multi_shot_lookup",
        axis_len=axis_len,
        n_voxels=n_voxels,
        n_shots=n_shots,
        python_time=python_time,
        cython_time=cython_time,
    )


def format_time(seconds: float) -> str:
    """Format time in human-readable units."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: Python vs Cython 3D Histogram Lookup")
    print("=" * 80)

    print(
        f"\n{'Function':<22} {'Axis':<6} {'Voxels':<10} {'Shots':<6} "
        f"{'Python':<12} {'Cython':<12} {'Speedup':<10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.name:<22} {r.axis_len:<6} {r.n_voxels:<10} {r.n_shots:<6} "
            f"{format_time(r.python_time):<12} {format_time(r.cython_time):<12} "
            f"{r.speedup:>6.1f}x"
        )

    print("-" * 80)
    avg_speedup = np.mean([r.speedup for r in results])
    max_speedup = max(r.speedup for r in results)
    print(f"\nAverage speedup: {avg_speedup:.1f}x")
    print(f"Maximum speedup: {max_speedup:.1f}x")
    print()


def verify_correctness() -> bool:
    """Verify that Cython and Python implementations produce identical results."""
    print("Verifying correctness...")

    # Test lookup_all_voxels
    for axis_len in [5, 10, 20]:
        hist = generate_synthetic_histogram(axis_len)
        py_result = lookup_py.lookup_all_voxels(hist, axis_len)
        cy_result = np.asarray(lookup_cy.lookup_all_voxels(hist, axis_len))
        if not np.allclose(py_result, cy_result):
            print(f"FAILED: lookup_all_voxels (axis_len={axis_len})")
            return False

    # Test multi_shot_lookup
    for axis_len in [5, 10]:
        for n_shots in [3, 10]:
            histograms = generate_synthetic_histogram(axis_len, n_shots)
            py_result = lookup_py.multi_shot_lookup(histograms, n_shots, axis_len)
            cy_out = np.empty((n_shots, axis_len**3), dtype=np.float64)
            lookup_cy.multi_shot_lookup(histograms, n_shots, axis_len, cy_out)
            if not np.allclose(py_result, cy_out):
                print(
                    f"FAILED: multi_shot_lookup (axis_len={axis_len}, "
                    f"n_shots={n_shots})"
                )
                return False

    print("All correctness checks passed.\n")
    return True


def run_benchmarks(quick: bool = False) -> list[BenchmarkResult]:
    """
    Run the full benchmark suite.

    Parameters
    ----------
    quick : bool
        If True, run a smaller set of benchmarks for quick testing

    Returns
    -------
    list[BenchmarkResult]
        List of benchmark results
    """
    results = []

    if quick:
        axis_lengths = [10, 20, 30]
        shot_configs = [(10, 50), (20, 20)]
    else:
        axis_lengths = [10, 20, 30, 40, 50]
        shot_configs = [(10, 100), (20, 50), (30, 20), (40, 10)]

    print("Running single-histogram benchmarks...")
    for axis_len in axis_lengths:
        result = benchmark_lookup_all_voxels(axis_len)
        results.append(result)
        print(f"  axis_len={axis_len}: {result.speedup:.1f}x speedup")

    print("\nRunning multi-shot benchmarks...")
    for axis_len, n_shots in shot_configs:
        result = benchmark_multi_shot(axis_len, n_shots)
        results.append(result)
        print(
            f"  axis_len={axis_len}, n_shots={n_shots}: {result.speedup:.1f}x speedup"
        )

    return results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python vs Cython histogram lookup"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmarks (smaller sizes)"
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip correctness verification"
    )
    args = parser.parse_args()

    if lookup_cy is None:
        print("ERROR: Cython module not compiled. Run 'pip install -e .' first.")
        return 1

    if not args.skip_verify:
        if not verify_correctness():
            return 1

    results = run_benchmarks(quick=args.quick)
    print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
