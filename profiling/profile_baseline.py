#!/usr/bin/env python3
"""
Profiling script demonstrating how the bottleneck was identified.

This script profiles the pure Python implementation to show that the
histogram lookup loop is the dominant cost, justifying the Cython optimization.

Usage:
    python profiling/profile_baseline.py

    # Visualize with snakeviz:
    pip install snakeviz
    snakeviz profiling/baseline.prof
"""

from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

import numpy as np

from histlookup import lookup_py


def workload() -> None:
    """Simulate a realistic workload: multiple histogram shots."""
    axis_len = 30
    n_shots = 50
    histograms = np.random.rand(n_shots, axis_len, axis_len, axis_len).astype(
        np.float64
    )
    _ = lookup_py.multi_shot_lookup(histograms, n_shots, axis_len)


def main() -> None:
    print("Profiling pure Python baseline...")
    print(f"Workload: 50 shots × 30³ voxels = {50 * 30**3:,} lookups\n")

    # Profile the workload
    profiler = cProfile.Profile()
    profiler.enable()
    workload()
    profiler.disable()

    # Save profile for visualization (e.g., snakeviz)
    output_path = Path(__file__).parent / "baseline.prof"
    profiler.dump_stats(output_path)
    print(f"Profile saved to: {output_path}")
    print("Visualize with: snakeviz profiling/baseline.prof\n")

    # Print summary statistics
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    print("=" * 60)
    print("TOP 10 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 60)
    stats.print_stats(10)

    print("\nKey insight: flat_to_3d and histogram indexing dominate.")
    print("This is the hot loop that benefits from Cython optimization.")


if __name__ == "__main__":
    main()
