# fast-histogram-lookup

[![CI](https://github.com/JPBureik/fast-histogram-lookup/actions/workflows/ci.yml/badge.svg)](https://github.com/JPBureik/fast-histogram-lookup/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JPBureik/fast-histogram-lookup/graph/badge.svg)](https://codecov.io/gh/JPBureik/fast-histogram-lookup)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)

Cython-accelerated 3D histogram bin lookups with **~40x speedup** over pure Python.

## Overview

This package provides optimized functions for accessing elements in large 3D
histograms. The hot loop is implemented in Cython with typed memoryviews,
bypassing Python's interpreter overhead for significant speedup in tight loops.

**Key optimizations:**
- Typed memoryviews (`double[:, :, :]`) for direct C-level array access
- Bounds checking disabled via compiler directive
- Negative index wrapping disabled
- Pure C integer arithmetic for index conversion

## Benchmarks

Measured on synthetic data matching realistic workloads (multiple histogram
"shots" with varying resolutions):

| Histogram Size | Voxels | Shots | Total Lookups | Python | Cython | Speedup |
|----------------|--------|-------|---------------|--------|--------|---------|
| 10³ | 1,000 | 100 | 100K | 35 ms | 1.0 ms | **34x** |
| 20³ | 8,000 | 50 | 400K | 155 ms | 4.1 ms | **37x** |
| 30³ | 27,000 | 20 | 540K | 214 ms | 5.6 ms | **38x** |
| 40³ | 64,000 | 10 | 640K | 258 ms | 6.6 ms | **39x** |
| 55³ | 166,375 | 100 | 16.6M | 6.0 s | 160 ms | **38x** |

**Average speedup: ~38x**

Run benchmarks yourself:
```bash
python benchmarks/benchmark.py
python benchmarks/benchmark.py --plot  # Generate bar chart
```

## Profiling

This module was extracted from a larger scientific analysis pipeline after
profiling identified the bottleneck. The original workflow was:

1. **Load data** — read experimental shots from HDF5
2. **Bin into 3D histogram** — create momentum-space voxel grid
3. **Look up all voxels** — extract values for each voxel across all shots ← *bottleneck*
4. **Calculate correlations** — compute high-order correlation functions

Profiling the full pipeline showed step 3 dominated runtime: millions of
Python-level array accesses with index arithmetic. This module isolates that
hot loop with a Cython implementation.

To profile the isolated baseline:

```bash
python profiling/profile_baseline.py
snakeviz profiling/baseline.prof  # pip install snakeviz
```

Within the isolated module, `flat_to_3d` and histogram indexing dominate —
exactly the operations that benefit from typed memoryviews and compiled
index arithmetic.

## Installation

```bash
git clone https://github.com/JPBureik/fast-histogram-lookup.git
cd fast-histogram-lookup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```python
import numpy as np
from histlookup import lookup_cy, lookup_py

# Create a 3D histogram (e.g., 50x50x50 bins)
hist = np.random.rand(50, 50, 50).astype(np.float64)

# Cython version (fast)
result_cy = np.asarray(lookup_cy.lookup_all_voxels(hist, 50))

# Python version (baseline)
result_py = lookup_py.lookup_all_voxels(hist, 50)

# Results are identical
assert np.allclose(result_cy, result_py)
```

## Development

```bash
# Run tests
pytest

# Run linting
ruff check src tests
ruff format --check src tests

# Run type checking
mypy src/histlookup

# Install pre-commit hooks
pre-commit install
```

## License

MIT License - see [LICENSE](LICENSE) for details.
