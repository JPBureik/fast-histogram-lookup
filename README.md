# fast-histogram-lookup

[![CI](https://github.com/JPBureik/fast-histogram-lookup/actions/workflows/ci.yml/badge.svg)](https://github.com/JPBureik/fast-histogram-lookup/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JPBureik/fast-histogram-lookup/graph/badge.svg)](https://codecov.io/gh/JPBureik/fast-histogram-lookup)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)

Cython-accelerated 3D histogram bin lookups with ~80x speedup over pure Python.

## Overview

This package provides optimized functions for accessing elements in large 3D
histograms. The hot loop is implemented in Cython with typed memoryviews,
bypassing Python's interpreter overhead for significant speedup in tight loops.

**Key optimizations:**
- Typed memoryviews (`double[:, :, :]`) for direct C-level array access
- Bounds checking disabled via compiler directive
- Negative index wrapping disabled
- Pure C integer arithmetic for index conversion

## Installation

```bash
pip install fast-histogram-lookup
```

Or from source:

```bash
git clone https://github.com/JPBureik/fast-histogram-lookup.git
cd fast-histogram-lookup
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

## Benchmarks

TODO: Add benchmark results

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

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
