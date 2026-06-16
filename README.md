# graphatoms

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/graphatoms.svg)](https://anaconda.org/conda-forge/graphatoms)  [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/graphatoms.svg)](https://anaconda.org/conda-forge/graphatoms)    [![Pypi version](https://img.shields.io/pypi/v/graphatoms)](https://pypi.org/project/graphatoms/) [![PyPI Downloads](https://static.pepy.tech/badge/graphatoms)](https://pepy.tech/projects/graphatoms)


The Chemical Core Class for Graph Theory Analysis.

## Overview

The `graphatoms` is a Python library designed for chemical graph theory analysis. It provides core classes for representing chemical systems and reactions with graph-based data structures.

## Features

- **Graph-based Chemical System Representation**: Represent chemical systems, clusters, and gas molecules using graph theory
- **Reaction Modeling**: Support for reaction classes, KMC (Kinetic Monte Carlo) events, and MC (Monte Carlo) moves
- **Geometry Operations**: Bond lists, distance calculations, neighbor lists, rotations, MIC (Minimum Image Convention), and sampling
- **Data Storage**: Support for HDF5 and SQLite databases for efficient data persistence
- **Dataclasses**: Pydantic-based data models for type-safe data handling
- **Array API Compatibility**: Support for array API standard for cross-framework compatibility

## Module Structure

```
src/graphatoms/
├── arrayapi/        # Array API compatibility layer
├── dataclasses/     # Pydantic-based data models
├── geometry/        # Geometric operations
├── reaction/        # Reaction classes and KMC events
│   ├── base/        # Abstract base classes
│   ├── event/       # KMC events (adsorption, desorption, reaction)
│   ├── mcmove/      # Monte Carlo moves
│   ├── mdwarpper/   # MD wrapper
│   └── network/     # Reaction network
├── system/          # Core system classes
│   ├── atoms/       # Atomic structure handling
│   ├── database/    # Database storage backends
│   └── graph/       # Graph-based system representation
└── utils/           # Utility functions
```

## Requirements

- Python >= 3.12
- ase
- pymatgen > 2023.6
- rdkit >= 2025
- scikit-learn >= 1.5
- array-api-compat >= 1.10
- array-api-extra >= 0.10
- pyarrow
- igraph >= 0.11
- h5py >= 3.16
- hydra-core
- numpy >= 2.0.0
- numpydantic
- ovld
- pydantic >= 2.10
- python-snappy >= 0.7.3
- loguru

## Installation

```bash
pip install graphatoms
```

Or with conda:

```bash
conda install -c conda-forge graphatoms
```

## Development

For development setup with pixi:

```bash
pixi install
pixi run test
```

## Running Tests

```bash
# Run all tests
pytest src/tests/ -v

# Run benchmark tests
pytest src/tests-benchmark/ -v
```

## License

GPL-3.0-or-later

## Authors

- LiuGaoyong (liugaoyong_88@163.com)

## Links

- Homepage: https://github.com/LiuGaoyong/GraphAtoms
- Repository: https://github.com/LiuGaoyong/GraphAtoms
- Issues: https://github.com/LiuGaoyong/GraphAtoms/issues/
