# graphatoms

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/graphatoms.svg)](https://anaconda.org/conda-forge/graphatoms)  [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/graphatoms.svg)](https://anaconda.org/conda-forge/graphatoms)    [![Pypi version](https://img.shields.io/pypi/v/graphatoms)](https://pypi.org/project/graphatoms/) [![PyPI Downloads](https://static.pepy.tech/badge/graphatoms)](https://pepy.tech/projects/graphatoms)


The Chemical Core Class for Graph Theory Analysis & Graph Neural Network.

## Overview

The `graphatoms` is a Python library designed for chemical graph theory analysis and graph neural network applications. It provides core classes for representing chemical systems and reactions with graph-based data structures.

## Features

- **Graph-based Chemical System Representation**: Represent chemical systems, clusters, and gas molecules using graph theory
- **Reaction Modeling**: Support for reaction classes, KMC (Kinetic Monte Carlo) events, and MC (Monte Carlo) moves
- **Geometry Operations**: Bond lists, distance calculations, neighbor lists, rotations, and sampling
- **Data Storage**: Support for HDF5 and SQLite databases for efficient data persistence
- **Dataclasses**: Pydantic-based data models for type-safe data handling

## Module Structure

```
src/graphatoms/
├── dataclasses/     # Pydantic-based data models
├── geometry/        # Geometric operations
├── reaction/        # Reaction classes and KMC events
├── system/          # Core system classes (System, Cluster, Gas)
└── utils/           # Utility functions
```

## Requirements

- Python >= 3.10
- ase
- pymatgen > 2023.6
- rdkit >= 2025
- scikit-learn >= 1.5
- pyarrow
- igraph >= 0.11
- h5py >= 3.16
- hydra-core
- numpydantic
- pydantic >= 2.11

## Installation

```bash
pip install graphatoms
```

## Development

For development setup with pixi:

```bash
pixi install
pixi run test
```

## License

GPL-3.0-or-later

## Authors

- LiuGaoyong (liugaoyong_88@163.com)

## Links

- Homepage: https://github.com/LiuGaoyong/GraphAtoms
- Repository: https://github.com/LiuGaoyong/GraphAtoms
- Issues: https://github.com/LiuGaoyong/GraphAtoms/issues/
