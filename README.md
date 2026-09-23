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
- **Array API Compatibility**: Full support for array API standard for cross-framework compatibility (NumPy, PyTorch, JAX, CuPy, etc.)
- **Subgraph Operations**: Backend-agnostic subgraph extraction with relabeling support using array-api-compat and array-api-extra
- **CLI Entry Points**: Three console scripts for configuration, execution, and inspection:
  - `graphatoms-config`: Resolve and print the Hydra/OmegaConf run configuration
  - `graphatoms-run`: Launch a run
    - `run_type=otfkmc` for on-the-fly kinetic Monte Carlo simulation
    - `run_type=rxngen` for reaction network generation
  - `graphatoms-network`: Inspect and visualize a stored reaction network
- **Hydra-driven Configuration**: Composable, override-friendly config via Hydra/OmegaConf with grouped groups (`atoms`, `bonds`, `calculator`)
- **Pluggable Parallel Backends**: Switch executors at the config level — `serial`, `multiprocessing`, `ray`, `dask`, `executorlib` — for distributed/on-the-fly KMC workflows

## Module Structure

```
src/graphatoms/
├── arrayapi/        # Array API compatibility layer
├── dataclasses/     # Pydantic-based data models
├── enterpoint/      # Entry points: CLI, config, runners, network, parallel
│   ├── config/      # Hydra/OmegaConf configuration (atoms, bonds, calculator)
│   ├── network/     # Reaction network: scheduler, recorder, metadata
│   ├── parallel/    # Pluggable executors (serial, multiprocessing, ray, dask, executorlib)
│   ├── runner/      # Runners (otfkmc, rxngen) and helpers
│   ├── steps/       # Step primitives for runners
│   └── view.py      # CLI viewer for reactions
├── geometry/        # Geometric operations
├── reaction/        # Reaction classes and KMC events
│   ├── _event.py    # Event base and event info
│   ├── reaction.py  # Reaction class
│   └── xxsorption.py # Adsorption/Desorption events
├── system/          # Core system classes
│   ├── atoms.py     # Atomic structure handling
│   ├── bonds.py     # Bond list operations
│   ├── graph.py     # Graph-based system representation
│   ├── system.py    # System abstract base
│   ├── sysCluster.py # Cluster system
│   ├── sysGas.py    # Gas molecule system
│   └── database/    # Database storage backends (HDF5, SQLite, folder)
└── utils/           # Utility functions
    ├── adsorption.py # Adsorption site helper
    ├── asetools.py  # ASE-related tools
    ├── bytestool.py # Byte-level helpers
    ├── logger.py    # Logging setup
    ├── parser.py    # Hydra argument parsing
    ├── rdutils.py   # RDKit utilities
    └── subgraph.py  # Array API compatible subgraph operations
```

## Requirements

- Python >= 3.12
- ase
- pymatgen > 2023.6
- rdkit >= 2025
- scikit-learn >= 1.5
- array-api-compat >= 1.15.0
- array-api-extra >= 0.11.0
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
- pandas >= 2
- scipy >= 1.10
- typer
- executorlib

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

# Run all tests

pytest src/tests/ -v

# Run benchmark tests

pytest src/tests-benchmark/ -v

## Array API Compatibility

The library leverages `array-api-compat` and `array-api-extra` for backend-agnostic array operations. Key utilities include:

- `subgraph()`: Extracts induced subgraphs from edge indices
- `map_index()`: Maps indices across arrays
- `index_to_mask()`: Converts index arrays to boolean masks
- `maybe_num_nodes()`: Determines the number of nodes from edge indices

These functions work seamlessly with NumPy, PyTorch, JAX, and other array API compliant libraries.

## License

GPL-3.0-or-later

## Authors

- LiuGaoyong (liugaoyong_88@163.com)

## Links

- Homepage: https://github.com/LiuGaoyong/GraphAtoms
- Repository: https://github.com/LiuGaoyong/GraphAtoms
- Issues: https://github.com/LiuGaoyong/GraphAtoms/issues/
