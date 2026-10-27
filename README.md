# graphatoms

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/graphatoms.svg)](https://anaconda.org/conda-forge/graphatoms)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/graphatoms.svg)](https://anaconda.org/conda-forge/graphatoms)
[![PyPI version](https://img.shields.io/pypi/v/graphatoms)](https://pypi.org/project/graphatoms/)
[![PyPI Downloads](https://static.pepy.tech/badge/graphatoms)](https://pepy.tech/projects/graphatoms)

The chemical core classes for graph-theory analysis of atomistic systems.

`graphatoms` represents extended systems, clusters and gas molecules as
coloured bond graphs. Everything is a frozen, validated
[Pydantic](https://docs.pydantic.dev/) model, so structures, energies and
graph metadata can be exchanged, stored and compared without losing type
safety.

> **Scope (v3)**: the application layer — CLI entry points, Hydra configs,
> reaction network generation, KMC runners and parallel executors — now lives
> in [RxNetGen](https://github.com/LiuGaoyong/RxNetGen). This package provides
> the reusable chemical core: `arrayapi`, `dataclasses`, `geometry`, `system`
> and `utils`.

## Features

- **Validated graph models**: `SysGraph`, `System`, `Cluster` and `Gas` are
  immutable Pydantic models with `numpydantic`-checked arrays.
- **Multi-format interoperability**: build from and export to ASE `Atoms`,
  pymatgen `Structure`/`Molecule`, RDKit molecules, igraph graphs and plain
  dictionaries.
- **Chemical graph analysis**: bond perception, neighbour lists, hop
  distances, chordless cycles, connected components, Weisfeiler–Lehman
  hashes, canonical keys (BLISS) and SMARTS extraction.
- **Structure matching**: pattern-to-system matching
  (`BondGraph.get_match_mode`) and system-to-system equivalence via coloured
  graph isomorphism or **ITLEE** — Invariant and Tolerant Local Environment
  Equivalence (Williams, 2024).
- **Geometry tooling**: minimum image convention, rotation, sphere sampling
  and distance/bond/neighbour construction.
- **Thermochemistry**: gas-phase enthalpy, entropy and free energy from
  vibrational frequencies, plus minima/transition-state checks.
- **Persistent storage**: a uniform database interface with in-memory
  (`dict`), folder, HDF5 and SQLite backends.
- **Array API compatibility**: backend-agnostic helpers that work with
  NumPy, PyTorch, JAX, CuPy and other array-API-compliant libraries.

## Installation

```bash
pip install graphatoms
```

With conda:

```bash
conda install -c conda-forge graphatoms
```

Python 3.12 or newer is required.

## Quick start

### A periodic system and a local cluster

```python
import numpy as np
from ase.build import fcc111

from graphatoms.system import Cluster, System

atoms = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
system = System.from_ase(
    atoms,
    parse_bonds={"method": "raw"},
    parse_atoms_is_outer_or_not=True,  # detect surface atoms
)

print(system.formula, system.natoms, system.nbonds)
print(system.is_periodic, system.idx_outer)

# Grow a movable cluster around atom 0; atoms farther than `env_threshold`
# are fixed, and atoms closer than `max_moved_threshold` may move.
cluster = Cluster.from_select(
    system,
    core=np.array([0]),
    method="hop",
    max_moved_threshold=1,
    env_threshold=2,
)
print(cluster.ncore, cluster.nmoved, cluster.nfix)
```

### A gas molecule

```python
import numpy as np

from graphatoms.system import Gas

gas = Gas.from_name("CH4", sticking=1.0)
gas = gas.update_energetics(
    energy=-0.5,
    fmax=0.02,
    frequencies=np.array(
        [1300.0, 1300.0, 1300.0, 1500.0, 1500.0, 1500.0, 2900.0, 2900.0, 2900.0]
    ),
)

print(gas.formula, gas.natoms)
print(gas.get_enthalpy(temp=300.0))
print(gas.get_free_energy(temp=300.0))
```

### Equivalence checks

```python
# `other` is another SysGraph with the same number of atoms.
same_topology = system.is_equivalent(other, method="graph")
same_geometry = system.is_equivalent(
    other,
    method="itlee",
    tolerance=0.1,  # L2 tolerance in Å
    fields=("numbers", "is_fix"),
)
```

`method="graph"` compares the coloured bond graph (hash bucket first, then
igraph isomorphism). `method="itlee"` compares equal-sized coloured point
clouds under rotation, translation and atom permutation. For
pattern-to-system queries, use `BondGraph.get_match_mode`.

### Storage

```python
from pathlib import Path

from graphatoms.system import SysGraph
from graphatoms.system.database import get_db

db = get_db(format="hdf5", path=Path("systems.h5"))
db.add(SysGraph.from_ase(atoms, parse_bonds={"method": "raw"}))

for key in db:
    stored = SysGraph.from_ase(db[key], parse_bonds=None)
    print(key, stored.formula)
```

Supported formats: `dict` (default), `folder`/`dir`, `hdf5`/`h5`, `sqlite`.
Records are stored as ASE `Atoms`; rebuild a graph from a record with
`SysGraph.from_ase()`.

## Module structure

```
src/graphatoms/
├── arrayapi/          # array-API namespace resolution and typing
├── dataclasses/       # Pydantic/numpydantic base models, PyArrow schemas
├── geometry/          # bond_list, neighbor_list, distance_pairs,
│                      # mic, rotation, sample, itlee
├── system/            # core classes
│   ├── atoms.py       # Box, Energetics, Matter, Structure
│   ├── bonds.py       # BondGraph: graph metrics, hashes, matching
│   ├── graph.py       # SysGraph: structure + bonds + atom tags
│   ├── system.py      # System: full periodic/extended systems
│   ├── sysCluster.py  # Cluster: movable/fixed regions around a site
│   ├── sysGas.py      # Gas: ideal-gas thermochemistry
│   └── database/      # dict, folder, HDF5, SQLite backends
└── utils/             # bytestool, parser (Hydra), rdutils, subgraph
```

The public objects live in their submodules rather than the package root:

```python
from graphatoms.system import SysGraph, System, Cluster, Gas
from graphatoms.system.database import get_db
from graphatoms.geometry import bond_list, neighbor_list, distance_pairs
```

## Data model

`SysGraph` combines four mixins:

| Mixin       | Contents                                                        |
| ----------- | --------------------------------------------------------------- |
| `Structure` | `numbers`, `positions`, `cell`/`pbc`, plus `Energetics`         |
| `BondGraph` | bond `pair`, `shift`, `order`, `coordination`, derived `P`/`CN` |
| `AtomTag`   | boolean tags `is_adsorbate`, `is_outer`, `is_core`, `is_fix`    |
| `GasMixin`  | `sticking` coefficient and the derived `is_gas` flag            |

`System` is a `SysGraph` for the full system, `Cluster` is a selected,
possibly partially fixed region of it, and `Gas` adds a `sticking`
coefficient and ideal-gas thermodynamics. All of them accept
`from_ase()`, `to_ase()`, `from_dict()`/`to_dict()`, `from_igraph()`/
`to_igraph()` and, where meaningful, pymatgen/RDKit conversions.

## Requirements

Runtime dependencies:

- Python >= 3.12
- ase
- pymatgen > 2023.6
- rdkit >= 2025
- numpy >= 2, < 3
- scipy >= 1.10
- scikit-learn >= 1.5
- array-api-compat >= 1.10
- array-api-extra >= 0.10
- pyarrow
- pandas
- igraph >= 0.11
- h5py >= 3.16
- numpydantic
- pydantic >= 2.10
- hydra-core >= 1.1
- pyyaml >= 6.0
- tomli-w
- typing-extensions

## Development

The project is managed with [pixi](https://pixi.sh/) (or `uv`):

```bash
pixi install
pixi run test        # pytest -s -vv
pixi run benchmark  # benchmark suite
```

With `uv`:

```bash
uv sync
uv run pytest src/tests/ -v
uv run pytest src/tests-benchmark/ -v
```

Lint and format with Ruff:

```bash
ruff check .
ruff format .
```

## Array API compatibility

`graphatoms.arrayapi` resolves the active array namespace, and
`graphatoms.utils.subgraph` implements backend-agnostic graph helpers:

- `subgraph()` — extract an induced subgraph from edge indices
- `map_index()` — map indices to positions in a reference index array
- `index_to_mask()` — convert indices to a boolean mask
- `maybe_num_nodes()` — infer the node count from edge indices

These functions only use Array API standard operations, so the same code runs
on NumPy, PyTorch and JAX arrays.

## License

GPL-3.0-or-later

## Authors

- LiuGaoyong (liugaoyong_88@163.com)

## Links

- Homepage: https://github.com/LiuGaoyong/GraphAtoms
- Repository: https://github.com/LiuGaoyong/GraphAtoms
- Issues: https://github.com/LiuGaoyong/GraphAtoms/issues/
