# """The steps for the on-the-fly KMC simulation."""

# from time import perf_counter
# from typing import override

# import numpy as np
# import pandas as pd
# from ase import Atoms
# from ase.io import read
# from ase.io.trajectory import TrajectoryWriter
# from pydantic import BaseModel

# from graphatoms.enterpoint.config import Config
# from graphatoms.enterpoint.runner._base import ExplorationABC
# from graphatoms.enterpoint.runner._helper import helper_apply, helper_match
# from graphatoms.system import System

# __all__ = ["ReactionNetworkGenerator"]


# class OTFKMCInfo(BaseModel):
#     time: float = 0.0
#     energy: float = 0
#     energy_real: float = np.nan
#     select_rxn_key: str = "Initial"
#     cost_exploration: float = 0
#     cost_matching: float = 0
#     cost_other: float = 0


# class ReactionNetworkGenerator(ExplorationABC):
#     """The class for the on-the-fly KMC simulation."""

#     @override
#     def __init__(self, *, config: Config) -> None:
#         super().__init__(config=config)
