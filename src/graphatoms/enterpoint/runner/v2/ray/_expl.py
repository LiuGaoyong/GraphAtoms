import ray
from ase import Atoms

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.enterpoint.runner.common._base import RunnerABC
from graphatoms.system import System

from ._helper import (
    helper_adsorption,
    helper_dimer,
    helper_optimization,
)


class ExplorationRayBase(RunnerABC):
    """The exploration class based on Ray cluster."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.config_ray = ray.put(config)
        self.sync_network_to_ray()

    def sync_network_to_ray(self) -> None:
        self.network.persistence()
        # self.network_ray = RayReactionNetwork.remote(
        #     format=self.config.event.db_format,
        #     config=self.config.event,
        #     path=self.network.path,
        #     restart=True,
        # )

    def sync_network_to_local(self) -> None:
        # ray.get(self.network_ray.persistence.remote())
        self.network = ReactionNetwork.from_config(self.config, restart=True)

    def explore(self, lst: list[System | Atoms] | None = None) -> None:
        # ---------------------------------------------
        #       analyze the system list
        # ---------------------------------------------
        if lst is None:
            lst = [self.get_system_for(None)]
        helper_optimization.remote, helper_adsorption, helper_dimer  # type: ignore

    def submit(self) -> None:
        """Submit the task to the ray cluster."""
        raise NotImplementedError("submit is not implemented yet.")

    def collect(self) -> None:
        """Collect the results from the ray cluster."""
        raise NotImplementedError("collect is not implemented yet.")
