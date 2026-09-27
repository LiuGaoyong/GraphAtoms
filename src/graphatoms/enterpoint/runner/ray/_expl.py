import os

from graphatoms.enterpoint.runner.common._base import RunnerABC

os.environ["RAY_"] = "ray_task_id"
import ray

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.runner.common import _helper as funcs
from graphatoms.system import System


@ray.remote
def optimize_system(system: System, config: Config) -> System | str:
    result, _, _ = funcs.helper_optimization(
        config=config,
        graph=system,
    )
    raise NotImplementedError("optimize_system is not implemented yet.")


@ray.remote
def analysis_system(system: System, config: Config) -> System | str:
    raise NotImplementedError("run_system is not implemented yet.")


class ExplorationRayBase(RunnerABC):
    """The class for the ray simulation."""

    def submit(self) -> None:
        """Submit the task to the ray cluster."""
        raise NotImplementedError("submit is not implemented yet.")

    def collect(self) -> None:
        """Collect the results from the ray cluster."""
        raise NotImplementedError("collect is not implemented yet.")
