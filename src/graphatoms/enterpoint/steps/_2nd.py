import os
from typing import override

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


from graphatoms.system import Cluster, Gas  # type: ignore

from ._0abc import BaseABC


class SecondStepSurface(BaseABC):
    """The class for exploring the surface process."""

    @override
    def run(self, cluster: Cluster) -> None:
        pass


class SecondStepBulk(BaseABC):
    """The class for exploring the bulk process."""

    @override
    def run(self, cluster: Cluster) -> None:
        pass


class SecondStepAdsorption(BaseABC):
    """The class for exploring the adsorption process."""

    @override
    def run(self, cluster: Cluster, gas: Gas) -> None:
        pass
