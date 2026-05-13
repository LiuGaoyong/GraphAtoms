from collections.abc import Sequence
from typing import override

from ._ase import ASEConverter
from ._graph import GraphConverter
from ._pygdata import PyTorchGeometricConverter

__all__ = ["Convert"]


class Convert(
    ASEConverter,
    GraphConverter,
    PyTorchGeometricConverter,
):
    """Unified converter for ASE, PyG, igraph, NetworkX, and Rustworkx formats."""
    @override
    @classmethod
    def SUPPORTED_CONVERT_FORMATS(cls) -> Sequence[str]:
        result = ("ase", "pygdata", "igraph", "networkx", "rustworkx")
        return tuple(super().SUPPORTED_CONVERT_FORMATS()) + result

    # @override
    # @classmethod
    # def SUPPORTED_IO_FORMATS(cls) -> Sequence[str]:
    #     result: tuple[str, ...] = ("ase", "pymatgen")
    #     return tuple(super().SUPPORTED_IO_FORMATS()) + result
