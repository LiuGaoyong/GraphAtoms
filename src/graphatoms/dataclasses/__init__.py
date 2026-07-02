"""The extended dataclasses by pydantic & numpydantic."""

from graphatoms.dataclasses._numpydantic import NDArray, numpy_validator
from graphatoms.dataclasses._pydantic2pyarrow import get_pyarrow_schema
from graphatoms.dataclasses._pydanticModel import OurBaseModel, OurFrozenModel

__all__ = [
    "NDArray",
    "numpy_validator",
    "get_pyarrow_schema",
    "OurFrozenModel",
    "OurBaseModel",
]
