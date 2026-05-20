"""The extended dataclasses by pydantic & numpydantic."""

from ._numpydantic import NDArray, numpy_validator
from ._pydantic2pyarrow import get_pyarrow_schema
from ._pydanticModel import OurBaseModel, OurFrozenModel

__all__ = [
    "NDArray",
    "numpy_validator",
    "get_pyarrow_schema",
    "OurFrozenModel",
    "OurBaseModel",
]
