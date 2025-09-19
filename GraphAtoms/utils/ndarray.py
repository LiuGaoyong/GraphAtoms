from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import pydantic
from numpydantic import NDArray as _NDArray

__all__ = ["NDArray", "numpy_validator"]
_ENCODING = "latin1"

NDArray = Annotated[
    _NDArray,
    pydantic.PlainSerializer(
        lambda x: x.tobytes().decode(_ENCODING),
        return_type=str,
    ),
    pydantic.WithJsonSchema(
        {"type": "string"},
        mode="serialization",
    ),
]
if TYPE_CHECKING:
    NDArray = np.typing.NDArray


@pydantic.validate_call
def __convert2numpy(
    x: Any,
    dtype: Any = "float",
    shape: Sequence[int] = (-1,),
) -> _NDArray:
    """Convert a string or bytes or array-like object to a numpy array.

    Raises:
        KeyError: If the input value is a scalar number.
    """
    if isinstance(x, bytes | str):
        x = x.encode(_ENCODING) if isinstance(x, str) else x
        return np.frombuffer(x, np.dtype(dtype)).reshape(shape)
    elif np.isscalar(x):
        raise KeyError(
            f"Invalid input value: {x}({type(x)})."
            " The scalar value is not supported."
        )
    else:
        return np.asarray(x, dtype=dtype).reshape(shape)


def numpy_validator(
    dtype: Any = "float",
    shape: Sequence[int] = (-1,),
) -> pydantic.BeforeValidator:
    """Create pydantic validator in `before` mode for numpy array."""
    return pydantic.BeforeValidator(
        partial(
            __convert2numpy,
            shape=shape,
            dtype=dtype,
        )
    )
