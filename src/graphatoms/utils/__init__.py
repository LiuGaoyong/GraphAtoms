"""The utils of this packages."""

from ._array_api_compat import get_namespace
from ._array_api_typing import Array, ArrayNamespace

__all__ = [
    "get_namespace",
    "ArrayNamespace",
    "Array",
]
