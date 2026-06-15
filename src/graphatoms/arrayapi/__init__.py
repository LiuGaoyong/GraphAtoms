"""The array api wrapper of this packages."""

from ._array_api_compat import get_namespace
from ._array_api_typing import Array, ArrayNamespace, LinalgNamespace

__all__ = [
    "get_namespace",
    "ArrayNamespace",
    "LinalgNamespace",
    "Array",
]
