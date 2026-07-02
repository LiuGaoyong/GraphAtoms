"""The array api wrapper of this packages."""

from .compat import get_namespace
from .typing import Array, ArrayNamespace, LinalgNamespace

__all__ = [
    "get_namespace",
    "ArrayNamespace",
    "LinalgNamespace",
    "Array",
]
