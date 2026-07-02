"""The array api wrapper of this packages."""

from graphatoms.arrayapi.compat import get_namespace
from graphatoms.arrayapi.typing import Array, ArrayNamespace, LinalgNamespace

__all__ = [
    "get_namespace",
    "ArrayNamespace",
    "LinalgNamespace",
    "Array",
]
