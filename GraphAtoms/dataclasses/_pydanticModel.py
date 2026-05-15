# ruff: noqa: D100 D102
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
from numbers import Real
from typing import override

import numpy as np
import pydantic
from numpy.typing import ArrayLike
from pyarrow import Schema

from ._pydantic2pyarrow import get_pyarrow_schema
from ._pydanticMixin import PydanticIoFactoryMixin as PydanticFactoryMixin

__all__ = ["OurBaseModel", "OurFrozenModel"]


class _IndexMaskMixin:
    """Mixin for index and mask operations on arrays."""

    @staticmethod
    def get_mask_or_index(k: ArrayLike, n: int) -> np.ndarray:
        """Convert input to index array.

        Args:
            k: Boolean mask or integer indices.
            n: Total length.

        Returns:
            Index array (integers).
        """
        if np.isscalar(k):
            if isinstance(k, bool):
                k = [k] * n
            elif isinstance(k, int):
                k = [k]
            else:
                raise TypeError(f"Unsupported type({type(k)}): {k}.")
        subset = np.asarray(k)
        if subset.dtype == bool:
            if subset.size != n:
                raise KeyError(
                    f"Except {n} boolean array, but {subset.size} got."
                )
        else:
            subset = np.unique(subset.astype(int))
            if not 0 <= max(subset) < n:
                raise KeyError(
                    f"Except 0-{n - 1} integer array, but  the "
                    f"max({max(subset)}),min({min(subset)}) got."
                )
        return subset.flatten()

    @classmethod
    def get_index(cls, k: ArrayLike, n: int) -> np.ndarray:
        """Get integer indices from mask or index array."""
        result = cls.get_mask_or_index(k=k, n=n)
        if result.dtype == bool:
            result = np.arange(n)[result]
        return result.astype(int, copy=False)

    @classmethod
    def get_mask(cls, k: ArrayLike, n: int) -> np.ndarray:
        """Get boolean mask from mask or index array."""
        result = cls.get_mask_or_index(k=k, n=n)
        if result.dtype != bool:
            result = np.isin(np.arange(n), result)
        return result.astype(bool, copy=False)


class OurBaseModel(PydanticFactoryMixin, _IndexMaskMixin):
    """Extended base class for Pydantic models with I/O support.

    Supports: json, npz, pickle, yaml/yml, toml, bytes (with compression).
    Only numpy ndarray and numpy-compatible scalars are supported.
    """

    @classmethod
    def get_pyarrow_schema(cls) -> Schema:
        """Get the pyarrow schema of this class."""
        return get_pyarrow_schema(cls)

    def __repr__(self) -> str:
        return super().__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        for k in self.__pydantic_fields__:
            v0 = getattr(self, k, None)
            v1 = getattr(other, k, None)
            if (v0 is None) != (v1 is None):
                return False
            elif v0 is None and v1 is None:
                continue
            elif np.isscalar(v0) and np.isscalar(v1):
                if (
                    isinstance(v0, Real)
                    and isinstance(v1, Real)
                    and abs(v0 - v1) > 1e-9  # type: ignore
                ):
                    return False
                elif v0 != v1:
                    return False
                else:
                    continue

            type0, type1 = type(v0), type(v1)
            if type0 is not type1:
                return False
            elif issubclass(type0, pydantic.BaseModel | Sequence | Mapping):
                if not v0.__eq__(v1):
                    return False
            elif type0 is np.ndarray:
                assert isinstance(v0, np.ndarray)
                assert isinstance(v1, np.ndarray)
                if v0.shape != v1.shape:
                    return False
                if np.all(v0 == v1):
                    return True
                elif not np.allclose(v0, v1):
                    return False
            else:
                raise NotImplementedError(
                    "Unsupported type for __eq__: ", type0
                )
        return True

    @override
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self._string()})"

    @abstractmethod
    def _string(self) -> str:
        """The string representation of this class.

        The expression like `str(obj)` will output
        `f"{self.__class__.__name__}({this_value})"`.
        """


class OurFrozenModel(OurBaseModel):
    """Immutable version of OurBaseModel.

    All instances are frozen (hashable, immutable).
    """

    model_config = pydantic.ConfigDict(frozen=True)

    @override
    def __hash__(self) -> int:
        """Hash based on content."""
        return hash(self.hash)

    @cached_property
    @abstractmethod
    def hash(self) -> str:
        """Get the HASH string of this object."""
