# ruff: noqa: D100 D102

import numpy as np
from numpy.typing import ArrayLike


class BaseMixin:
    """The base mixin class which provides some useful classmethods."""

    @staticmethod
    def get_mask_or_index(k: ArrayLike, n: int) -> np.ndarray:
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
        result = cls.get_mask_or_index(k=k, n=n)
        if result.dtype == bool:
            result = np.arange(n)[result]
        return result.astype(int, copy=False)

    @classmethod
    def get_mask(cls, k: ArrayLike, n: int) -> np.ndarray:
        result = cls.get_mask_or_index(k=k, n=n)
        if result.dtype != bool:
            result = np.isin(np.arange(n), result)
        return result.astype(bool, copy=False)


class XxxKeyMixin:
    @property
    def _DICT(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in dir(self) if k[0] != "_"}
