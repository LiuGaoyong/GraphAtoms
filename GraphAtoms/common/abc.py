# ruff: noqa: D100 D102
import sys
from collections.abc import Sequence
from pickle import dumps, loads
from typing import Annotated, override

import numpy as np
import pydantic
import tomli_w as toml_w
import yaml
from joblib import dump, load
from numpy.typing import ArrayLike
from typing_extensions import Self

from GraphAtoms.common.string import (
    SUPPORTED_COMPRESS_FORMATS,
    compress,
    decompress,
)

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib
__all__ = [
    "BaseModel",
    "NpzPklBaseModel",
]


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


class BaseModel(pydantic.BaseModel):
    """A base class for creating Pydantic models.

    This class support many file formats:
        json        by `pydantic`
    """

    model_config = pydantic.ConfigDict(frozen=True)

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._string()})"

    def _string(self) -> str:
        """The string representation of this class.

        The expression like `str(obj)` will output
        `f"{self.__class__.__name__}({this_value})"`.
        """
        return "object"

    @staticmethod
    @pydantic.validate_call
    def __get_format(fname: pydantic.FilePath | pydantic.NewPath) -> str:
        format: str = fname.name.split(".")[-1].lower()
        if format == "yml":
            format = "yaml"
        elif format == "pkl":
            format = "pickle"
        return format

    def write(self, fname: pydantic.FilePath, **kwargs) -> pydantic.FilePath:
        f = getattr(self, f"write_{self.__get_format(fname)}")
        return f(fname, **kwargs)

    @pydantic.validate_call
    def as_bytes(
        self,
        compressformat: SUPPORTED_COMPRESS_FORMATS = "snappy",
        compresslevel: Annotated[int, pydantic.Field(ge=0, le=9)] = 0,
    ) -> bytes:
        """Return the json bytes of this object."""
        return compress(
            dumps(self.model_dump(exclude_none=True)),
            format=compressformat,  # type: ignore
            compresslevel=compresslevel,
        )

    @classmethod
    @pydantic.validate_call
    def from_bytes(
        cls,
        data: bytes,
        compressformat: SUPPORTED_COMPRESS_FORMATS = "snappy",
    ) -> Self:
        return cls.model_validate(
            loads(
                decompress(
                    data,
                    format=compressformat,  # type: ignore
                )
            )
        )

    @pydantic.validate_call
    def write_json(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        indent: int | None = 4,
        **kwargs,
    ) -> pydantic.FilePath:
        filename.write_text(
            self.model_dump_json(
                indent=indent,
                exclude_none=True,
                **kwargs,
            ),
            encoding="utf-8",
        )
        return filename

    @classmethod
    @pydantic.validate_call
    def read_json(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        return cls.model_validate_json(filename.read_bytes(), **kwargs)

    @classmethod
    def read(cls, fname: pydantic.FilePath, **kwargs) -> Self:
        f = getattr(cls, f"read_{cls.__get_format(fname)}")
        return f(fname, **kwargs)


class ExtendedBaseModel(BaseModel):
    """A extended base class for creating Pydantic models.

    This class support many file formats:
        json        by `pydantic`
        yaml/yml    by `pyyaml`
        toml        by `tomli_w`, `tomli`, & `toml`
    """

    @pydantic.validate_call
    def write_toml(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        indent: int = 4,
        **kwargs,
    ) -> pydantic.FilePath:
        kwargs["exclude_none"] = True
        with filename.open("wb") as f:
            toml_w.dump(
                self.model_dump(mode="python", **kwargs),
                f,
                indent=indent,
                multiline_strings=True,
            )
        return filename

    @pydantic.validate_call
    def write_yaml(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        indent: int | None = 2,
        **kwargs,
    ) -> pydantic.FilePath:
        with filename.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(mode="python", **kwargs),
                f,
                encoding="utf-8",
                default_flow_style=False,
                indent=indent,
            )
        return filename

    @classmethod
    @pydantic.validate_call
    def read_yaml(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        with filename.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data, **kwargs)

    @classmethod
    @pydantic.validate_call
    def read_toml(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        with filename.open("rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data, **kwargs)


class NpzPklBaseModel(BaseModel):
    """A extended base class for creating Pydantic models.

    This class support many file formats:
        json        by `pydantic`
        pickle      by `joblib`
        npz         by `numpy`

    Note: only numpy ndarray & numpy-compatible scalar value supported.
    """

    @classmethod
    def _convert(cls) -> dict[str, tuple[tuple, str]]:
        """Override if needed to specify the dtype and shape of attributes.

        Note: must be implemented by children class.

        Example:
            return dict(a=((None, 3), "uint8"))
        This `None` indicates it can take any shape along the first axis.
        """
        return {}

    @classmethod
    def _npz_available(cls) -> bool:
        return True

    @staticmethod
    def __validate_ndarray_and_convert(  # noqa: D103
        data: np.ndarray | Sequence,
        shape: Sequence[int | None],
        dtype: str,
    ) -> np.ndarray:
        assert shape.count(None) <= 1, (type(shape), shape)
        shape = tuple(int(i) if i is not None else -1 for i in shape)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        return np.asarray(data, dtype).reshape(shape)

    @pydantic.model_validator(mode="before")
    @classmethod
    def __convert(cls, data):
        if isinstance(data, dict):
            for key, (shape, dtype) in cls._convert().items():
                if key in data and data[key] is not None:
                    kw = dict(data=data[key], shape=shape, dtype=dtype)
                    data[key] = cls.__validate_ndarray_and_convert(**kw)  # type: ignore
        return data

    @pydantic.model_validator(mode="after")
    def __check_shape(self) -> Self:
        for k in self.__pydantic_fields__:
            v = getattr(self, k, None)
            if v is None or np.isscalar(v) or isinstance(v, np.ndarray):
                pass
            else:
                raise ValueError(f"Invalid value: {v}")
        return self

    @override
    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        for k in self.__pydantic_fields__:
            v0 = getattr(self, k, None)
            v1 = getattr(other, k, None)
            if v0 is None or np.isscalar(v0):
                if v0 is None or np.isscalar(v0):
                    if v0 != v1:
                        return False
                else:
                    return False
            elif isinstance(v0, np.ndarray):
                if isinstance(v1, np.ndarray):
                    if v0.shape != v1.shape:
                        return False
                    elif np.any(v0 != v1):
                        return False
                else:
                    return False
            else:
                raise NotImplementedError(f"Field({k}): {type(v0)}")
        return True

    @pydantic.validate_call
    def write_pickle(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        compress: bool | int | tuple[str, int] = 3,
        **kwargs,
    ) -> pydantic.FilePath:
        """Persist the dictionary of this object into one file.

        Read more in the reference:
          https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html

        Args:
            filename (str | Path | TextIOWrapper): The file object or path
                of the file in which it is to be stored. The compression
                method corresponding to one of the supported filename
                extensions ('.z', '.gz', '.bz2', '.xz' or '.lzma')
                will be used automatically.
            compress (bool | int | tuple[str, int], optional):
                Optional compression level for the data.
                    0 or False is no compression.
                    Higher value means more compression, but also
                        slower read and write times.
                    Using a value of 3 is often a good compromise.
                Defaults to 3.
            exclude_defaults (bool): ...
            exclude_none (bool): ...
            **kwargs: ...

        Note:
            If compress is True, the compression level used is 3.
            If compress is a 2-tuple, the first element must correspond to
                a string between supported compressors (e.g 'zlib', 'gzip',
                'bz2', 'lzma' 'xz'), the second element must be an integer
                from 0 to 9, corresponding to the compression level.
        """
        kwargs["exclude_none"] = True
        kwargs["exclude_defaults"] = False
        dump(
            self.model_dump(mode="python", **kwargs),
            filename,
            compress=compress,  # type: ignore
        )
        return filename

    @pydantic.validate_call
    def write_npz(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        compress: bool = True,
        **kwargs,
    ) -> pydantic.FilePath:
        if not self._npz_available():
            raise RuntimeError(
                "Subclass must have class method named"
                " `_npz_available()` to be True."
            )
        kwargs["exclude_none"] = True
        kwargs["exclude_defaults"] = False
        f_savez = np.savez_compressed if compress else np.savez
        data = self.model_dump(mode="python", **kwargs)
        dct: dict[str, ArrayLike] = {}
        for k, v in data.items():
            if not isinstance(v, dict):
                dct[k] = v
        f_savez(
            filename,
            allow_pickle=False,
            **self.model_dump(mode="python", **kwargs),
        )
        return filename

    @classmethod
    @pydantic.validate_call
    def read_npz(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        if not cls._npz_available():
            raise RuntimeError(
                "Subclass must have class method named"
                " `_npz_available()` to be True."
            )
        return cls.model_validate(np.load(filename), **kwargs)

    @classmethod
    @pydantic.validate_call
    def read_pickle(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        return cls.model_validate(load(filename), **kwargs)
