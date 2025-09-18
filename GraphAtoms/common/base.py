# ruff: noqa: D100 D102
from collections.abc import Hashable, Mapping, Sequence
from pickle import dumps, loads
from sys import version_info
from typing import Annotated, override

import numpy as np
import pydantic
import tomli_w as toml_w
import yaml
from joblib import dump, load
from numpy.typing import ArrayLike
from typing_extensions import Any, Self

from GraphAtoms.utils import bytes as bytesutils

if version_info < (3, 11):
    import tomli as tomllib  # type: ignore
else:
    import tomllib
__all__ = [
    "BaseModel",
    "ExtendedBaseModel",
    "NpzPklBaseModel",
]


class __Json(pydantic.BaseModel):
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


class __Toml(pydantic.BaseModel):
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

    @classmethod
    @pydantic.validate_call
    def read_toml(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        with filename.open("rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data, **kwargs)


class __Yaml(pydantic.BaseModel):
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


class __Npz(pydantic.BaseModel):
    @pydantic.validate_call
    def write_npz(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        compress: bool = True,
        **kwargs,
    ) -> pydantic.FilePath:
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
        return cls.model_validate(np.load(filename), **kwargs)


class __Pickle(pydantic.BaseModel):
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

    @classmethod
    @pydantic.validate_call
    def read_pickle(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        return cls.model_validate(load(filename), **kwargs)


class _IoFactoryMixin(__Json):
    @classmethod
    def SUPPORTED_IO_FORMATS(cls) -> tuple[str]:
        return ("json",)

    @classmethod
    @pydantic.validate_call
    def __get_format(cls, fname: pydantic.FilePath | pydantic.NewPath) -> str:
        format: str = fname.name.split(".")[-1].lower()
        assert format in cls.SUPPORTED_IO_FORMATS(), (
            f"Invalid format: {format}. Only "
            f"{cls.SUPPORTED_IO_FORMATS()} are available."
        )
        if format in ("yml", "yaml"):
            format = "yaml"
        elif format == "json":
            format = "json"
        elif format == "toml":
            format = "toml"
        elif format == "npz":
            format = "npz"
        elif format in ("pkl", "pickle"):
            format = "pickle"
        return format

    @pydantic.validate_call
    def write(self, fname: pydantic.FilePath, **kwargs) -> pydantic.FilePath:
        """The factory method for writing file of this object.

        Support: json, yml/yaml, toml, pkl(pickle).
        """
        f = getattr(self, f"write_{self.__get_format(fname)}")
        return f(fname, **kwargs)

    @classmethod
    @pydantic.validate_call
    def read(cls, fname: pydantic.FilePath, **kwargs) -> Self:
        """The factory classmethod for reading file.

        Support: json, yml/yaml, toml, pkl(pickle).
        """
        f = getattr(cls, f"read_{cls.__get_format(fname)}")
        return f(fname, **kwargs)


class __Bytes(pydantic.BaseModel):
    @pydantic.validate_call
    def to_bytes(
        self,
        compressformat: str = "snappy",
        compresslevel: Annotated[int, pydantic.Field(ge=0, le=9)] = 0,
        **kw,
    ) -> bytes:
        """Return the json bytes of this object."""
        if "exclude_none" not in kw:
            kw["exclude_none"] = True
        return bytesutils.compress(
            dumps(self.model_dump_json(**kw)),
            format=compressformat,  # type: ignore
            compresslevel=compresslevel,
        )

    @classmethod
    @pydantic.validate_call
    def from_bytes(
        cls,
        data: bytes,
        compressformat: str = "snappy",
        **kw,
    ) -> Self:
        return cls.model_validate_json(
            loads(
                bytesutils.decompress(
                    data,
                    format=compressformat,  # type: ignore
                )
            ),
            **kw,
        )


class __Str(pydantic.BaseModel):
    @pydantic.validate_call
    def to_str(self, **kw) -> str:
        """Return the json string of this object."""
        if "exclude_none" not in kw:
            kw["exclude_none"] = True
        return self.model_dump_json(**kw)

    @classmethod
    @pydantic.validate_call
    def from_str(cls, data: str, **kw) -> Self:
        return cls.model_validate_json(data, **kw)


class _ConvertFactoryMixin(__Bytes, __Str):
    @pydantic.validate_call
    def convert_to(self, format="bytes", **kw) -> Any:
        return getattr(self, f"to_{format}")(**kw)

    @classmethod
    @pydantic.validate_call
    def convert_from(cls, data: Any, format="bytes", **kw) -> Self:
        return getattr(cls, f"from_{format}")(data, **kw)


class BaseModel(_IoFactoryMixin, _ConvertFactoryMixin, Hashable):
    """A base class for creating Pydantic models.

    This class support many file formats:
        json        by `pydantic`
    And it support many object format:
        bytes
        str
    """

    model_config = pydantic.ConfigDict(frozen=True)

    def _string(self) -> str:
        """The string representation of this class.

        The expression like `str(obj)` will output
        `f"{self.__class__.__name__}({this_value})"`.
        """
        return "object"

    @classmethod
    def _convert(cls) -> dict[str, tuple[tuple, str]]:
        """Override if needed to specify the dtype and shape of attributes.

        Note: must be implemented by children class.

        Example:
            return dict(a=((-1, 3), "uint8"))
        This `None` indicates it can take any shape along the first axis.
        """
        return {}

    @staticmethod
    def __validate_ndarray_and_convert(  # noqa: D103
        data: np.ndarray | Sequence,
        shape: Sequence[int | None],
        dtype: str,
    ) -> np.ndarray:
        assert shape.count(None) <= 1, (type(shape), shape)
        shape = tuple(int(i) if i is not None else -1 for i in shape)
        data = np.asarray(data) if not isinstance(data, np.ndarray) else data
        return np.asarray(data, dtype).reshape(shape)  # .tobytes()

    @pydantic.model_validator(mode="before")
    @classmethod
    def __convert(cls, data) -> dict[str, Any]:
        if isinstance(data, dict):
            for key, (shape, dtype) in cls._convert().items():
                if key in data and data[key] is not None:
                    kw = dict(data=data[key], shape=shape, dtype=dtype)
                    data[key] = cls.__validate_ndarray_and_convert(**kw)  # type: ignore
        return data

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._string()})"

    @override
    def __repr__(self) -> str:
        return super().__repr__()

    @override
    def __hash__(self) -> int:
        return hash(bytesutils.hash(self.to_bytes()))

    @override
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
                if abs(v0 - v1) > 1e-9:  # type: ignore
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
                if not np.allclose(v0, v1):
                    return False
            else:
                raise NotImplementedError(
                    "Unsupported type for __eq__: ", type0
                )
        return True


class ExtendedBaseModel(BaseModel, __Yaml, __Toml, __Pickle):
    """A extended base class for creating Pydantic models.

    This class support many file formats:
        json        by `pydantic`
        yaml/yml    by `pyyaml`
        pickle      by `joblib`
        toml        by `tomli_w`, `tomli`, & `toml`
    And it support many object format:
        bytes
        str
    """

    @classmethod
    @override
    def SUPPORTED_IO_FORMATS(cls) -> tuple[str]:
        result: tuple[str] = super().SUPPORTED_IO_FORMATS()
        return result + ("yaml", "yml", "toml", "pickle", "pkl")  # type: ignore

    @override
    def __hash__(self) -> int:
        return hash(bytesutils.hash(self.to_bytes()))


class NpzPklBaseModel(BaseModel, __Npz, __Pickle):
    """A extended base class for creating Pydantic models.

    This class support many file formats:
        json        by `pydantic`
        pickle      by `joblib`
        npz         by `numpy`
    And it support many object format:
        bytes
        str

    Note: only numpy ndarray & numpy-compatible scalar value supported.
    """

    @classmethod
    @override
    def SUPPORTED_IO_FORMATS(cls) -> tuple[str]:
        result: tuple[str] = super().SUPPORTED_IO_FORMATS()
        return result + ("npz", "pickle", "pkl")  # type: ignore

    @override
    def __hash__(self) -> int:
        return hash(bytesutils.hash(self.to_bytes()))
