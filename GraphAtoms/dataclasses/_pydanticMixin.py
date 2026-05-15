# ruff: noqa: D100 D102 E501
from collections.abc import Mapping, Sequence
from sys import version_info

import numpy as np
import pydantic
import tomli_w as toml_w
import yaml
from joblib import dump, load
from omegaconf import OmegaConf
from typing_extensions import Any, Self

from GraphAtoms.utils import bytestool as bytesutils

if version_info < (3, 11):
    import tomli as tomllib  # type: ignore
else:
    import tomllib


class PydanticConvertFactoryMixin(pydantic.BaseModel):
    """Mixin for serializing Pydantic models to various formats.

    Supports: dict, str, bytes (with optional compression), json, yaml, toml, npz, pickle.
    """

    @pydantic.validate_call
    def to_dict(
        self,
        *,
        exclude_none: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_computed_fields: bool = True,
        numpy_ndarray_compatible: bool = True,
        numpy_convert_to_list: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert model to dictionary."""
        kwargs.update(
            dict(
                mode="python",
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_computed_fields=exclude_computed_fields,
            )
        )
        result = self.model_dump(**kwargs)
        if numpy_ndarray_compatible:
            for k, v0 in result.items():
                v1 = getattr(self, k, None)
                if type(v0) is not type(v1) and isinstance(v1, np.ndarray):
                    result[k] = v1
                if np.isscalar(v1):
                    pass
                elif isinstance(v1, list | tuple):
                    assert isinstance(np.asarray(list(v1)).flat[0], np.str_), (
                        "Only list[str] is supported."
                    )
                else:
                    assert isinstance(v1, np.ndarray) or (
                        "Only numpy array or scalar is supported. "
                        f"But got `{type(v0)}` for {k}."
                    )
        if numpy_convert_to_list:
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    result[k] = v.tolist()
        return result

    @pydantic.validate_call
    def to_str(
        self,
        *,
        indent: int | None = None,
        ensure_ascii: bool = True,
        exclude_none: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_computed_fields: bool = True,
        **kwargs,
    ) -> str:
        """Convert model to JSON string."""
        kwargs.update(
            dict(
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_computed_fields=exclude_computed_fields,
                # ensure_ascii=True will be bigger (~2-3x) than
                # ensure_ascii=False for non-ascii characters.
                ensure_ascii=ensure_ascii,
                indent=indent,
            )
        )
        return self.model_dump_json(**kwargs)

    @pydantic.validate_call
    def to_bytes(
        self,
        compressformat: str | None = None,
        compresslevel: int = 0,
        exclude_none: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_computed_fields: bool = True,
        **kwargs,
    ) -> bytes:
        kwargs.update(
            dict(
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_computed_fields=exclude_computed_fields,
                # ensure_ascii=True will be bigger (~2-3x) than
                # ensure_ascii=False for non-ascii characters.
                ensure_ascii=True,
                indent=None,
            )
        )
        obj: str = self.to_str(**kwargs)
        if compressformat is None:
            return obj.encode("utf-8")
        else:
            return bytesutils.compress(
                obj,
                encoding="utf-8",
                format=compressformat,
                compresslevel=compresslevel,
            )

    @classmethod
    def SUPPORTED_CONVERT_FORMATS(cls) -> Sequence[str]:
        fmts = bytesutils.SUPPORTED_COMPRESS_FORMATS
        result = tuple(f"bytes-{i}" for i in fmts)
        return result + ("bytes", "str", "dict")

    @pydantic.validate_call
    def convert_to(self, format="bytes", **kw) -> Any:
        assert format in self.SUPPORTED_CONVERT_FORMATS(), (
            f"Invalid format: {format}. Only "
            f"{self.SUPPORTED_CONVERT_FORMATS()} are available."
        )
        if format.startswith("bytes"):
            if format == "bytes":
                kw["compressformat"] = None
            else:
                format, kw["compressformat"] = format.split("-")
        return getattr(self, f"to_{format}")(**kw)

    @classmethod
    @pydantic.validate_call
    def convert_from(cls, data: Any, format="bytes", **kw) -> Self:
        assert format in cls.SUPPORTED_CONVERT_FORMATS(), (
            f"Invalid format: {format}. Only "
            f"{cls.SUPPORTED_CONVERT_FORMATS()} are available."
        )
        if format.startswith("bytes"):
            if format == "bytes":
                kw["compressformat"] = None
            else:
                format, kw["compressformat"] = format.split("-")
        return getattr(cls, f"from_{format}")(data, **kw)

    @classmethod
    @pydantic.validate_call
    def from_bytes(
        cls,
        data: bytes,
        compressformat: str | None = None,
        **kwargs,
    ) -> Self:
        if compressformat is not None:
            data = bytesutils.decompress(value=data, format=compressformat)
        return cls.from_str(data=data.decode("utf-8"), **kwargs)

    @classmethod
    @pydantic.validate_call
    def from_str(cls, data: str, **kw) -> Self:
        dct = cls.model_validate_json(data).to_dict()
        return cls.from_dict(dct, **kw)

    @classmethod
    @pydantic.validate_call
    def from_dict(cls, data: Mapping[str, Any], **kw) -> Self:
        return cls.model_validate({**data, **kw})


class __Json(PydanticConvertFactoryMixin):
    @pydantic.validate_call
    def write_json(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        indent: int | None = 4,
        exclude_none: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_computed_fields: bool = True,
        **kwargs,
    ) -> pydantic.FilePath:
        kwargs.update(
            dict(
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_computed_fields=exclude_computed_fields,
                ensure_ascii=True,
                indent=indent,
            )
        )
        filename.write_text(self.to_str(**kwargs), "utf-8")
        return filename

    @classmethod
    @pydantic.validate_call
    def read_json(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        return cls.model_validate_json(filename.read_bytes(), **kwargs)


class __Toml(PydanticConvertFactoryMixin):
    @pydantic.validate_call
    def write_toml(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        exclude_computed_fields: bool = True,
        exclude_defaults: bool = False,
        exclude_unset: bool = False,
        exclude_none: bool = True,
        indent: int = 4,
        **kwargs,
    ) -> pydantic.FilePath:
        kwargs.update(
            dict(
                exclude_computed_fields=exclude_computed_fields,
                exclude_defaults=exclude_defaults,
                exclude_unset=exclude_unset,
                exclude_none=exclude_none,
                numpy_convert_to_list=True,
                numpy_ndarray_compatible=True,
            )
        )
        with filename.open("wb") as f:
            toml_w.dump(
                self.to_dict(**kwargs),
                f,
                indent=indent,
                multiline_strings=True,
            )
        return filename

    @classmethod
    @pydantic.validate_call
    def read_toml(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        with filename.open("rb") as f:
            return cls.from_dict(
                tomllib.load(f),
                **kwargs,
            )


class __Yaml(PydanticConvertFactoryMixin):
    @pydantic.validate_call
    def write_yaml(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        exclude_computed_fields: bool = True,
        exclude_defaults: bool = False,
        exclude_unset: bool = False,
        exclude_none: bool = True,
        indent: int | None = 2,
        **kwargs,
    ) -> pydantic.FilePath:
        kwargs.update(
            dict(
                exclude_computed_fields=exclude_computed_fields,
                exclude_defaults=exclude_defaults,
                exclude_unset=exclude_unset,
                exclude_none=exclude_none,
                numpy_convert_to_list=True,
                numpy_ndarray_compatible=True,
            )
        )
        with filename.open("w", encoding="utf-8") as f:
            f.write(
                OmegaConf.to_yaml(
                    self.to_dict(**kwargs),
                    # f,
                    # encoding="utf-8",
                    # # default_flow_style=False,
                    # indent=4,
                )
            )
        return filename

    @classmethod
    @pydantic.validate_call
    def read_yaml(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        with filename.open(encoding="utf-8") as f:
            return cls.from_dict(
                yaml.safe_load(f),
                **kwargs,
            )


class __Npz(PydanticConvertFactoryMixin):
    @pydantic.validate_call
    def write_npz(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        exclude_computed_fields: bool = True,
        exclude_defaults: bool = False,
        exclude_unset: bool = False,
        exclude_none: bool = True,
        compress: bool = True,
        **kwargs,
    ) -> pydantic.FilePath:
        kwargs.update(
            dict(
                exclude_computed_fields=exclude_computed_fields,
                exclude_defaults=exclude_defaults,
                exclude_unset=exclude_unset,
                exclude_none=exclude_none,
                numpy_convert_to_list=False,
                numpy_ndarray_compatible=True,
            )
        )
        (np.savez_compressed if compress else np.savez)(
            filename,
            allow_pickle=False,
            **self.to_dict(**kwargs),
        )
        return filename

    @classmethod
    @pydantic.validate_call
    def read_npz(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        return cls.from_dict(np.load(filename), **kwargs)


class __Pickle(PydanticConvertFactoryMixin):
    @pydantic.validate_call
    def write_pickle(
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        compress: bool | int | tuple[str, int] = 3,
        exclude_computed_fields: bool = True,
        exclude_defaults: bool = False,
        exclude_unset: bool = False,
        exclude_none: bool = True,
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
            exclude_computed_fields: bool = True,
            exclude_defaults (bool): ...
            exclude_unset: bool = False,
            exclude_none (bool): ...
            **kwargs: ...

        Note:
            If compress is True, the compression level used is 3.
            If compress is a 2-tuple, the first element must correspond to
                a string between supported compressors (e.g 'zlib', 'gzip',
                'bz2', 'lzma' 'xz'), the second element must be an integer
                from 0 to 9, corresponding to the compression level.
        """
        kwargs.update(
            dict(
                exclude_computed_fields=exclude_computed_fields,
                exclude_defaults=exclude_defaults,
                exclude_unset=exclude_unset,
                exclude_none=exclude_none,
                numpy_convert_to_list=False,
                numpy_ndarray_compatible=True,
            )
        )
        dump(self.to_dict(**kwargs), filename, compress=compress)
        return filename

    @classmethod
    @pydantic.validate_call
    def read_pickle(cls, filename: pydantic.FilePath, **kwargs) -> Self:
        return cls.from_dict(load(filename), **kwargs)


class PydanticIoFactoryMixin(__Json, __Toml, __Yaml, __Npz, __Pickle):
    @classmethod
    def SUPPORTED_IO_FORMATS(cls) -> Sequence[str]:
        return ("json", "toml", "yaml", "yml", "npz", "pickle", "pkl")

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
    def write(
        self,
        fname: pydantic.FilePath | pydantic.NewPath,
        **kwargs,
    ) -> pydantic.FilePath:
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
