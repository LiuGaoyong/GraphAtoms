"""The utils for strings."""

import bz2  # Added in Python 3.3
import gzip  # Added in Python 3.2
import zlib  # Added in Python 3.0
from collections.abc import Callable
from functools import partial
from hashlib import blake2b
from random import sample
from string import ascii_lowercase
from typing import Annotated, Literal

import pydantic
from ase.db.core import convert_str_to_int_float_bool_or_str


class CompressorAndDecompressor:
    """The Compressor and Decompressor."""

    @pydantic.validate_call
    def __init__(  # noqa: D107
        self,
        format: Literal["z", "gz", "bz2", "xz", "lzma"] = "xz",
        compresslevel: Annotated[int, pydantic.Field(ge=0, le=9)] = 0,
    ) -> None:
        if format == "z":
            self.__func_compress: Callable[..., bytes] = partial(
                zlib.compress,
                level=compresslevel if compresslevel != 0 else -1,
            )
            self.__func_decompress: Callable[..., bytes] = zlib.decompress
        elif format == "gz":
            self.__func_compress: Callable[..., bytes] = partial(
                gzip.compress,
                compresslevel=compresslevel if compresslevel != 0 else 9,
            )
            self.__func_decompress: Callable[..., bytes] = gzip.decompress
        elif format == "bz2":
            self.__func_compress: Callable[..., bytes] = partial(
                bz2.compress,
                compresslevel=compresslevel if compresslevel != 0 else 9,
            )
            self.__func_decompress: Callable[..., bytes] = bz2.decompress
        else:
            import lzma  # Added in Python 3.3

            self.__func_compress: Callable[..., bytes] = partial(
                lzma.compress,
                format=lzma.FORMAT_XZ if format == "xz" else lzma.FORMAT_ALONE,
            )
            self.__func_decompress: Callable[..., bytes] = lzma.decompress

    @pydantic.validate_call
    def compress(self, value: bytes) -> bytes:
        """Return the compressed bytes of the given bytes."""
        return self.__func_compress(value)

    @pydantic.validate_call
    def decompress(self, value: bytes) -> bytes:
        """Return the decompressed bytes of the given bytes."""
        return self.__func_decompress(value)


def compress_string(
    str_value: str,
    format: Literal["z", "gz", "bz2", "xz", "lzma"] = "xz",
    encoding: str = "utf-8",
    compresslevel: int = 0,
) -> bytes:
    """Return the compressed bytes of the given string."""
    return CompressorAndDecompressor(
        format=format,
        compresslevel=compresslevel,
    ).compress(str_value.encode(encoding))


def decompress_string(
    value: bytes,
    format: Literal["z", "gz", "bz2", "xz", "lzma"] = "xz",
    encoding: str = "utf-8",
) -> str:
    """Return the decompressed string of the given bytes."""
    return (
        CompressorAndDecompressor(format=format)
        .decompress(value)
        .decode(encoding)
    )


def random_string(length: int = 6) -> str:
    """Return the random string that has the given length."""
    return "".join(sample(ascii_lowercase, int(length)))


def _hash(str_value: str, digest_size: int = 6) -> str:
    obj = blake2b(
        str_value.encode("ascii"),
        digest_size=int(digest_size / 2),
    )
    return obj.hexdigest()


def hash_string(str_value: str, digest_size: int = 6) -> str:
    """Return the hashing string of the given string.

    Note: ensure the result cannot be convert to bool, int and float.
    """
    result: str = _hash(str_value, digest_size=digest_size)
    value = convert_str_to_int_float_bool_or_str(result)
    while not isinstance(value, str):
        result = _hash(f"{value}_{result}", digest_size)
        value = convert_str_to_int_float_bool_or_str(result)
    return result
