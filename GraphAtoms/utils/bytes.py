"""The utils for bytes."""

import bz2
import gzip
import hashlib
import lzma
import zlib
from collections.abc import Callable
from functools import partial
from typing import Annotated

import snappy
from pydantic import Field, validate_call

SUPPORTED_COMPRESS_FORMATS = ["z", "gz", "bz2", "snappy", "xz", "lzma"]
SUPPORTED_HASHLIB_FORMATS = list(hashlib.algorithms_available)
__all__ = ["compress", "decompress", "hash"]


class __CompressorAndDecompressor:
    """The Compressor and Decompressor."""

    @validate_call
    def __init__(  # noqa: D107
        self,
        format: str = "snappy",
        compresslevel: Annotated[int, Field(ge=0, le=9)] = 0,
    ) -> None:
        assert format in SUPPORTED_COMPRESS_FORMATS, (
            f"Invalid format: {format}. Only "
            f"{SUPPORTED_COMPRESS_FORMATS} are available."
        )
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
        elif format == "snappy":
            self.__func_compress: Callable[..., bytes] = snappy.compress
            self.__func_decompress: Callable[..., bytes] = partial(
                snappy.uncompress, decoding=None
            )  # type: ignore
        else:
            self.__func_compress: Callable[..., bytes] = partial(
                lzma.compress,
                format=lzma.FORMAT_XZ if format == "xz" else lzma.FORMAT_ALONE,
            )
            self.__func_decompress: Callable[..., bytes] = lzma.decompress

    @validate_call
    def compress(self, value: bytes) -> bytes:
        """Return the compressed bytes of the given bytes."""
        return self.__func_compress(value)

    @validate_call
    def decompress(self, value: bytes) -> bytes:
        """Return the decompressed bytes of the given bytes."""
        return self.__func_decompress(value)


def compress(
    value: bytes,
    format: str = "snappy",
    compresslevel: int = 0,
) -> bytes:
    """Return the compressed bytes of the given bytes."""
    return __CompressorAndDecompressor(
        format=format,
        compresslevel=compresslevel,
    ).compress(value)


def decompress(
    value: bytes,
    format: str = "snappy",
) -> bytes:
    """Return the decompressed bytes of the given bytes."""
    return __CompressorAndDecompressor(format=format).decompress(value)


@validate_call
def hash(
    value: bytes | str,
    return_string: bool = True,
    algo: str = "md5",
) -> bytes | str:
    """Return the hashing string of the given value.

    Args:
        value (bytes | str): The input value
        return_string (bool, optional): Defaults to True.
        algo (str, optional): The hash algorithms. Defaults to "md5".

    Returns:
        bytes | str: The output value
    """
    assert algo in SUPPORTED_HASHLIB_FORMATS, (
        f"Invalid algorithm: {algo}. Only "
        f"{SUPPORTED_HASHLIB_FORMATS} are available."
    )
    if not isinstance(value, bytes):
        value = value.encode("utf-8")
    h = hashlib.new(algo)
    h.update(value)
    return h.hexdigest() if return_string else h.digest()
