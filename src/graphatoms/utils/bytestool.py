"""The module of utils for bytes."""

import gzip
import hashlib
import string
import zlib
from collections.abc import Callable
from functools import partial
from typing import Annotated

from numpy.random import choice
from pydantic import Field, validate_call

try:
    import bz2
    import lzma
except ImportError:
    bz2, lzma = None, None
try:
    import snappy
except ImportError:
    snappy = None
SUPPORTED_COMPRESS_FORMATS = ["z", "gz"]
for m, lst in [
    # (zlib, ["z"]),
    # (gzip, ["gz"]),
    (bz2, ["bz2"]),
    (lzma, ["lzma", "xz"]),
    (snappy, ["snappy"]),
]:
    if m is not None:
        SUPPORTED_COMPRESS_FORMATS.extend(lst)
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
            assert bz2 is not None, "bz2 is not available."
            self.__func_compress: Callable[..., bytes] = partial(
                bz2.compress,
                compresslevel=compresslevel if compresslevel != 0 else 9,
            )
            self.__func_decompress: Callable[..., bytes] = bz2.decompress
        elif format == "snappy":
            assert snappy is not None, "snappy is not available."
            self.__func_compress: Callable[..., bytes] = snappy.compress
            self.__func_decompress: Callable[..., bytes] = partial(
                snappy.uncompress, decoding=None
            )  # type: ignore
        else:
            assert lzma is not None, "lzma is not available."
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
    value: bytes | str,
    format: str = "gz",
    encoding: str = "utf-8",
    compresslevel: int = 0,
) -> bytes:
    """Return the compressed bytes of the given bytes.

    Parameters
    ----------
    value: bytes | str
        The bytes or string to be compressed.
    encoding: str, optional
        The encoding of the string. Defaults to "utf-8".
        It only takes effect when the value's type is str.
    compresslevel: int, optional
        The compression level. Defaults to 0.
        It only takes effect when the format is "z/gz/bz2".
    format: str, optional
        The compression format. Defaults to "gz".
    """
    if isinstance(value, str):
        value = value.encode(encoding)
    return __CompressorAndDecompressor(
        format=format,
        compresslevel=compresslevel,
    ).compress(value)


def decompress(value: bytes, format: str = "gz") -> bytes:
    """Return the decompressed bytes of the given bytes."""
    return __CompressorAndDecompressor(format=format).decompress(value)


@validate_call
def hash(
    value: bytes | str,
    return_string: bool = True,
    algo: str = "blake2b",
) -> bytes | str:
    """Return the hashing string of the given value.

    Args:
        value (bytes | str): The input value
        return_string (bool, optional): Defaults to True.
        algo (str, optional): The hash algorithms. Defaults to "blake2b".

    Returns:
        bytes | str: The output value
    """
    assert algo in SUPPORTED_HASHLIB_FORMATS, (
        f"Invalid algorithm: {algo}. Only "
        f"{SUPPORTED_HASHLIB_FORMATS} are available."
    )
    if not isinstance(value, bytes):
        value = value.encode("utf-8")  # type: ignore
    h = hashlib.new(algo)
    h.update(value)
    return h.hexdigest() if return_string else h.digest()


@validate_call
def hash_string(value: bytes | str, algo: str = "blake2b") -> str:
    """Return the hashing string of the given value.

    Args:
        value (bytes | str): The input value
        algo (str, optional): The hash algorithms. Defaults to "blake2b".

    Returns:
        str: The output value of 32 characters.

    """
    if not isinstance(value, bytes):
        value = value.encode("utf-8")  # type: ignore
    assert isinstance(value, bytes), "The `value` must be bytes."
    return hashlib.blake2b(
        value,
        digest_size=16,
        person=b"GraphAtoms",
    ).hexdigest()


def random_string(n: int) -> str:
    pool: list[str] = list(string.ascii_letters + string.digits)
    return "".join(choice(pool, size=int(n)))


#######################################################################
#                                   Test
#######################################################################


def test_compress_decompress() -> None:
    for fmt in SUPPORTED_COMPRESS_FORMATS:
        data = random_string(1000)
        compressed = compress(data, fmt, "utf-8")
        data0 = decompress(compressed, fmt)
        assert data == data0.decode("utf-8")
        result = hash_string(data)
        print(len(result), result)
    lst = list(string.ascii_letters + string.digits)
    print(len(lst))


if __name__ == "__main__":
    test_compress_decompress()
