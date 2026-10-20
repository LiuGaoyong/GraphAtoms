"""The module of utils for bytes."""

import gzip
import hashlib
import os
import string
import uuid
import zlib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, Any

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


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory so a rename is durable (POSIX)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(target: Path, writer: Callable[[Path], Any]) -> None:
    """Write `target` atomically.

    The `writer` callback receives a temporary path in the same
    directory and must create/write the file there. The temp file is
    fsynced and then os.replace()d onto `target`. Because both paths
    live in the same directory, os.replace is atomic on POSIX.

    The temp file keeps `target.suffix`, so callers such as
    ``np.savez`` that auto-append ".npz" behave correctly.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_stem = f".{target.stem}.tmp.{os.getpid()}{uuid.uuid4().hex}"
    tmp = target.parent / f"{file_stem}{target.suffix}"
    try:
        writer(tmp)
        # ensure the file is synced to disk
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(target: Path, text: str, encoding="utf-8") -> None:
    atomic_write(target, lambda p: p.write_text(text, encoding=encoding))


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
