"""The utils for strings."""

from random import sample
from string import ascii_lowercase

from ase.db.core import convert_str_to_int_float_bool_or_str

from GraphAtoms.utils import bytes as bytesutils

__all__ = ["compress", "decompress", "hash", "random"]


def compress(
    str_value: str,
    format: str = "snappy",
    encoding: str = "utf-8",
    compresslevel: int = 0,
) -> bytes:
    """Return the compressed bytes of the given string."""
    return bytesutils.compress(
        str_value.encode(encoding),
        format=format,
        compresslevel=compresslevel,
    )


def decompress(
    value: bytes,
    format: str = "snappy",
    encoding: str = "utf-8",
) -> str:
    """Return the decompressed string of the given bytes."""
    return bytesutils.decompress(value, format=format).decode(encoding)


def __hash(v: str, digest_size: int) -> str:
    return bytesutils.hashlib.blake2b(
        v.encode(encoding="utf-8"),
        digest_size=int(digest_size / 2),
    ).hexdigest()


def hash(
    value: str,
    algo: str = "md5",
    digest_size: int = 6,
) -> str:
    """Return the hashing string of the given value.

    Note: ensure the result cannot be convert to bool, int and float.

    Args:
        value (str): The input value
        algo (str, optional): The hash algorithms. Defaults to "md5".
        digest_size (int, optional): Defaults to 6.

    Returns:
        bytes | str: The output value
    """
    _value = bytesutils.hash(value, True, algo)
    result = __hash(_value, digest_size)  # type: ignore
    _value = convert_str_to_int_float_bool_or_str(result)
    while not isinstance(_value, str):
        result = __hash(f"{_value}_{result}", digest_size)
        _value = convert_str_to_int_float_bool_or_str(result)
    return result


def random(length: int = 6) -> str:
    """Return the random string that has the given length."""
    return "".join(sample(ascii_lowercase, int(length)))
