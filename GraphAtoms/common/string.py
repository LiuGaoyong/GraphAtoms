"""The utils for strings."""

from hashlib import blake2b
from random import sample
from string import ascii_lowercase

from ase.db.core import convert_str_to_int_float_bool_or_str


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
