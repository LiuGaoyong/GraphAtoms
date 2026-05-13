from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, override

import numpy as np
import pytest

from GraphAtoms.dataclasses._numpydantic import NDArray, numpy_validator
from GraphAtoms.dataclasses._pydanticModel import OurBaseModel


###########################################################################
# Test for OurBaseModel(_ndarray.py & _pydantic.py & _pydantic2pyarrow.py)
class Mock(OurBaseModel):
    frequencies: Annotated[NDArray, numpy_validator()] | None = None
    energy: float | None = None

    def _string(self) -> str:
        return "Mock"

    @cached_property
    @override
    def hash(self) -> str:
        return ""


@pytest.fixture(scope="module")
def mock_object() -> Mock:
    return Mock(
        frequencies=np.random.rand(10, 3),
        energy=1.0,
    )


@pytest.mark.parametrize("format", OurBaseModel.SUPPORTED_IO_FORMATS())
def test_io_for_OurBaseModel(mock_object: Mock, format) -> None:
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / f"mock.{format}"
        p = mock_object.write(p)
        if format not in ("npz", "pickle", "pkl"):
            print("\n", "=" * 32, format, "=" * 32)
            print(p.read_text(), "\n")
        new = Mock.read(p)
        assert new == mock_object


@pytest.mark.parametrize("format", OurBaseModel.SUPPORTED_CONVERT_FORMATS())
def test_convert_for_OurBaseModel(mock_object: Mock, format) -> None:
    print()
    print("\n", "=" * 32, format, "=" * 32)
    print("Original object:\n", repr(mock_object), "\n")
    obj = mock_object.convert_to(format)
    print("Converted object:\n", obj, "\n")
    if isinstance(obj, (str, bytes)):
        print("The length of converted object:", len(obj), "\n")
    new = Mock.convert_from(obj, format)
    print("Reconstructed object:\n", repr(new), "\n")
    assert new == mock_object


def test_get_pyarrow_schema() -> None:
    print(Mock.get_pyarrow_schema())
