# ruff: noqa: D100, D101, D102, D103
import itertools
import warnings
from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from numpydantic import NDArray

from GraphAtoms.common import BaseModel, ExtendedBaseModel, NpzPklBaseModel


class MockNpzBaseModel(NpzPklBaseModel):
    arr: NDArray = np.random.rand(5, 3)
    v: float = 5


class MockExtendedBaseModel(ExtendedBaseModel):
    arr: list[str] = ["1", "sd"]
    v: float = 5


class Test_ABC_Pydantic_Model:
    def test_json_schema(self):
        print()
        obj = MockNpzBaseModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pprint(obj.model_json_schema())
        print(repr(obj))
        print(str(obj))

    @staticmethod
    def __run(obj: BaseModel, cls: type[BaseModel], format: str) -> None:
        with TemporaryDirectory(delete=False) as tmp:
            fname = Path(tmp) / f"data.{format}"
            fname2 = obj.write(fname)
            assert fname.exists()
            assert fname2 == fname
            new_obj = cls.read(fname)
            assert isinstance(new_obj, cls)
            np.testing.assert_array_equal(obj.arr, new_obj.arr)  # type: ignore
            assert new_obj.__eq__(obj)

    @pytest.mark.parametrize("format", ["yaml", "toml", "json"])
    def test_convert(self, format: str) -> None:
        self.__run(MockExtendedBaseModel(), MockExtendedBaseModel, format)

    @pytest.mark.parametrize("format", ["pkl", "npz", "json"])
    def test_convert_numpy(self, format: str) -> None:
        self.__run(MockNpzBaseModel(), MockNpzBaseModel, format)


@pytest.mark.parametrize("cls", [MockExtendedBaseModel, MockNpzBaseModel])
@pytest.mark.parametrize(
    "fmt,level",
    [
        # ("xz", 0),
        # ("lzma", 0),
        ("snappy", 0),
    ]
    + sorted(
        itertools.product(
            ["z", "gz", "bz2"],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
    ),
)
def test_compress(fmt, level: int, cls: type[BaseModel]) -> None:
    obj = cls()
    b = obj.to_bytes(compressformat=fmt, compresslevel=level)
    obj2 = cls.from_bytes(b, compressformat=fmt)
    print(fmt, level, len(b), sep="\t")
    assert obj == obj2


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--maxfail=1"])
