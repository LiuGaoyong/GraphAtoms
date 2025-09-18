# ruff: noqa: D100, D101, D102, D103
import itertools
import warnings
from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory

import numpy as np
import pydantic
import pytest

from GraphAtoms.common.base import BaseModel


class MockBaseModel(BaseModel):
    arr: pydantic.Base64Bytes = np.random.rand(5, 3).tobytes()
    arr2: list[str] = ["1", "sd"]
    v: float = 5.0

    # def model_dump_json(self, *args, **kwargs) -> str:
    #     return self.__pydantic_serializer__.to_json(
    #         self,
    #         *args,
    #         **kwargs,
    #     ).decode("ISO-8859-1")


class Test_ABC_Pydantic_Model:
    def test_json_schema(self):
        print()
        obj = MockBaseModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pprint(MockBaseModel.model_json_schema())
        print(repr(obj))
        print(str(obj))

    @pytest.mark.parametrize("format", ["yaml", "toml", "json", "pkl"])
    def test_io(self, format: str) -> None:
        with TemporaryDirectory(delete=False) as tmp:
            obj = MockBaseModel()
            fname = Path(tmp) / f"data.{format}"
            fname.touch()
            fname2 = obj.write(fname)
            assert fname.exists()
            assert fname2 == fname
            new_obj = MockBaseModel.read(fname)
            assert isinstance(new_obj, MockBaseModel)
            print(repr(new_obj))
            print(repr(obj))
            assert new_obj == obj

    @pytest.mark.parametrize("format", ["bytes", "str"])
    def test_convert(self, format: str) -> None:
        obj = MockBaseModel()
        middle_obj = obj.convert_to(format)
        obj2 = MockBaseModel.convert_from(middle_obj, format)
        assert isinstance(obj2, MockBaseModel)
        assert obj2 == obj


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
def test_compress(fmt, level: int) -> None:
    obj = MockBaseModel()
    b = obj.to_bytes(compressformat=fmt, compresslevel=level)
    obj2 = MockBaseModel.from_bytes(b, compressformat=fmt)
    print(fmt, level, len(b), sep="\t")
    assert obj == obj2


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
