# ruff: noqa: D100, D101, D102, D103
import warnings
from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from GraphAtoms.common.abc import BaseModel, NpzPklBaseModel
from GraphAtoms.common.ndarray import NDArray


class MockNpzBaseModel(NpzPklBaseModel):
    arr: NDArray = np.random.rand(5, 3)
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
            assert new_obj == obj

    @pytest.mark.parametrize("format", ["pkl", "npz", "json"])
    def test_convert_numpy(self, format: str) -> None:
        self.__run(MockNpzBaseModel(), MockNpzBaseModel, format)


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-s", "--maxfail=1"])
