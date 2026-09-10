from typing import Self, override

import pydantic

from graphatoms.dataclasses._numpydantic import NDArray
from graphatoms.dataclasses._pydanticModel import OurBaseModel


class Scheduler(OurBaseModel):
    diffposition: dict[str, NDArray] = {}

    @override
    def _string(self) -> str:  # type: ignore
        return f"{len(self.diffposition)}dR"

    @pydantic.validate_call
    @override
    def write_npz(  # type: ignore
        self,
        filename: pydantic.FilePath | pydantic.NewPath,
        *,
        compress: bool = True,
        **kwargs,
    ) -> pydantic.FilePath:
        (np.savez_compressed if compress else np.savez)(
            filename,
            allow_pickle=False,
            **self.diffposition,
        )
        return filename

    @classmethod
    @pydantic.validate_call
    @override
    def read_npz(  # type: ignore
        cls,
        filename: pydantic.FilePath,
        *args,
        **kwargs,
    ) -> Self:
        return cls(diffposition=dict(np.load(filename)))


if __name__ == "__main__":
    import numpy as np

    obj = Scheduler()
    obj.diffposition["fdsafs"] = np.zeros([3, 3])
    print(obj)
    print(repr(obj))
    obj.write_npz("a.npz")  # type: ignore
    print(Scheduler.read_npz("a.npz"))  # type: ignore
