from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Self, override

import numpy as np
from pydantic import BaseModel, FilePath, NewPath, validate_call
from sklearn.metrics.pairwise import cosine_similarity

from graphatoms.dataclasses._numpydantic import NDArray
from graphatoms.system import SysGraph  # type: ignore


class Scheduler(BaseModel):
    dR: dict[str, NDArray] = {}
    is_fixed: dict[str, NDArray] = {}

    @override
    def __str__(self) -> str:
        msg = " ".join(f"{k}:{len(v)}dR" for k, v in self.dR.items())
        return f"{self.__class__.__name__}({msg})"

    @validate_call
    def write_npz(
        self,
        filename: FilePath | NewPath,
        *,
        compress: bool = True,
        **kwargs,
    ) -> FilePath:
        save: Callable = np.savez_compressed if compress else np.savez
        dct = {f"{k}_is_fixed": v for k, v in self.is_fixed.items()}
        dct = {f"{k}_dR": v for k, v in self.dR.items()} | dct
        save(filename, allow_pickle=False, **dct)
        return filename

    @classmethod
    @validate_call
    def read_npz(cls, filename: FilePath, *args, **kwargs) -> Self:
        data: dict[str, dict[str, NDArray]] = defaultdict(dict)
        dct: Mapping[str, NDArray] = np.load(filename)
        for k, v in dct.items():
            k2, k1 = k.split("_")
            data[k1][k2] = v
        return cls(**data)

    def can_be_skip(
        self,
        graph: SysGraph | str,
        diffpositions: np.ndarray,
        *,
        is_fixed: np.ndarray | None = None,
        thetacutoff: float = 30.0,
        **kwargs,
    ) -> tuple[bool, float]:
        diffpositions = np.asarray(diffpositions, dtype=float)
        print(diffpositions)
        assert 0 <= thetacutoff <= 180, "thetacutoff must be in 0-180"
        if isinstance(graph, str):
            if is_fixed is None:
                is_fixed = np.zeros_like(diffpositions, dtype=bool).flatten()
            else:
                is_fixed = np.asarray(is_fixed, dtype=bool)
                if is_fixed.ndim == 1:
                    lst = [is_fixed, is_fixed, is_fixed]
                    is_fixed = np.column_stack(lst).flatten()
                elif is_fixed.ndim == 2:
                    assert is_fixed.shape == diffpositions.shape
                    is_fixed = is_fixed.flatten()
                else:
                    raise ValueError(
                        "is_fixed must be 1D or 2D, "
                        + f"but we got {is_fixed.shape}"
                    )
            key = str(graph)
        elif isinstance(graph, SysGraph):
            key = graph.get_key_for_metadata()
            if graph.is_fix is None:
                is_fixed = np.zeros_like(diffpositions, dtype=bool).flatten()
            else:
                lst = [graph.is_fix, graph.is_fix, graph.is_fix]
                is_fixed = np.column_stack(lst).flatten()
        else:
            raise ValueError(
                "graph must be str or SysGraph, "
                + f"but we got type={type(graph)}"
            )
        dR_new = diffpositions.flatten()[~is_fixed].reshape(1, -1)

        if key not in self.dR:
            self.dR[key] = dR_new
            self.is_fixed[key] = is_fixed
            # cosine similarity is -1.0 if
            # the angle is 180 degrees
            return (False, -1.0)
        else:
            assert np.all(is_fixed == self.is_fixed[key]), (
                f"is_fixed must be the same for the same graph({key})."
            )
            dR: NDArray = self.dR[key]
            assert dR.ndim == 2 and dR.shape[1] == dR_new.shape[1]
            cos: np.ndarray = cosine_similarity(dR, dR_new)
            if np.max(cos) <= np.cos(np.deg2rad(thetacutoff)):
                self.dR[key] = np.vstack((dR, dR_new))  # row_stack
                print(self.dR[key])
                return False, np.max(cos)
            return True, np.max(cos)


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    arr = np.array([[1, 2], [3, 4], [5, 6]])
    print(arr)
    print(arr.reshape(1, -1))
    print(arr.flatten())
    print(arr.reshape(1, -1).flatten())
    print(arr.flatten().reshape(1, -1))

    obj = Scheduler()

    plt.figure(figsize=(5, 5))
    for i in range(20):
        print("-----------------")
        point = np.random.rand(2) - np.array([0.5, 0.5])
        skip, cos = obj.can_be_skip(
            "aaa",
            diffpositions=point,
            thetacutoff=15.0,
        )
        print(skip, cos, point)
        if not skip:
            plt.plot(
                [0, point[0]],
                [0, point[1]],
                c="blue",
            )
        plt.scatter(
            [point[0]],
            [point[1]],
            c="red" if skip else "blue",
            label="skip" if skip else "keep",
        )
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    plt.title("Blue: Keep; Red: Skip")

    p = Path(__file__).with_suffix(".png")
    plt.savefig(p)
    obj.write_npz(p.with_suffix(".npz"))
