from collections.abc import Callable
from pathlib import Path
from typing import Self, override

import numpy as np
from pydantic import FilePath, NewPath, validate_call
from sklearn.metrics.pairwise import cosine_similarity

from graphatoms.dataclasses._numpydantic import NDArray
from graphatoms.dataclasses._pydanticModel import OurBaseModel
from graphatoms.system import Cluster  # type: ignore
from graphatoms.system.database.abc import DatabaseABC


class Scheduler(OurBaseModel):
    dR: dict[str, NDArray] = {}

    @override
    def _string(self) -> str:  # type: ignore
        return " ".join(f"{k}:{len(v)}dR" for k, v in self.dR.items())

    @validate_call
    @override
    def write_npz(  # type: ignore
        self,
        filename: FilePath | NewPath,
        *,
        compress: bool = True,
        **kwargs,
    ) -> FilePath:
        save: Callable = np.savez_compressed if compress else np.savez
        save(filename, allow_pickle=False, **self.dR)
        return filename

    @classmethod
    @validate_call
    @override
    def read_npz(  # type: ignore
        cls,
        filename: FilePath,
        *args,
        **kwargs,
    ) -> Self:
        return cls(dR=dict(np.load(filename)))

    def can_be_skip(
        self,
        cluster: Cluster | str,
        diffpositions: np.ndarray,
        thetacutoff: float = 30.0,
    ) -> bool:
        assert 0 <= thetacutoff <= 180, "thetacutoff must be in 0-180"
        key = cluster if isinstance(cluster, str) else self.get_key_of(cluster)
        if key not in self.dR:
            self.dR[key] = diffpositions.reshape(1, -1)
            return False

        dR: NDArray = self.dR[key]
        dR0 = diffpositions.reshape(1, -1)
        assert dR.ndim == 2 and dR.shape[1] == dR0.shape[1]
        cos: np.ndarray = cosine_similarity(dR, dR0)
        if np.max(cos) < np.cos(np.deg2rad(thetacutoff)):
            self.dR[key] = np.vstack((dR, dR0))  # row_stack
            return False
        return True

    @staticmethod
    def get_key_of(cluster: Cluster) -> str:
        return DatabaseABC.get_key_of(cluster)


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    obj = Scheduler()

    plt.figure(figsize=(5, 5))
    for i in range(200):
        point = np.random.rand(2) - np.array([0.5, 0.5])
        skip = obj.can_be_skip(
            "aaa",
            diffpositions=point,
            thetacutoff=15.0,
        )
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
