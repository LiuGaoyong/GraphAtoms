import pickle

import numpy as np
import pytest


class HelperException(Exception):
    def __init__(
        self,
        msg: str,
        cost_time: float | None = None,
        label: str | None = None,
    ) -> None:
        cost = f"CostTime={cost_time:.2f}. " if cost_time is not None else ""
        label = f" for ({label})" if label is not None else ""
        super().__init__(cost + msg + label)


class OptimizationFailed(HelperException):
    def __init__(
        self,
        type: str,
        cost_time: float | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(
            msg=f"{type} optimization not coveraged",
            cost_time=cost_time,
            label=label,
        )


class CheckVibrationFailed(HelperException):
    def __init__(
        self,
        fqmin: float | None = None,
        frequencies: np.ndarray | None = None,
        cost_time: float | None = None,
        label: str | None = None,
    ) -> None:
        if frequencies is None:
            fstr = ""
        else:
            fstr = ",".join(f"{f:.2f}" for f in frequencies[:3])
            if fqmin is None:
                fstr = f"({fstr})"
            else:
                fstr = f"({fstr}|FQMIN={fqmin:.2f})"
        super().__init__(
            msg=f"check frequencies{fstr} failed",
            cost_time=cost_time,
            label=label,
        )


@pytest.mark.parametrize(
    "exc",
    [
        HelperException(msg="test"),
        HelperException(msg="test", cost_time=1.23, label="dimer"),
    ],
)
def test_helper_exception_picklable(exc):
    restored = pickle.loads(pickle.dumps(exc))
    assert type(restored) is type(exc)
    # assert restored.msg == exc.msg
    # assert restored.cost_time == exc.cost_time
    # assert restored.label == exc.label


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
