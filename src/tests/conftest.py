import os
import time

import numpy as np
import pytest


def return_big_object(i) -> np.ndarray:
    time.sleep(0.1)
    print(i, "adfasd")
    return i * np.ones((10000, 200), dtype=np.float64)


@pytest.fixture(scope="session", autouse=True)
def set_ray_env() -> None:
    os.environ["RAY_DEDUP_LOGS"] = "0"


