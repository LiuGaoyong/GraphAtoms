import os
from typing import override

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


import pandas as pd

from ._0abc import BaseABC


class ThirdStep(BaseABC):
    """The class for exploring the surface process."""

    @override
    def run(self) -> pd.DataFrame:
        raise NotImplementedError
