from typing import override

import numpy as np
import pandas as pd

from graphatoms.system import Cluster, System  # type: ignore

from ._0abc import BaseABC


class ThirdStep(BaseABC):
    """The class for Matching process step."""

    @override
    def run(self) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def helper_match(
        cluster: Cluster,
        system: System,
    ) -> None | np.ndarray:
        """Helper function for matching process."""
        result = system.get_match_mode(  # type: ignore
            pattern=cluster,
            algorithm="lad",
            return_match_target=True,
            only_number_color=False,
            only_count=False,
        )
        assert not isinstance(result, int)
        return result
