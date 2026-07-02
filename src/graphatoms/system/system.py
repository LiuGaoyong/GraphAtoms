from typing import Self

import numpy as np
from pydantic import model_validator

from graphatoms.system.graph import SysGraph
from graphatoms.utils.bytestool import hash_string

from .bonds import DEFAULT_WH_HASH_SIZE


class System(SysGraph):
    @model_validator(mode="after")
    def __some_keys_should_be_none(self) -> Self:
        msg = "The `{:s}` should be None for " + f"{self.__class__.__name__}."
        assert self.coordination is None, msg.format("coordination")
        return self

    def get_site_core(
        self,
        max_ncore: int = 3,
        exclude_adsorbate_atoms: bool = False,
    ) -> np.ndarray:
        """Get the site core for the adsorption process."""
        result = analysis_site(
            self,  # type: ignore
            active=self.is_outer,
            max_ncore=max_ncore,
        )
        if exclude_adsorbate_atoms and self.is_adsorbate is not None:
            not_adsorbate = np.logical_not(self.is_adsorbate)
            result = np.logical_and(result, not_adsorbate)
        return result


def analysis_site(
    sys: System,
    active: np.ndarray | None = None,
    max_ncore: int = 3,
) -> np.ndarray:
    """A helper function for analysis of `graphatoms.system.System`."""
    sys.hash  # call hash calculation to set hash & hashes
    assert sys.pair is not None, "Please run analysis first on the System."
    assert sys.hashes is not None, "Please run analysis first on the System."
    assert sys.is_outer is not None, "Please run analysis first on the System."
    assert 0 <= max_ncore <= 6, "max_ncore must be greater than or equal to 0."
    if active is None:
        active = sys.is_outer
    else:
        active0 = sys.is_outer
        assert active.shape == active0.shape
        active = active & active0
    assert isinstance(active, np.ndarray)
    arange = np.arange(len(sys))
    result = [
        np.isin(arange, [i])
        for i in arange  #
        if active[i]
    ]
    if max_ncore >= 2:
        result.extend(
            [
                np.isin(arange, [i, j])
                for i, j in sys.pair  #
                if active[i] and active[j]
            ]
        )
    result = np.asarray(result)
    if max_ncore >= 3:
        result_2 = sys.get_chordless_cycles(
            batch_nbr_order=0,
            max_ncore=max_ncore,  # type: ignore
            batch=active,
        )
        result = np.vstack([result, result_2])
    hashes = [
        hash_string(
            "-".join([sys.hashes[i] for i in range(len(sys)) if site[i]]),
            digest_size=DEFAULT_WH_HASH_SIZE,
        )
        for site in result
    ]
    _, idx = np.unique(hashes, return_index=True)
    return result[idx]
