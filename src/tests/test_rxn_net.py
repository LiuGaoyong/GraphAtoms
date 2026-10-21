from pathlib import Path

import pandas as pd
import pytest

from graphatoms.enterpoint.network import MetaData
from graphatoms.enterpoint.network._metadata import _MetaDataTable

DATA_DIR = Path(__file__).parent.parent / "tests-datasets" / "for-rxnet"


def test_ReactionNetwork_metadata() -> None:
    metadata = MetaData.from_storage(DATA_DIR)
    info = metadata.read(0)
    df = pd.DataFrame([i.to_dict() for i in [info, info.reversed]])
    print()
    print(df.T)
    # print(metadata.bkl_solver([1] * len(metadata), [1] * len(metadata)))


def _mixed_null_table_dataframe() -> pd.DataFrame:
    """DataFrame containing both kinds of events with missing keys.

    The missing values are ``nan``, exactly like the ones returned by
    ``pandas.read_feather`` since pandas 3.0 (the default string dtype
    uses ``numpy.nan`` as the missing value marker).
    """
    nan = float("nan")
    return pd.DataFrame(
        {
            "key_rxn": ["r0", "r1"],
            "key_r": ["a", "b"],
            "key_g": ["gas-0", nan],
            "key_t": [nan, "ts-1"],
            "key_p": ["p0", "p1"],
            "Ea_forword": [1.0, 2.0],
            "rate_forword": [1.0, 2.0],
            "rate_reversed": [1.0, 2.0],
            "Ea_reversed": [1.0, 2.0],
            "for_cluster": ["c0", "c1"],
            "for_system": ["", ""],
            "count": [1, 1],
            "dE": [0.1, -0.1],
        }
    )


def test_metadata_table_from_dataframe_with_missing_keys() -> None:
    """Regression test for pandas>=3.0 NaN missing value markers."""
    table = _MetaDataTable.from_dataframe(_mixed_null_table_dataframe())
    assert table.key_g == ["gas-0", None]
    assert table.key_t == [None, "ts-1"]
    assert len(table) == 2


def test_metadata_persistence_with_missing_keys(tmp_path: Path) -> None:
    """The table with missing optional keys can be persisted and reloaded."""
    table = _MetaDataTable.from_dataframe(_mixed_null_table_dataframe())
    MetaData(table=table).persistence(tmp_path)
    reloaded = MetaData.from_storage(tmp_path)
    assert reloaded.table.key_g == ["gas-0", None]
    assert reloaded.table.key_t == [None, "ts-1"]
    assert reloaded.read(0).key_t is None
    assert reloaded.read(1).key_g is None


if __name__ == "__main__":
    print(DATA_DIR)
    pytest.main([__file__, "-s"])
