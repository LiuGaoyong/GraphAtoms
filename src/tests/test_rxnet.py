from pathlib import Path

import pandas as pd
import pytest

from graphatoms.enterpoint.network import MetaData

DATA_DIR = Path(__file__).parent.parent / "tests-datasets" / "for-rxnet"


def test_rxnet_metadata() -> None:
    metadata = MetaData.from_storage(DATA_DIR)
    info = metadata.read(0)
    df = pd.DataFrame([i.to_dict() for i in [info, info.reversed]])
    print()
    print(df.T)
    # print(metadata.bkl_solver([1] * len(metadata), [1] * len(metadata)))


if __name__ == "__main__":
    print(DATA_DIR)
    pytest.main([__file__, "-s"])
