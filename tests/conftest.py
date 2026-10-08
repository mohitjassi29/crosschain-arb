import numpy as np
import pandas as pd
import pytest

from crosschain_arb.config import DATA_DIR


def pytest_collection_modifyitems(config, items):
    if (DATA_DIR / "eth_gas_with_depth.parquet").exists():
        return
    skip = pytest.mark.skip(reason="real data files not present in data/")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def toy_panel():
    """1 Hz panel with a known gap structure.

    seconds 0-9 positive, 10-14 negative, 15-16 positive, 17-29 negative, 30-39 positive.
    """
    signs = [1] * 10 + [-1] * 5 + [1] * 2 + [-1] * 13 + [1] * 10
    n = len(signs)
    return pd.DataFrame(
        {
            "time": pd.date_range("2023-03-01", periods=n, freq="s"),
            "basis_t": np.array(signs) * np.arange(1, n + 1, dtype=float),
            "latency_t": 3.0,
            "gas_gwei": 20.0,
            "depth": 0.5,
            "spike_dummy": 0,
        }
    )
