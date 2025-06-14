import numpy as np
import pandas as pd

from riskpilot.synthetic import SyntheticVintageGenerator


def test_synthetic_generator_basic():
    df = pd.DataFrame(
        {
            "id": range(10),
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "a": np.arange(10, dtype=float),
            "b": list("abcdefghij"),
        }
    )
    gen = SyntheticVintageGenerator(id_cols=["id"], date_cols=["date"])
    gen.fit(df)
    synth = gen.generate(n_periods=2, freq="D", n_per_vintage=5)
    assert set(["id", "date", "a", "b"]).issubset(synth.columns)
    assert len(synth) == 10
