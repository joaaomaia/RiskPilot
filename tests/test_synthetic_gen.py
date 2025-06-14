import warnings

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
    assert (synth["date"].dt.normalize() == synth["date"]).all()


def test_generate_with_end_vintage():
    df = pd.DataFrame({"id": range(3), "date": pd.date_range("2024-01-01", periods=3)})
    gen = SyntheticVintageGenerator(id_cols=["id"], date_cols=["date"]).fit(df)
    end = df["date"].max() + pd.offsets.Day(5)
    synth = gen.generate(end_vintage=end, freq="D", n_per_vintage=1)
    assert synth["date"].max() == end


def test_synthetic_generator_deprecated_freq():
    df = pd.DataFrame(
        {
            "id": range(5),
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        }
    )
    gen = SyntheticVintageGenerator(id_cols=["id"], date_cols=["date"])
    gen.fit(df)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)
        gen.generate(n_periods=1, freq="M", n_per_vintage=2)
