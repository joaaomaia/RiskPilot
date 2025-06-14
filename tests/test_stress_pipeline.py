from importlib.util import find_spec

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from riskpilot.evaluation import BinaryPerformanceEvaluator
from riskpilot.synthetic import SyntheticVintageGenerator

kaleido_available = find_spec("kaleido") is not None
pytestmark = pytest.mark.skipif(not kaleido_available, reason="kaleido not installed")


def test_run_stress_pipeline(tmp_path):
    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    df["id"] = range(100)
    df["date"] = pd.date_range("2020-01-01", periods=100)

    train = df.iloc[:80].reset_index(drop=True)
    test = df.iloc[80:].reset_index(drop=True)

    gen = SyntheticVintageGenerator(id_cols=["id"], date_cols=["date"]).fit(train)
    model = LogisticRegression().fit(
        train[[f"f{i}" for i in range(4)]], train["target"]
    )

    bev = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        synthetic_gen=gen,
        stress_periods=1,
        stress_freq="D",
    )
    report = bev.run_stress_test()
    assert "meta" in report and "sha256" in report["meta"]
