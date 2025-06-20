from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from riskpilot.evaluation import BinaryPerformanceEvaluator
from tests.conftest import shap_available


def _data():
    X, y = make_classification(
        n_samples=50,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=["a", "b"])
    df["target"] = y
    df["id"] = range(len(df))
    df["date"] = pd.date_range("2020-01-01", periods=len(df))
    train = df.iloc[:30].reset_index(drop=True)
    test = df.iloc[30:].reset_index(drop=True)
    return train, test


@shap_available
@pytest.mark.parametrize(
    "plot_type",
    ["bar", "layered", "beeswarm", "dependence", "trend", "waterfall"],
)
def test_plot_shap_smoke(plot_type, monkeypatch):
    train, test = _data()
    model = LogisticRegression().fit(train[["a", "b"]], train["target"])
    bev = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
    )

    monkeypatch.setattr(bev, "export_report", lambda *a, **k: Path("."))

    kwargs = {"max_display": 3}
    if plot_type in {"dependence", "trend"}:
        kwargs["focus_feature"] = "a"
    if plot_type == "trend":
        kwargs["min_samples"] = 1
    if plot_type == "waterfall":
        kwargs["record_index"] = 0

    out = bev.plot_shap(
        splits=["train"],
        plot_type=plot_type,
        reference_split="train",
        summary=True,
        return_data=True,
        **kwargs,
    )
    assert "figures" in out and "data" in out and "summary" in out
    assert all(isinstance(f, go.Figure) for f in out["figures"])
    assert len(out["summary"]) <= 3
