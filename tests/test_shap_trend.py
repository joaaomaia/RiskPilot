import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import plotly.graph_objects as go

from riskpilot.evaluation import BinaryPerformanceEvaluator


class DummyExpl:
    def __init__(self, values, feature_names):
        self.values = np.asarray(values)
        self.feature_names = feature_names


def _evaluator():
    X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b"])
    df["target"] = y
    df["id"] = range(len(df))
    df["date"] = pd.date_range("2020-01-01", periods=len(df))
    train = df.iloc[:30].reset_index(drop=True)
    test = df.iloc[30:].reset_index(drop=True)
    model = LogisticRegression().fit(train[["a", "b"]], train["target"])
    return BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
    )


def test_prepare_time_series_basic():
    bev = _evaluator()
    shap_dict = {
        "train": DummyExpl([[1, 2], [3, 4]], ["a", "b"]),
    }
    dates = {"train": pd.Series(pd.to_datetime(["2020-01-05", "2020-02-05"]))}
    ts = bev._prepare_shap_time_series(shap_dict, date_lookup=dates, freq="M", min_samples=1)
    jan = ts.loc[(ts["period"] == pd.Period("2020-01", "M")) & (ts["feature"] == "a"), "importance"].iloc[0]
    feb = ts.loc[(ts["period"] == pd.Period("2020-02", "M")) & (ts["feature"] == "b"), "importance"].iloc[0]
    assert jan == 1.0
    assert feb == 4.0


def test_prepare_time_series_topk():
    bev = _evaluator()
    shap_dict = {
        "train": DummyExpl(np.ones((3, 5)), [f"f{i}" for i in range(5)]),
    }
    dates = {"train": pd.Series(pd.date_range("2020-01-01", periods=3))}
    ts = bev._prepare_shap_time_series(shap_dict, date_lookup=dates, max_display=3, min_samples=1)
    assert len(ts["feature"].unique()) <= 3


def test_trend_plot_markers():
    bev = _evaluator()
    periods = pd.period_range("2020-01", periods=3, freq="M")
    ts_df = pd.DataFrame(
        {
            "period": list(periods) * 2,
            "feature": ["a"] * 6,
            "split": ["train", "test"] * 3,
            "importance": np.arange(6.0),
        }
    )
    drift = pd.DataFrame(
        {
            "period": [periods[1]],
            "feature": ["a"],
            "split": ["test"],
            "flag": [True],
            "metric": ["psi"],
            "value": [0.2],
        }
    )
    fig = bev._build_shap_trend_plot(ts_df, feature="a", splits=["train", "test"], drift_df=drift)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
