import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

from riskpilot.evaluation import BinaryPerformanceEvaluator


def _split():
    X, y = make_classification(
        n_samples=60,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = y
    df["id"] = range(len(df))
    df["date"] = pd.date_range("2020-01-01", periods=len(df))
    train = df.iloc[:40]
    test = df.iloc[40:]
    return train, test


def test_seaborn_plots_smoke():
    train, test = _split()
    model = LogisticRegression().fit(train[["a", "b", "c"]], train["target"])
    bev = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        homogeneous_group="auto",
    )
    figs = bev.plot_event_rate()
    assert all(isinstance(f, go.Figure) for f in figs)
