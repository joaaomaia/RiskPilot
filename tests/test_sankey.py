import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import LogisticRegression

from riskpilot.evaluation import BinaryPerformanceEvaluator


def _make_df(overlap=True, mismatch_dtype=False):
    rng = np.random.RandomState(0)
    n = 20
    X1 = rng.randn(n, 3)
    X2 = rng.randn(n, 3)
    df1 = pd.DataFrame(X1, columns=["a", "b", "c"])
    df2 = pd.DataFrame(X2, columns=["a", "b", "c"])
    df1["target"] = rng.randint(0, 2, n)
    df2["target"] = rng.randint(0, 2, n)
    df1["id"] = np.arange(n)
    df2_ids = np.arange(n) if overlap else np.arange(n, 2 * n)
    if mismatch_dtype:
        df2["id"] = [f"A{i}" for i in df2_ids]
    else:
        df2["id"] = df2_ids
    df1["date"] = pd.Timestamp("2022-01-01")
    df2["date"] = pd.Timestamp("2022-02-01")
    return df1.reset_index(drop=True), df2.reset_index(drop=True)


def _evaluator(df1, df2):
    model = LogisticRegression().fit(
        pd.concat([df1, df2])[["a", "b", "c"]],
        pd.concat([df1, df2])["target"],
    )
    return BinaryPerformanceEvaluator(
        model=model,
        df_train=df1,
        df_test=df2,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        homogeneous_group=2,
    )


def test_sankey_normal_case():
    df1, df2 = _make_df()
    bev = _evaluator(df1, df2)
    figs, metrics = bev.plot_sankey_migration("2022-01", "2022-02")
    assert isinstance(figs, list) and isinstance(figs[0], go.Figure)
    assert len(figs[0].data[0].link.value) > 0
    assert metrics


def test_sankey_no_overlap():
    df1, df2 = _make_df(overlap=False)
    bev = _evaluator(df1, df2)
    with pytest.raises(ValueError):
        bev.plot_sankey_migration("2022-01", "2022-02")


def test_sankey_mismatched_dtype():
    df1, df2 = _make_df(mismatch_dtype=True)
    bev = _evaluator(df1, df2)
    with pytest.raises(ValueError):
        bev.plot_sankey_migration("2022-01", "2022-02")
