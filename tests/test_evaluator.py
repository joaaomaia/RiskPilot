import os
import sys

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from binary_performance_evaluator import BinaryPerformanceEvaluator


def _create_split():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    df["id"] = range(len(df))
    df["date"] = pd.date_range("2020-01-01", periods=len(df))
    df["grp"] = np.where(np.arange(len(df)) % 2 == 0, "A", "B")
    train = df.iloc[:150].reset_index(drop=True)
    test = df.iloc[150:].reset_index(drop=True)
    return train, test


def test_auto_grouping():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])

    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        group_col="segment",
        homogeneous_group="auto",
    )

    assert evaluator.group_ is not None
    assert evaluator.binning_table_ is not None
    assert evaluator.df_train["segment"].notna().all()


def test_radar_plot_returns_figure():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])

    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        homogeneous_group=2,
    )

    fig = evaluator.plot_group_radar(features=["f0", "f1"])
    assert isinstance(fig, go.Figure)


def test_decile_ks_wrapper():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])

    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        group_col="grp",
        homogeneous_group=None,
    )

    fig = evaluator.plot_decile_ks(n_bins=5)
    assert isinstance(fig, go.Figure)


def test_group_col_required_without_auto():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])

    with pytest.raises(ValueError):
        BinaryPerformanceEvaluator(
            model=model,
            df_train=train,
            df_test=test,
            target_col="target",
            id_cols=["id"],
            date_col="date",
            homogeneous_group=None,
        )


def test_event_rate_plot_with_auto_groups():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])

    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        homogeneous_group="auto",
    )

    fig = evaluator.plot_event_rate()
    assert isinstance(fig, go.Figure)
