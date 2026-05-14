from importlib.util import find_spec

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from riskpilot.evaluation import BinaryPerformanceEvaluator

optbinning_available = find_spec("optbinning") is not None
skip_if_no_optbinning = pytest.mark.skipif(not optbinning_available, reason="optbinning not installed")


def _confusion_heat(fig):
    return np.asarray(fig.axes[0].collections[0].get_array()).reshape(2, 2)


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


@skip_if_no_optbinning
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


def test_plot_confusion_default_uses_evaluator_threshold_with_arrays():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])
    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        threshold=0.35,
    )
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.20, 0.40, 0.45, 0.80])

    fig = evaluator.plot_confusion(y_true, y_pred_proba)

    np.testing.assert_array_equal(_confusion_heat(fig), np.array([[1, 1], [0, 2]]))
    assert evaluator.threshold == 0.35


def test_plot_confusion_explicit_threshold_overrides_evaluator_threshold():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])
    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        threshold=0.35,
    )
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.20, 0.40, 0.45, 0.80])

    fig = evaluator.plot_confusion(y_true, y_pred_proba, threshold=0.50)

    np.testing.assert_array_equal(_confusion_heat(fig), np.array([[2, 0], [1, 1]]))
    assert evaluator.threshold == 0.35


def test_plot_confusion_internal_split_uses_evaluator_threshold():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])
    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        threshold=0.35,
    )
    proba = np.resize(np.array([0.20, 0.40, 0.45, 0.80]), len(evaluator.df_test))
    y_true = np.resize(np.array([0, 0, 1, 1]), len(evaluator.df_test))
    evaluator.df_test[evaluator.score_col_] = proba
    evaluator.df_test[evaluator.target_col] = y_true

    fig = evaluator.plot_confusion(splits=["test"])
    expected = confusion_matrix(y_true, (proba >= 0.35).astype(int), labels=[0, 1])

    np.testing.assert_array_equal(_confusion_heat(fig), expected)
    assert evaluator.threshold == 0.35


def test_plot_confusion_symbolic_threshold_still_works():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])
    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        threshold=0.35,
    )

    fig = evaluator.plot_confusion(splits=["test"], threshold="ks")

    assert fig.axes
    assert evaluator.threshold == 0.35


def test_init_without_groups_uses_default_none_and_compute_metrics():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])

    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
    )

    assert evaluator.homogeneous_group is None
    assert evaluator.group_ is None
    assert evaluator.binning_table_ is None
    metrics = evaluator.compute_metrics()
    assert "AUC_ROC" in metrics.columns


def test_init_explicit_none_without_group_col_compute_metrics():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])

    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        group_col=None,
        homogeneous_group=None,
    )

    assert evaluator.group_ is None
    metrics = evaluator.compute_metrics()
    assert metrics.loc["Test", "metric_status"] == "ok"


def test_group_col_validated_when_provided_without_auto():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])

    with pytest.raises(KeyError):
        BinaryPerformanceEvaluator(
            model=model,
            df_train=train,
            df_test=test,
            target_col="target",
            id_cols=["id"],
            date_col="date",
            group_col="missing_group",
            homogeneous_group=None,
        )


@skip_if_no_optbinning
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

    figs = evaluator.plot_event_rate()
    assert isinstance(figs, tuple) and len(figs) == 2
    assert all(isinstance(f, go.Figure) for f in figs)


@skip_if_no_optbinning
def test_binning_table_method():
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

    assert evaluator.binning_table() is not None


def test_custom_plot_title():
    train, test = _create_split()
    model = LogisticRegression().fit(train[[f"f{i}" for i in range(5)]], train["target"])

    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
    )

    fig = evaluator.plot_calibration(title="My Plot")
    assert fig.layout.title.text == "My Plot"
