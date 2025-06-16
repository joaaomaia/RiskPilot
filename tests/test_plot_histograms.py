import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scipy.stats
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from riskpilot.evaluation import BinaryPerformanceEvaluator
from riskpilot.evaluation.binary_performance_evaluator import _compute_psi


def _split():
    X, y = make_classification(n_samples=120, n_features=3, n_informative=2, n_redundant=0, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = y
    df["id"] = range(len(df))
    df["date"] = pd.date_range("2020-01-01", periods=len(df), freq="MS")
    train = df.iloc[:60].reset_index(drop=True)
    test = df.iloc[60:].reset_index(drop=True)
    return train, test


def test_psi_zero_identical():
    train, _ = _split()
    counts, _ = np.histogram(train["a"], bins=10)
    psi = _compute_psi(counts, counts)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_ks_detects_shift():
    train, test = _split()
    test_shift = test.copy()
    test_shift["a"] += 5
    ks = scipy.stats.ks_2samp(train["a"], test_shift["a"])
    assert ks.pvalue < 0.05


def test_plot_histograms_errors():
    train, test = _split()
    model = LogisticRegression().fit(train[["a", "b", "c"]], train["target"])
    bev = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
    )

    with pytest.raises(ValueError):
        bev.plot_histograms(feature="z", show=False)

    with pytest.warns(UserWarning):
        bev.plot_histograms(
            feature="a",
            reference={"train": [209901]},
            compare={"test": [209902]},
            show=False,
        )
