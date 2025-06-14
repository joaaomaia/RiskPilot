import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from riskpilot.evaluation import BinaryPerformanceEvaluator
from riskpilot.evaluation.binary_performance_evaluator import _psi_single


def test_psi_single_known_value():
    base = np.array([0.25, 0.25, 0.25, 0.25])
    test = np.array([0.2, 0.3, 0.2, 0.3])
    psi = _psi_single(base, test)
    assert np.isclose(psi, 0.04054651081081642)


def test_plot_confusion_normalization():
    X, y = make_classification(
        n_samples=50,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = y
    df["id"] = range(len(df))
    df["date"] = pd.date_range("2020-01-01", periods=len(df))

    train = df.iloc[:30]
    test = df.iloc[30:]

    model = LogisticRegression().fit(train[["a", "b", "c"]], train["target"])
    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        date_col="date",
    )

    y_true = test["target"]
    y_pred_proba = model.predict_proba(test[["a", "b", "c"]])[:, 1]
    fig = evaluator.plot_confusion(y_true, y_pred_proba, normalize=True)
    cm_abs = np.array(fig.data[0].z)
    assert np.isclose(cm_abs.sum(), 1.0)
