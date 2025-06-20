import logging
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from riskpilot.evaluation import BinaryPerformanceEvaluator
from tests.conftest import shap_available


@shap_available
def test_explainer_selection(caplog):
    import shap

    X, y = make_classification(n_samples=50, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    df["id"] = range(len(df))
    train = df.iloc[:40].reset_index(drop=True)
    test = df.iloc[40:].reset_index(drop=True)

    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(
        train[[f"f{i}" for i in range(4)]], train["target"]
    )
    bev_rf = BinaryPerformanceEvaluator(
        model=rf,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
    )
    assert isinstance(bev_rf._get_shap_explainer(), shap.TreeExplainer)

    lr = LogisticRegression().fit(train[[f"f{i}" for i in range(4)]], train["target"])
    bev_lr = BinaryPerformanceEvaluator(
        model=lr,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
    )
    assert isinstance(bev_lr._get_shap_explainer(), shap.LinearExplainer)

    kn = KNeighborsClassifier().fit(train[[f"f{i}" for i in range(4)]], train["target"])
    bev_kn = BinaryPerformanceEvaluator(
        model=kn,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
    )
    with caplog.at_level(logging.WARNING):
        expl = bev_kn._get_shap_explainer()
    assert isinstance(expl, shap.KernelExplainer)
    assert any("KernelExplainer" in r.message for r in caplog.records)
