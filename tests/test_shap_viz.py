from importlib.util import find_spec

import numpy as np
import plotly.graph_objects as go
import pytest
import shap

from riskpilot.evaluation import BinaryPerformanceEvaluator

shap_available = find_spec("shap") is not None
skip_if_no_shap = pytest.mark.skipif(not shap_available, reason="shap not installed")


def _dummy_evaluator():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(
        n_samples=50, n_features=3, n_informative=2, n_redundant=0, random_state=0
    )
    df = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    import pandas as pd

    df = pd.DataFrame(df, columns=["a", "b", "c", "target"])
    train = df.iloc[:30].copy()
    test = df.iloc[30:].copy()
    model = LogisticRegression().fit(train[["a", "b", "c"]], train["target"])
    return BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=[],
        homogeneous_group=2,
        font_family="Verdana",
    )


@skip_if_no_shap
def test_beeswarm_smoke():
    bev = _dummy_evaluator()
    expl = shap.Explanation(
        np.random.randn(40, 3),
        base_values=np.zeros(40),
        data=np.random.randn(40, 3),
        feature_names=["a", "b", "c"],
    )
    fig = bev._build_shap_beeswarm(expl, max_display=2)
    assert isinstance(fig, go.Figure)
    assert fig.layout.font.family == bev.font_family


@skip_if_no_shap
def test_dependence_and_waterfall_smoke():
    bev = _dummy_evaluator()
    expl1 = shap.Explanation(
        np.random.randn(30, 3),
        base_values=np.zeros(30),
        data=np.random.randn(30, 3),
        feature_names=["a", "b", "c"],
    )
    expl2 = shap.Explanation(
        np.random.randn(20, 3),
        base_values=np.zeros(20),
        data=np.random.randn(20, 3),
        feature_names=["a", "b", "c"],
    )
    fig = bev._build_shap_dependence({"train": expl1, "test": expl2}, feature="a")
    assert isinstance(fig, go.Figure)
    fig2 = bev._build_shap_waterfall(expl2, row_index=0)
    assert isinstance(fig2, go.Figure)
