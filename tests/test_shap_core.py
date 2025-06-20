import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from riskpilot.evaluation import BinaryPerformanceEvaluator
from .conftest import shap_available


@shap_available
def test_beeswarm_runs():
    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    df["id"] = range(len(df))
    train = df.iloc[:40]
    test = df.iloc[40:]
    model = LogisticRegression().fit(
        train[[f"f{i}" for i in range(4)]], train["target"]
    )

    bev = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
    )

    expl = bev._compute_shap_values(train[bev.predictor_cols], split_name="train")
    fig = bev._build_shap_beeswarm(expl)
    assert isinstance(fig, go.Figure)
