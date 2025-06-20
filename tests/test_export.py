from importlib.util import find_spec

import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from riskpilot.evaluation import BinaryPerformanceEvaluator

kaleido_available = find_spec("kaleido") is not None
pytestmark = pytest.mark.skipif(not kaleido_available, reason="kaleido not installed")


def _evaluator(tmp_path):
    X, y = make_classification(
        n_samples=20,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=["a", "b"])
    df["target"] = y
    df["id"] = range(len(df))
    train = df.iloc[:10].reset_index(drop=True)
    test = df.iloc[10:].reset_index(drop=True)
    model = LogisticRegression().fit(train[["a", "b"]], train["target"])
    return BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        save_dir=tmp_path,
    )


def test_export_paths_exist(tmp_path):
    bev = _evaluator(tmp_path)
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    df = pd.DataFrame(
        {
            "feature": ["a"],
            "split": ["train"],
            "importance": [0.5],
            "variation_flag": [False],
        }
    )
    out_dir = bev.export_report(
        figs=fig, summary_df=df, bullets=["ok"], export_dir=tmp_path
    )
    assert any(out_dir.glob("*.png"))
    assert (out_dir / "shap_summary.csv").is_file()
    assert (out_dir / "shap_summary.html").is_file()
    assert (out_dir / "insights.txt").is_file()


def test_export_zip(tmp_path):
    bev = _evaluator(tmp_path)
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    out_dir = bev.export_report(figs=fig, export_dir=tmp_path, zip_bundle=True)
    zip_path = out_dir.with_suffix(".zip")
    assert zip_path.is_file()


def test_export_multiple_formats(tmp_path):
    bev = _evaluator(tmp_path)
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    out_dir = bev.export_report(figs=fig, export_dir=tmp_path, formats=("png", "svg"))
    assert (out_dir / "fig_0.png").is_file()
    assert (out_dir / "fig_0.svg").is_file()
