import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from riskpilot.evaluation import BinaryPerformanceEvaluator


class DummyExpl:
    def __init__(self, values, feature_names):
        self.values = np.asarray(values)
        self.feature_names = feature_names


def _evaluator(tmp_path, monkeypatch):
    X, y = make_classification(
        n_samples=50,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = y
    df["id"] = range(len(df))
    train = df.iloc[:30].reset_index(drop=True)
    test = df.iloc[30:].reset_index(drop=True)
    model = LogisticRegression().fit(train[["a", "b", "c"]], train["target"])
    bev = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        target_col="target",
        id_cols=["id"],
        save_dir=tmp_path,
    )

    def slow_shap(X):
        time.sleep(0.05)
        return DummyExpl(np.ones((len(X), len(bev.predictor_cols))), bev.predictor_cols)

    monkeypatch.setattr(bev, "_compute_shap_values_no_cache", slow_shap)
    monkeypatch.setattr(
        "riskpilot.evaluation.binary_performance_evaluator.shap", object()
    )
    return bev


def test_cache_hit(tmp_path, monkeypatch):
    bev = _evaluator(tmp_path, monkeypatch)
    t0 = time.time()
    bev._compute_shap_values(bev.df_train[bev.predictor_cols], split_name="train")
    t1 = time.time() - t0
    t0 = time.time()
    bev._compute_shap_values(bev.df_train[bev.predictor_cols], split_name="train")
    t2 = time.time() - t0
    assert t2 < t1 * 0.5


def test_disk_cache(tmp_path, monkeypatch):
    bev1 = _evaluator(tmp_path, monkeypatch)
    bev1._compute_shap_values(bev1.df_train[bev1.predictor_cols], split_name="train")
    bev2 = _evaluator(tmp_path, monkeypatch)
    bev2._compute_shap_values(bev2.df_train[bev2.predictor_cols], split_name="train")
    assert bev2.cache_stats()["disk_hits"] == 1


def test_clear_cache(tmp_path, monkeypatch):
    bev = _evaluator(tmp_path, monkeypatch)
    bev._compute_shap_values(bev.df_train[bev.predictor_cols], split_name="train")
    path = bev._cache_path("train")
    assert path.is_file()
    bev.clear_shap_cache()
    assert not path.exists()
    assert bev.cache_stats()["size_mb"] == 0
