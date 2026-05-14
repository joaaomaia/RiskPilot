import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import LogisticRegression

from riskpilot.evaluation import BinaryPerformanceEvaluator


PREDICTORS = ["f0", "f1"]


def _make_splits(n: int = 180) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    latent = 2.8 * f0 - 1.5 * f1 + rng.normal(scale=0.35, size=n)
    target = (latent > np.median(latent)).astype(int)

    df = pd.DataFrame(
        {
            "f0": f0,
            "f1": f1,
            "target": target,
            "id": np.arange(n),
            "grp": np.where(np.arange(n) % 2 == 0, "A", "B"),
        }
    )
    df["date"] = pd.to_datetime(
        np.select(
            [
                df.index < 45,
                df.index < 90,
                df.index < 135,
            ],
            ["2020-01-01", "2020-02-01", "2020-03-01"],
            default="2020-04-01",
        )
    )
    train = df.iloc[:100].reset_index(drop=True)
    test = df.iloc[100:145].reset_index(drop=True)
    val = df.iloc[145:].reset_index(drop=True)
    return train, test, val


def _make_evaluator(
    *,
    test: pd.DataFrame | None = None,
    df_val: pd.DataFrame | None = None,
) -> tuple[BinaryPerformanceEvaluator, LogisticRegression, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, base_test, val = _make_splits()
    test = base_test if test is None else test
    model = LogisticRegression(random_state=0).fit(train[PREDICTORS], train["target"])
    evaluator = BinaryPerformanceEvaluator(
        model=model,
        df_train=train,
        df_test=test,
        df_val=df_val,
        target_col="target",
        id_cols=["id"],
        date_col="date",
        group_col="grp",
        homogeneous_group=None,
    )
    return evaluator, model, train, test, val


def test_internal_score_is_event_probability() -> None:
    evaluator, model, _, test, _ = _make_evaluator()

    pos_idx = list(model.classes_).index(1)
    expected = model.predict_proba(test[PREDICTORS])[:, pos_idx]

    np.testing.assert_allclose(evaluator.df_test[evaluator.score_col_].to_numpy(), expected)
    assert not np.allclose(evaluator.df_test[evaluator.score_col_].to_numpy(), 1 - expected)


def test_compute_metrics_includes_signed_gini_and_no_default_auto_flip() -> None:
    evaluator, _, _, _, _ = _make_evaluator()
    metrics = evaluator.compute_metrics()
    auc = metrics.loc["Test", "AUC_ROC"]
    assert auc > 0.5
    assert metrics.loc["Test", "Gini"] == pytest.approx(2 * auc - 1)

    evaluator.df_test[evaluator.score_col_] = 1 - evaluator.df_test[evaluator.score_col_]
    inverted = evaluator.compute_metrics()
    inverted_auc = inverted.loc["Test", "AUC_ROC"]

    assert inverted_auc < 0.5
    assert inverted.loc["Test", "Gini"] == pytest.approx(2 * inverted_auc - 1)
    assert inverted.loc["Test", "score_orientation"] == "event_score"


def test_single_class_by_date_metrics_and_plot_ks_are_auditable() -> None:
    _, base_test, _ = _make_splits()
    test = base_test.copy()
    test.loc[:14, "target"] = 0
    test.loc[:14, "date"] = pd.Timestamp("2020-03-01")
    test.loc[15:, "date"] = pd.Timestamp("2020-04-01")
    evaluator, _, _, _, _ = _make_evaluator(test=test)

    metrics = evaluator.compute_metrics(by_date_col=True)
    row = metrics.loc[("Test", pd.Timestamp("2020-03-01"))]

    assert row["metric_status"] == "single_class"
    assert np.isnan(row["AUC_ROC"])
    assert np.isnan(row["Gini"])
    assert np.isnan(row["KS"])

    fig = evaluator.plot_ks()
    assert isinstance(fig, go.Figure)
    assert "single_class" in set(evaluator.ks_table_["metric_status"])


def test_plot_decile_ks_defaults_to_split_specific_tables() -> None:
    train, test, val = _make_splits()
    evaluator, _, _, _, _ = _make_evaluator(test=test, df_val=val)

    fig = evaluator.plot_decile_ks(n_bins=5)
    assert isinstance(fig, go.Figure)

    table = evaluator.decile_ks_table_
    assert set(table["Split"]) == {"Train", "Test", "Val"}
    assert table.groupby("Split")["total"].sum().to_dict() == {
        "Train": len(train),
        "Test": len(test),
        "Val": len(val),
    }


def test_plot_psi_reference_last_period_uses_last_reference_month() -> None:
    evaluator, _, _, _, _ = _make_evaluator()

    _, psi_df = evaluator.plot_psi(
        feature="f0",
        reference_last_period=True,
        min_obs=1,
    )

    assert not psi_df.empty
    assert set(psi_df["reference_type"]) == {"last_period"}
    assert set(pd.to_datetime(psi_df["reference_period"]).dt.to_period("M")) == {pd.Period("2020-03")}
