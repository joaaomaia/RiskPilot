import numpy as np
import pandas as pd

from riskpilot.evaluation.decile_plot import ks_table


def test_ks_table_is_deterministic_with_tied_scores() -> None:
    df = pd.DataFrame(
        {
            "score": np.repeat([0.1, 0.2, 0.3, 0.4], 6),
            "target": [0, 1, 0, 1, 0, 1] * 4,
        }
    )

    table_1, ks_1 = ks_table(df, score_col="score", target_col="target", n_bins=5)
    table_2, ks_2 = ks_table(df, score_col="score", target_col="target", n_bins=5)

    pd.testing.assert_frame_equal(table_1, table_2)
    assert ks_1 == ks_2


def test_ks_table_single_class_returns_nan_status() -> None:
    df = pd.DataFrame(
        {
            "score": np.linspace(0.01, 0.99, 12),
            "target": np.zeros(12, dtype=int),
        }
    )

    table, ks_value = ks_table(df, score_col="score", target_col="target", n_bins=20)

    assert np.isnan(ks_value)
    assert set(table["metric_status"]) == {"single_class"}
    assert table["KS"].isna().all()
    assert table["total"].sum() == len(df)
