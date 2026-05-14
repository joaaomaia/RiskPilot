"""binary_performance_evaluator.py
-----------------------------------
A self‑contained utility to evaluate the performance of an *already‑trained* binary
classifier across train / test / (optional) validation datasets.

Requirements
------------
Python >= 3.9
pandas, numpy, scikit‑learn, matplotlib, seaborn, plotly, kaleido

Quick Example
-------------
from riskpilot.evaluation import BinaryPerformanceEvaluator

evaluator = BinaryPerformanceEvaluator(
    model='modelo_treinado.pkl',
    df_train=df_train,
    df_test=df_test,
    df_val=df_val,                 # optional
    target_col='default_90d',
    id_cols=['contract_id'],
    date_col='snapshot_date',      # optional
    group_col='product_type',      # optional
    save_dir='figs',               # optional
    threshold=0.5                  # optional
)

evaluator.compute_metrics()
evaluator.plot_confusion(save=True)
evaluator.plot_calibration()
evaluator.plot_event_rate()
evaluator.plot_psi(reference_last_period=True)
evaluator.plot_ks()

stress = evaluator.run_stress_test(
    n_periods=36,
    freq="ME",
    scenario="stress",
)
print(stress["metrics"])
"""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
import shutil
import os
import uuid
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

import riskpilot

try:
    from optbinning import OptimalBinning
except ImportError:  # pragma: no cover - optional dependency
    OptimalBinning = None  # type: ignore[assignment]
    logging.warning("Optional dependency 'optbinning' is missing. " "Install with `pip install riskpilot[binning]`.")
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None  # type: ignore[assignment]
    logging.warning("Optional dependency 'shap' is missing. Install with `pip install riskpilot[viz]`.")

try:  # pragma: no cover - optional dependency
    from xgboost import XGBModel
except Exception:  # pragma: no cover - optional dependency
    XGBModel = object  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from lightgbm.sklearn import LGBMModel
except Exception:  # pragma: no cover - optional dependency
    LGBMModel = object  # type: ignore[assignment]

from ..synthetic import LookAhead
from .decile_plot import decile_analysis_plot, ks_table

sns.set(style="whitegrid")  # consistent style throughout


def _psi_single(p_base: np.ndarray, p_test: np.ndarray) -> float:
    """Compute PSI for two proportional histograms (no zeros allowed)."""
    mask = (p_base > 0) & (p_test > 0)
    if not mask.any():
        return 0.0
    delta = (p_test[mask] - p_base[mask]) * np.log(p_test[mask] / p_base[mask])
    return float(delta.sum())


def _rgba(color: str, alpha: float) -> str:
    """Return RGBA string with given opacity from ``rgb(r,g,b)`` input."""
    if color.startswith("rgb(") and color.endswith(")"):
        return color.replace("rgb", "rgba").replace(")", f",{alpha})")
    return color


def _style_plotly(fig: go.Figure, *, title: str | None = None) -> go.Figure:
    """Apply common styling to Plotly figures."""

    fig.update_layout(template="simple_white", title=title, legend=dict(x=1.02))
    fig.update_xaxes(showgrid=False, tickformat="%Y-%m")
    fig.update_yaxes(showgrid=False)
    return fig


# --- Helper utilities ---


def _compute_psi(counts_ref: np.ndarray, counts_cmp: np.ndarray, eps: float = 1e-6) -> float:
    """Compute PSI using histogram bin counts."""
    p_ref = (counts_ref + eps) / (counts_ref.sum() + eps * len(counts_ref))
    p_cmp = (counts_cmp + eps) / (counts_cmp.sum() + eps * len(counts_cmp))
    return _psi_single(p_ref, p_cmp)


def _filter_by_vintages(df: pd.DataFrame, date_col: str, vintages: list) -> pd.DataFrame:
    """Return rows matching the specified vintages.

    Vintages may be provided as ``YYYYMM`` integers/strings or any format
    recognised by ``pandas.to_datetime``. Integers in ``YYYYMM`` form are
    interpreted as year/month combinations.
    """

    vintages_series = pd.Series(vintages).astype(str)

    if vintages_series.str.fullmatch(r"\d{6}").all():
        parsed = pd.to_datetime(vintages_series, format="%Y%m", errors="coerce")
    else:
        parsed = pd.to_datetime(vintages_series, errors="coerce")
        if parsed.isna().any():
            parsed = parsed.fillna(pd.to_datetime(vintages_series, format="%Y%m", errors="coerce"))

    vintages_period = parsed.dropna().dt.to_period("M")
    periods = pd.to_datetime(df[date_col]).dt.to_period("M")
    return df.loc[periods.isin(vintages_period)]


def _has_two_classes(y_true: Any) -> bool:
    """Return whether a target vector has at least two non-null classes."""
    values = pd.Series(y_true).dropna().unique()
    return len(values) >= 2


def _safe_ks_stat(y_true: Any, score: Any) -> float:
    """Compute KS for event-oriented scores, returning NaN when undefined."""
    data = pd.DataFrame({"target": y_true, "score": score}).dropna()
    if data.empty or not _has_two_classes(data["target"]):
        return float("nan")
    fpr, tpr, _ = roc_curve(data["target"].to_numpy(), data["score"].to_numpy())
    return float(np.max(np.abs(tpr - fpr)))


class BinaryPerformanceEvaluator:
    """Evaluate binary classifier performance on multiple splits.

    Parameters
    ----------
    model : Union[str, Path, object]
        Path to `.joblib`/`.pkl` file **or** an in‑memory estimator that
        implements `predict_proba` (usual scikit‑learn API).
    df_train : pd.DataFrame
        Training set including predictors + target column.
    df_test : pd.DataFrame
        Test set including predictors + target column.
    df_val : Optional[pd.DataFrame], default=None
        Optional validation set.
    target_col : str
        Name of the binary target column (0 = negative, 1 = positive).
    id_cols : List[str]
        Columns that uniquely identify instances (e.g., contract or customer id).
    date_col : Optional[str], default=None
        Datetime column for temporal analyses.
    group_col : Optional[str], default=None
        Categorical column for group analyses. If ``homogeneous_group='auto'``,
        this column will be created with the labels generated by OptimalBinning.
        When ``homogeneous_group`` is ``None`` and ``group_col`` is provided,
        the column must already exist in all provided datasets.
    save_dir : Optional[str|Path], default=None
        If provided, figures are saved to this directory in PNG format.
    threshold : float, default 0.5
        Probability cutoff used to convert scores into class labels.
    homogeneous_group : str | int | pd.Series | np.ndarray | None, default None
        Strategy to create homogeneous groups. See :meth:`plot_group_radar`.
    stress_n_periods : int, default 12
        Default horizon for :meth:`run_stress_test`.
    stress_freq : str, default "M"
        Frequency of synthetic vintages.
    stress_scenario : {"base", "stress"}, default "stress"
        Intensity of random noise for generated data.
    stress_eval_funcs : sequence of str, optional
        Methods executed during :meth:`run_stress_test`.

    Deprecated Parameters
    ---------------------
    stress_periods : int, optional
        Use ``stress_n_periods`` instead.

    Notes
    -----
    * All DataFrames **must** contain `target_col`.
    * The class tries to automatically select predictor columns:
      - all columns present in *all* datasets
      - excluding id/date/target/group columns
    """

    ## ---------- constructor ----------
    def __init__(
        self,
        *,
        model: Union[str, Path, object],
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
        target_col: str,
        id_cols: List[str],
        date_col: Optional[str] = None,
        group_col: Optional[str] = None,
        save_dir: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
        homogeneous_group: Optional[Union[str, int, pd.Series, np.ndarray]] = None,
        synthetic_gen: LookAhead | None = None,
        stress_n_periods: int = 12,
        stress_freq: str = "M",
        stress_periods: int | None = None,
        stress_scenario: Literal["base", "stress"] = "stress",
        stress_eval_funcs: Sequence[str] = (
            "compute_metrics",
            "plot_psi",
            "plot_ks",
        ),
    ) -> None:
        if stress_periods is not None:
            warnings.warn(
                "'stress_periods' is deprecated, use 'stress_n_periods'",
                DeprecationWarning,
                stacklevel=2,
            )
            stress_n_periods = stress_periods

        self.model = self._load_model(model)
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.df_val = df_val.copy() if df_val is not None else None
        self.target_col = target_col
        self.id_cols = id_cols
        self.date_col = date_col
        self.group_col = group_col
        self.threshold = threshold
        self.homogeneous_group = homogeneous_group
        self.synthetic_gen = synthetic_gen
        self.stress_n_periods = stress_n_periods
        self.stress_freq = stress_freq
        self.stress_scenario = stress_scenario
        self.stress_eval_funcs = list(stress_eval_funcs)

        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "classes_") and 1 in getattr(self.model, "classes_"):
            self._pos_class_idx = list(self.model.classes_).index(1)
        else:
            self._pos_class_idx = 1

        self._validate_data()
        self._parse_date_col()
        self.predictor_cols = self._infer_predictors()
        self.model_feature_names = self._get_model_feature_names()
        self.model_n_features = self._get_model_n_features()
        self.predictor_cols = self._align_predictors_with_model(self.predictor_cols)
        self._validate_predictors()
        self.report: Dict[str, Dict[str, float]] = {}
        self.score_col_ = "y_pred_proba"
        self.label_col_ = "y_pred_label"
        self.group_col_ = self.group_col if self.group_col else "homogeneous_group"
        self.group_: Dict[str, pd.Series] | None = None
        self.binning_table_: Any | None = None
        self.group_palette_: Dict[Any, str] | None = None

        self._score_datasets()
        self._assign_groups()

        # --- SHAP caching ---
        self._shap_cache: dict[tuple[str, str], Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._disk_hits = 0
        self._fingerprint = hashlib.sha256(
            pickle.dumps(self.model, protocol=4)[:2_000_000]
            + b"|"
            + ",".join(self.predictor_cols).encode()
            + b"|"
            + riskpilot.__version__.encode()
        ).hexdigest()[:12]

    ## ---------- public API ----------

    def compute_metrics(
        self,
        *,
        by_date_col: bool = False,
        score_higher_is_positive: bool | None = True,
        auto_flip_score: bool = False,
    ) -> pd.DataFrame:
        """Calcula métricas globais ou por safra.

        Por padrao, ``self.score_col_`` deve conter ``P(target == 1)``. Se
        ``score_higher_is_positive`` for:
        - ``True``  → valores maiores indicam maior chance do evento (target==1)
        - ``False`` → valores maiores indicam menor chance do evento (target==1)
        - ``None``  → tenta detectar automaticamente via AUC‑ROC; se < 0.5
                       o vetor de probabilidades será invertido ``1‑p``.
        """

        # ------------- helper ------------------------------------- #
        def _row(df_slice: pd.DataFrame, split: str, period=None) -> dict:
            metric_data = df_slice[[self.target_col, self.score_col_]].dropna()
            y_true = metric_data[self.target_col].to_numpy()
            proba = metric_data[self.score_col_].to_numpy(dtype=float)
            n_obs = int(len(metric_data))
            n_events = int(np.sum(y_true == 1)) if n_obs else 0
            event_rate = float(n_events / n_obs) if n_obs else np.nan

            # --- orientação dos scores ---
            orientation_status = "event_score"
            auto_flip = auto_flip_score
            proba_evt = proba
            if score_higher_is_positive is True:
                proba_evt = proba
            elif score_higher_is_positive is False:
                proba_evt = 1.0 - proba
                orientation_status = "score_inverted_by_parameter"
                auto_flip = False
            else:  # auto‑detect
                auto_flip = True
                orientation_status = "auto_flip_requested"

            two_classes = n_obs > 0 and _has_two_classes(y_true)
            if auto_flip and two_classes:
                auc_raw = roc_auc_score(y_true, proba_evt)
                if auc_raw < 0.5:
                    proba_evt = 1.0 - proba_evt
                    orientation_status = "auto_flipped"

            # --- binário pela regra de corte ---
            y_pred = (proba_evt >= self.threshold).astype(int) if n_obs else np.array([], dtype=int)

            if n_obs == 0:
                metric_status = "no_valid_score"
                auc_roc = np.nan
                gini = np.nan
                ks = np.nan
                auc_pr = np.nan
                mcc = np.nan
                precision = np.nan
                recall = np.nan
                brier = np.nan
            elif not two_classes:
                metric_status = "single_class"
                auc_roc = np.nan
                gini = np.nan
                ks = np.nan
                auc_pr = np.nan
                mcc = matthews_corrcoef(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                brier = brier_score_loss(y_true, proba_evt)
            else:
                metric_status = "ok"
                auc_roc = roc_auc_score(y_true, proba_evt)
                gini = 2 * auc_roc - 1
                ks = _safe_ks_stat(y_true, proba_evt)
                auc_pr = average_precision_score(y_true, proba_evt)
                mcc = matthews_corrcoef(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                brier = brier_score_loss(y_true, proba_evt)

            return {
                "Split": split.capitalize(),
                **({"Period": period} if period is not None else {}),
                "MCC": mcc,
                "AUC_ROC": auc_roc,
                "Gini": gini,
                "KS": ks,
                "AUC_PR": auc_pr,
                "Precision": precision,
                "Recall": recall,
                "Brier": brier,
                "n_obs": n_obs,
                "n_events": n_events,
                "event_rate": event_rate,
                "metric_status": metric_status,
                "score_orientation": orientation_status,
            }

        # ------------- splits ------------------------------------- #
        splits = {"train": self.df_train, "test": self.df_test}
        if getattr(self, "df_val", None) is not None:
            splits["val"] = self.df_val

        date_col = self.date_col if by_date_col else None
        if by_date_col and not date_col:
            raise ValueError("`date_col` precisa ser definido para by_date_col=True.")

        records: list[dict] = []
        for split_name, df in splits.items():
            if by_date_col and date_col in df.columns:
                for period, df_period in df.sort_values(date_col).groupby(date_col, sort=True):
                    records.append(_row(df_period, split_name, period))
            else:
                records.append(_row(df, split_name))

        metrics_df = pd.DataFrame(records)

        # ---- índice ------------------------------------------------ #
        if by_date_col:
            metrics_df.set_index(["Split", "Period"], inplace=True)
        else:
            metrics_df.set_index("Split", inplace=True)

        return metrics_df

    def run_stress_test(
        self,
        *,
        n_periods: int | None = None,
        freq: str | None = None,
        scenario: str | None = None,
        start_vintage: str | pd.Timestamp | None = None,
        end_vintage: str | pd.Timestamp | None = None,
        align_with_history: bool = True,
        skip_train_overlap: bool = True,
        shocks: dict | None = None,
    ) -> Dict[str, Any]:
        """Generate synthetic vintages and evaluate stress metrics.

        Parameters
        ----------
        n_periods : int, optional
            Number of periods to generate. ``None`` → ``self.stress_n_periods``.
        freq : str, optional
            Frequency passed to :meth:`LookAhead.generate`. ``None`` → ``self.stress_freq``.
        scenario : str, optional
            Overrides :pyattr:`self.stress_scenario` when provided.
        start_vintage, end_vintage : str | pd.Timestamp, optional
            Boundaries for generation. With ``align_with_history=True`` and
            ``start_vintage`` omitted, the first vintage begins immediately
            after the maximum ``date_cols`` seen during :meth:`LookAhead.fit`.
        align_with_history : bool, default True
            Guarantee temporal continuity with the historical data.
        skip_train_overlap : bool, default True
            Prevent overlap with the training set when aligning with history.
        shocks : dict, optional
            Column‑level shocks forwarded to :meth:`LookAhead.generate`.

        Returns
        -------
        Dict[str, Any]
            Metrics and paths for the generated artifacts.
        """
        if self.synthetic_gen is None:
            raise ValueError("synthetic_gen is required for stress testing")

        logger = logging.getLogger("riskpilot")
        logger.info("Running stress test")

        n_periods = n_periods if n_periods is not None else self.stress_n_periods
        freq = freq if freq is not None else self.stress_freq

        df_synth = self.synthetic_gen.generate(
            n_periods=n_periods,
            freq=freq,
            scenario=scenario or self.stress_scenario,
            start_vintage=start_vintage,
            end_vintage=end_vintage,
            align_with_history=align_with_history,
            skip_train_overlap=skip_train_overlap,
            shocks=shocks,
        )

        run_id = uuid.uuid4().hex
        art_dir = Path("artifacts") / run_id / "stress"
        art_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = art_dir / "synthetic.parquet"
        df_synth.to_parquet(parquet_path)

        sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

        original_test = self.df_test
        self.df_test = df_synth
        self._score_datasets()
        self._assign_groups()

        results: Dict[str, Any] = {}
        for name in self.stress_eval_funcs:
            func = getattr(self, name)
            out = func()
            if isinstance(out, go.Figure):
                fig_path = art_dir / f"{name}.png"
                try:
                    out.write_image(fig_path)
                except ValueError as err:
                    if "kaleido" in str(err).lower():
                        raise RuntimeError(
                            "Plotly image export requires the 'kaleido' package. "
                            "Install it with 'pip install kaleido'."
                        ) from err
                    raise
                results[name] = str(fig_path)
            else:
                results[name] = out

        self.df_test = original_test
        self._score_datasets()
        self._assign_groups()

        self.report["stress"] = {
            "metrics": results.get("compute_metrics"),
            "figures": {k: v for k, v in results.items() if k != "compute_metrics"},
            "meta": {"file": str(parquet_path), "sha256": sha},
        }
        return self.report["stress"]

    def binning_table(self) -> Any | None:
        """Return the binning table used for homogeneous groups."""
        return self.binning_table_

    def plot_confusion(
        self,
        y_true: Sequence[int] | None = None,
        y_pred_proba: Sequence[float] | None = None,
        *,
        threshold: float | str = 0.5,
        splits: list[str] | None = None,
        normalize: bool = False,
        cmap: str = "Blues",
        figsize: tuple[int, int] = (5, 5),
        save: bool = False,
        display: bool = False,
        title_prefix: str = "Matriz de Confusão",
    ):
        """
        Desenha matrizes de confusão. Pode receber ``y_true`` e
        ``y_pred_proba`` diretamente ou utilizar os splits internos
        (train/test/val) do objeto.

        Parameters
        ----------
        y_true, y_pred_proba : array-like, optional
            Valores verdadeiros e probabilidades preditas para gerar uma única
            matriz. Se omitidos, são utilizados os dados internos divididos em
            ``splits``.
        threshold : float | {"ks","youden"}
            Cut-off fixo ou regra baseada no split Train.
        splits : list[str] | None
            Lista de splits desejados ("train","test","val"). None → todos disponíveis.
        normalize : bool
            Se True, escala de cor = %. Se False, = contagem absoluta.
        cmap : str
            Paleta seaborn/matplotlib.
        figsize : tuple
            Tamanho de CADA subplot.
        save : bool
            Salva PNG em `self.save_dir`.
        display : bool
            Exibe figura no notebook; se False evita duplicação.
        title_prefix : str
            Prefixo do título; o nome do split é acrescentado.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve

        # ---------------- helpers ---------------- #
        def _pos_idx() -> int:
            """Retorna o índice da classe positiva (1) no predict_proba."""
            return list(self.model.classes_).index(1)

        def _best_threshold(y, p, meth: str) -> float:
            fpr, tpr, thr = roc_curve(y, p)
            return float(thr[np.nanargmax(tpr - fpr)])  # KS = Youden

        # ---------------- provided arrays ---------------- #
        if y_true is not None and y_pred_proba is not None:
            y_true = np.asarray(y_true)
            y_pred_proba = np.asarray(y_pred_proba)
            y_pred = (y_pred_proba >= float(threshold)).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_pct = cm / cm.sum()
            heat = cm_pct if normalize else cm
            annot = np.array([[f"{cm[i, j]}\n{cm_pct[i, j]:.1%}" for j in range(2)] for i in range(2)])

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                heat,
                annot=annot,
                fmt="",
                cmap=cmap,
                cbar=False,
                vmin=0,
                vmax=heat.max() if heat.max() > 0 else 1,
                ax=ax,
            )
            ax.set_xlabel("Previsto")
            ax.set_ylabel("Real")
            ax.set_xticklabels(["Previsto 0", "Previsto 1"])
            ax.set_yticklabels(["Real 0", "Real 1"], rotation=0)
            ax.set_title(title_prefix)
            fig.tight_layout()

            if save and self.save_dir:
                fname = title_prefix.lower().replace(" ", "_") + ".png"
                fig.savefig(self.save_dir / fname, bbox_inches="tight")

            if display:
                plt.show()
            else:
                plt.close(fig)

            return fig

        # ---------------- splits approach ---------------- #
        available = {"train": self.df_train, "test": self.df_test}
        if getattr(self, "df_val", None) is not None:
            available["val"] = self.df_val

        if splits is None:
            splits = list(available.keys())
        else:
            splits = [s.lower() for s in splits]
            invalid = [s for s in splits if s not in available]
            if invalid:
                raise ValueError(f"Splits inválidos: {invalid}")

        pos_idx = _pos_idx()
        if isinstance(threshold, str):
            ref = available["train"]
            thr_y = ref[self.target_col].values
            thr_p = self.model.predict_proba(ref[self.predictor_cols])[:, pos_idx]
            threshold = _best_threshold(thr_y, thr_p, threshold.lower())

        n = len(splits)
        fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
        if n == 1:
            axes = [axes]

        for ax, split in zip(axes, splits):
            df_split = available[split]
            y_true_s = df_split[self.target_col].values
            y_proba_s = self.model.predict_proba(df_split[self.predictor_cols])[:, pos_idx]
            y_pred_s = (y_proba_s >= float(threshold)).astype(int)

            cm = confusion_matrix(y_true_s, y_pred_s, labels=[0, 1])
            cm_pct = cm / cm.sum()
            heat = cm_pct if normalize else cm
            annot = np.array([[f"{cm[i, j]}\n{cm_pct[i, j]:.1%}" for j in range(2)] for i in range(2)])

            sns.heatmap(
                heat,
                annot=annot,
                fmt="",
                cmap=cmap,
                cbar=False,
                vmin=0,
                vmax=heat.max() if heat.max() > 0 else 1,
                ax=ax,
            )
            ax.set_xlabel("Previsto")
            ax.set_ylabel("Real")
            ax.set_xticklabels(["Previsto 0", "Previsto 1"])
            ax.set_yticklabels(["Real 0", "Real 1"], rotation=0)
            ax.set_title(split.capitalize())

        fig.suptitle(title_prefix)
        fig.tight_layout()

        if save and self.save_dir:
            fname = title_prefix.lower().replace(" ", "_") + ".png"
            fig.savefig(self.save_dir / fname, bbox_inches="tight")

        if display:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_calibration(
        self,
        *,
        n_bins: int = 10,
        splits: list[str] | None = None,
        save: bool = False,
        display: bool = False,
        title: str = "Curva de Calibração",
    ) -> go.Figure:
        """
        Desenha curvas de calibração (Train/Test/Val) usando seaborn.

        Parameters
        ----------
        n_bins : int
            Número de bins para o `calibration_curve`.
        splits : list[str] | None
            Ex.: ["train","test"]. None → todos os disponíveis.
        save : bool
            Se True, salva PNG em `self.save_dir`.
        display : bool
            Se False fecha a figura (útil p/ evitar duplicação).
        title : str
            Título do gráfico.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss

        # ----------- obter splits disponíveis ----------
        available = {"train": self.df_train, "test": self.df_test}
        if getattr(self, "df_val", None) is not None:
            available["val"] = self.df_val

        if splits is None:
            splits = list(available.keys())
        else:
            splits = [s.lower() for s in splits]
            invalid = [s for s in splits if s not in available]
            if invalid:
                raise ValueError(f"Splits inválidos: {invalid}")

        # ----------- índice da classe positiva ----------
        pos_idx = list(self.model.classes_).index(1)

        infos = []
        for split in splits:
            df = available[split]
            y_true = df[self.target_col].values
            y_proba = self.model.predict_proba(df[self.predictor_cols])[:, pos_idx]

            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
            brier = brier_score_loss(y_true, y_proba)
            infos.append((split, prob_pred, prob_true, brier))

        n = len(infos)
        titles = [f"{s.capitalize()} (Brier = {b:.4f})" for s, _, _, b in infos]
        fig = make_subplots(rows=1, cols=n, subplot_titles=titles)

        for c, (split, prob_pred, prob_true, brier) in enumerate(infos, 1):
            fig.add_trace(
                go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Modelo"),
                row=1,
                col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="grey"),
                    showlegend=False,
                ),
                row=1,
                col=c,
            )

            fig.update_xaxes(
                title_text="Probabilidade Prevista",
                range=[0, 1],
                constrain="domain",  # usa 100 % do espaço horizontal
                showgrid=False,
                row=1,
                col=c,
            )
            fig.update_yaxes(
                title_text="Frequência Observada",
                range=[0, 1],
                constrain="domain",  # usa 100 % do espaço horizontal
                scaleanchor=f"x{c}",
                scaleratio=1,
                row=1,
                col=c,
                showgrid=False,
            )

        fig.update_layout(
            title=title,
            template="simple_white",
            showlegend=False,
            height=400,
            width=400 * n,
        )

        if save and self.save_dir:
            fname = title.lower().replace(" ", "_") + ".png"
            fig.write_image(self.save_dir / fname)

        if display:
            fig.show()

        return fig

    def plot_event_rate(
        self,
        *,
        splits: list[str] | None = None,
        separated: bool = False,
        save: bool = False,
        title: str = "",
        custom_colors: list[str] | None = None,
    ) -> tuple[go.Figure, go.Figure]:
        """
        1) Bad-Rate por GH
        2) Participação de GHs

        Parameters
        ----------
        splits : list[str] | None
            Subconjunto de ["train","test","val"]. None → todos disponíveis.
        separated : bool, default False
            • False → gráficos combinados (igual versão anterior).
            • True  → um subplot por split.
        save : bool
            Se True, salva PNG em `self.save_dir`.
        title : str
            Título base para o gráfico de Bad-Rate.
        custom_colors : list[str] | None
            Cores hex que sobrescrevem a paleta automática na ordem GH1→GHn.

        Returns
        -------
        (fig_rate, fig_share)
            Ambos são `plotly.graph_objects.Figure`.
        """
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # ----------- validações e splits -----------------
        if self.date_col is None:
            raise ValueError("`date_col` é obrigatório para plot_event_rate().")

        available = {"train": self.df_train, "test": self.df_test}
        if getattr(self, "df_val", None) is not None:
            available["val"] = self.df_val

        if splits is None:
            splits = list(available.keys())
        else:
            splits = [s.lower() for s in splits]
        invalid = [s for s in splits if s not in available]
        if invalid:
            raise ValueError(f"Splits inválidos: {invalid}")

        # ----------- paleta + ordem de GH ----------------
        df_full = pd.concat([available[s].assign(Split=s.capitalize()) for s in splits], axis=0)
        df_full[self.date_col] = pd.to_datetime(df_full[self.date_col])

        group_col = next(c for c in [self.group_col, self.group_col_] if c and c in df_full.columns)
        br_order = df_full.groupby(group_col)[self.target_col].mean().sort_values(ascending=False)
        gh_order = list(br_order.index)  # pior → melhor
        gh_label = {g: f"GH{i+1}" for i, g in enumerate(gh_order)}

        self._compute_group_palette()  # paleta automática
        colors = {
            **self.group_palette_,
            **{g: col for g, col in zip(gh_order, custom_colors or [])},
        }

        # ------------- helper p/ um split ----------------
        def _tables(df):
            pivot_br = (
                df.groupby([self.date_col, group_col])[self.target_col]
                .mean()
                .unstack(group_col)
                .reindex(columns=gh_order)
                .sort_index()
            )
            counts = (
                df.groupby([self.date_col, group_col])
                .size()
                .unstack(group_col)
                .reindex(columns=gh_order, fill_value=0)
                .sort_index()
            )
            pct = counts.div(counts.sum(axis=1), axis=0)
            return pivot_br, counts, pct

        # =================================================
        #            FIGURA 1 – Bad-Rate
        # =================================================
        n_cols = len(splits) if separated else 1
        fig_rate = make_subplots(
            rows=1,
            cols=n_cols,
            subplot_titles=[s.capitalize() for s in splits] if separated else None,
        )

        # =================================================
        #            FIGURA 2 – Participação
        # =================================================
        fig_share = make_subplots(
            rows=1,
            cols=n_cols,
            subplot_titles=[s.capitalize() for s in splits] if separated else None,
            specs=[[{"type": "bar"}] * n_cols],
        )

        # ----------------- loop splits -------------------
        for c, split in enumerate(splits, 1):
            df_s = available[split]
            df_s[self.date_col] = pd.to_datetime(df_s[self.date_col])
            pivot_br, counts, pct = _tables(df_s)

            # ----- Bad-Rate traces -----
            for g in gh_order:
                fig_rate.add_trace(
                    go.Scatter(
                        x=pivot_br.index,
                        y=pivot_br[g],
                        mode="lines+markers",
                        name=gh_label[g] if separated else f"{gh_label[g]} – {split}",
                        marker=dict(color=colors[g]),
                        line=dict(color=colors[g]),
                        showlegend=not separated,  # legenda global no modo combinado
                    ),
                    row=1,
                    col=c if separated else 1,
                )

            # ----- Participação traces -----
            for g in gh_order:
                fig_share.add_trace(
                    go.Bar(
                        x=pct.index,
                        y=pct[g],
                        name=gh_label[g] if separated else f"{gh_label[g]} – {split}",
                        marker=dict(color=colors[g]),
                        showlegend=False,  # legenda = fig_rate
                    ),
                    row=1,
                    col=c if separated else 1,
                )

        # --------------- layout global -------------------
        common_layout = dict(
            template="plotly_white",
            legend_title="Grupos Homogêneos",
        )
        fig_rate.update_layout(
            title=title or "Bad Rate por GH",
            hovermode="x unified",
            yaxis_tickformat=".0%",
            **common_layout,
        )
        fig_share.update_layout(
            title="Participação por GH" if title == "" else title + " – Participação",
            yaxis_tickformat=".0%",
            barmode="stack",
            **common_layout,
        )

        # ---- remove grade de TODOS os eixos --------------
        fig_rate.update_xaxes(showgrid=False)
        fig_rate.update_yaxes(showgrid=False, tickformat=".0%")
        fig_share.update_xaxes(showgrid=False)
        fig_share.update_yaxes(showgrid=False, tickformat=".0%")

        # --------------- salvar opcional -----------------
        if save and self.save_dir:
            fig_rate.write_image(self.save_dir / "event_rate.png", scale=2)
            fig_share.write_image(self.save_dir / "event_share.png", scale=2)

        return fig_rate, fig_share

    def plot_psi(
        self,
        *,
        reference_df: Optional[pd.DataFrame] = None,
        bin_strategy: Optional[Dict[str, Any]] = None,
        min_obs: int = 100,
        eps: float = 1e-9,
        reference_last_period: bool = False,
        save: bool = False,
        title: str = "",
        feature: Optional[str] = None,
        smart_view: bool = False,
        psi_threshold: float = 0.10,  # usado quando smart_view=True
    ) -> Union[
        tuple[go.Figure, pd.DataFrame],  # quando feature != None
        tuple[Dict[str, go.Figure], pd.DataFrame],  # quando feature == None
    ]:
        """
        PSI por variável ao longo do tempo.

        Parameters
        ----------
        feature : str | None
            • None  → gera um gráfico para **cada** variável.
            • "var" → gera gráfico só para essa variável.
        smart_view : bool, default False
            Se True, inclui somente variáveis cujo PSI >= `psi_threshold`
            em pelo menos um split / período.
        """

        import warnings

        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go

        if self.date_col is None:
            raise ValueError("`date_col` is required for plot_psi().")

        reference_df = reference_df if reference_df is not None else self.df_train
        bin_strategy = bin_strategy or {"type": "quantile", "n_bins": 10}
        reference_type = "train_global"
        reference_period = pd.NaT
        if reference_last_period:
            if self.date_col not in reference_df.columns:
                raise ValueError("`reference_last_period=True` requires `date_col` in reference_df.")
            ref_periods = pd.to_datetime(reference_df[self.date_col]).dt.to_period("M")
            valid_periods = ref_periods.dropna()
            if valid_periods.empty:
                raise ValueError("`reference_last_period=True` requires at least one valid reference period.")
            last_period = valid_periods.max()
            reference_df = reference_df.loc[ref_periods == last_period].copy()
            reference_type = "last_period"
            reference_period = last_period.to_timestamp()

        # ------------- helper p/ edges -----------------
        def _get_edges(series: pd.Series) -> np.ndarray:
            ser = pd.to_numeric(series, errors="coerce").dropna()
            if ser.empty:
                return np.array([])
            if bin_strategy.get("type") == "quantile":
                try:
                    _, edges = pd.qcut(
                        ser,
                        q=bin_strategy.get("n_bins", 10),
                        retbins=True,
                        duplicates="drop",
                    )
                except ValueError:
                    edges = np.linspace(
                        ser.min(),
                        ser.max(),
                        bin_strategy.get("n_bins", 10) + 1,
                    )
            else:
                edges = np.linspace(
                    ser.min(),
                    ser.max(),
                    bin_strategy.get("n_bins", 10) + 1,
                )
            edges[0] = min(edges[0], ser.min())
            edges[-1] = max(edges[-1], ser.max())
            return edges

        # ------------- splits --------------------------
        splits = [
            ("Train", self.df_train),
            ("Test", self.df_test),
            *([("Val", self.df_val)] if self.df_val is not None else []),
        ]

        psi_records: List[Dict[str, Any]] = []

        # --------------------------------------------
        # lista inicial de variáveis
        variables = [feature] if feature else self._psi_variables()

        # 🔒 remove colunas internas de score/label
        forbidden = {self.score_col_, self.label_col_}
        variables = [v for v in variables if v not in forbidden]

        # se o usuário pediu exatamente uma variável proibida, avise
        if feature in forbidden:
            raise ValueError(f"'{feature}' é coluna interna (score/label) e não deve ser usada para PSI.")
        # --------------------------------------------

        # -------- global_edges (referência train) ------
        global_edges: Dict[str, np.ndarray] = {}
        for var in variables:
            edges = _get_edges(reference_df[var])
            if edges.size > 0:
                global_edges[var] = edges

        for split_name, df in splits:
            periods = pd.to_datetime(df[self.date_col]).dt.to_period("M").sort_values().unique()
            for var in variables:
                edges = global_edges.get(var)
                if edges is None:
                    continue
                ref_series = pd.to_numeric(reference_df[var], errors="coerce").dropna()
                counts_ref = np.histogram(ref_series, bins=edges)[0].astype(float) + eps
                p_ref = counts_ref / counts_ref.sum()

                for period in periods:
                    subset = df[pd.to_datetime(df[self.date_col]).dt.to_period("M") == period]
                    if len(subset) < min_obs:
                        continue
                    ser = pd.to_numeric(subset[var], errors="coerce").dropna()
                    if ser.empty:
                        continue

                    edges_adj = edges.copy()
                    if ser.min() < edges_adj[0]:
                        edges_adj[0] = ser.min()
                    if ser.max() > edges_adj[-1]:
                        edges_adj[-1] = ser.max()

                    counts_test = np.histogram(ser, bins=edges_adj)[0].astype(float) + eps
                    p_test = counts_test / counts_test.sum()
                    psi_val = _psi_single(p_ref, p_test)

                    psi_records.append(
                        {
                            "Variable": var,
                            "Period": period.to_timestamp(),
                            "PSI": psi_val,
                            "Split": split_name,
                            "reference_type": reference_type,
                            "reference_period": reference_period,
                        }
                    )

        psi_df = pd.DataFrame(psi_records)
        if psi_df.empty:
            warnings.warn("PSI could not be computed (insufficient data).")
            return go.Figure(), psi_df

        # -------- smart_view --------------------------
        if smart_view and feature is None:
            keep_vars = psi_df.groupby("Variable")["PSI"].max().loc[lambda s: s >= psi_threshold].index
            psi_df = psi_df[psi_df["Variable"].isin(keep_vars)]

        # ------------- retorno: 1 figura por variável --------------
        figures: Dict[str, go.Figure] = {}
        for var, grp_var in psi_df.groupby("Variable"):
            fig = go.Figure()
            for split, grp_split in grp_var.groupby("Split"):
                fig.add_trace(
                    go.Scatter(
                        x=grp_split["Period"],
                        y=grp_split["PSI"],
                        mode="lines+markers",
                        name=split,
                    )
                )
            # linhas-limite discretas
            for yline, color in [(0.10, "gray"), (0.25, "gray")]:
                fig.add_hline(
                    y=yline,
                    line=dict(color=color, dash="dash", width=1),
                    opacity=0.4,  # 👈 fora do dict line
                    annotation_text=f"{yline:.2f}",
                    annotation_position="top right",
                    annotation_font_color=color,
                )

            fig.update_layout(
                title=(title or f"PSI ao longo do tempo – {var}"),
                xaxis_title="Safra",
                yaxis_title="PSI",
                template="plotly_white",
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                yaxis_tickformat=".2f",
            )
            figures[var] = fig

            if save and self.save_dir:
                safe_var = var.replace("/", "_")
                fig.write_image(self.save_dir / f"psi_{safe_var}.png")

        # ------------- casos de retorno -----------------
        if feature:
            return figures[feature], psi_df[psi_df["Variable"] == feature]

        # figura única com todas as variáveis
        global_fig = go.Figure()
        for (var, split), grp in psi_df.groupby(["Variable", "Split"]):
            global_fig.add_trace(
                go.Scatter(
                    x=grp["Period"],
                    y=grp["PSI"],
                    mode="lines+markers",
                    name=f"{var} ({split})",
                )
            )
        for yline, color in [(0.10, "gray"), (0.25, "gray")]:
            global_fig.add_hline(
                y=yline,
                line=dict(color=color, dash="dash", width=1),
                opacity=0.4,
                annotation_text=f"{yline:.2f}",
                annotation_position="top right",
                annotation_font_color=color,
            )
        global_fig.update_layout(
            title=title or "PSI ao longo do tempo",
            xaxis_title="Safra",
            yaxis_title="PSI",
            template="plotly_white",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_tickformat=".2f",
        )

        if save and self.save_dir:
            global_fig.write_image(self.save_dir / "psi_all.png")

        return global_fig, psi_df

    def plot_histograms(  # noqa: C901
        self,
        feature: str | Sequence[str],
        *,
        # ------------------------------------------------------------------
        # What to plot
        reference: dict[str, list[int]] | None = None,
        compare: dict[str, list[int]] | None = None,
        # Histogram rules
        bins: int | str = "auto",
        stat: Literal["count", "density", "probability"] = "density",
        normalize_to_reference: bool = True,
        highlight_drift: bool = True,
        drift_metric: Literal["psi", "ks"] | None = "psi",
        # Visual elements
        kde: bool = True,
        bars: bool = True,
        kde_fill_alpha: float | None = None,  # 0‒1 → área sombreada
        alpha: float = 0.40,  # opacidade das barras  # TODO(use DEFAULT_ALPHA)
        log_scale: bool = False,
        # Styling
        figsize: tuple[int, int] = (6, 4),
        cmap_reference: str = "#b4b5b6",  # cinza  # TODO(use COLOR_PRIMARY)
        cmap_compare: str = "#f88825",  # laranja  # TODO(use COLOR_PRIMARY)
        show_metric: bool = True,  # ⬅️  nova anotação discreta
        show_legend: bool = True,  # ⬅️  liga/desliga legenda
        # Retro‑compatibilidade
        show_table: bool | None = None,  # obsoleto – mantém assinatura
        # Output behaviour
        save: str | Path | None = None,
        show: bool = True,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
    ) -> None:
        """Visualize distribution drift for one or more *numeric* features.

        Novos parâmetros
        ----------------
        show_metric : bool, default True
            Exibe a métrica (*PSI* ou *KS*) como texto discreto no canto
            superior‑direito do subplot.
        show_legend : bool, default True
            Exibe a legenda "Reference" / "Compare".

        O parâmetro *show_table* passa a ser **obsoleto**; se for ``True`` para
        fins de retro‑compatibilidade, a anotação textual também será mostrada.
        """
        # ------------------------------------------------------------------ #
        # 0 ─ Imports                                                       #
        # ------------------------------------------------------------------ #
        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_theme(style="white", rc={"axes.grid": False})
        else:
            from plotly.subplots import make_subplots

        import warnings
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from scipy import stats

        # ------------------------------------------------------------------ #
        # 1 ─ Validation                                                    #
        # ------------------------------------------------------------------ #
        features = [feature] if isinstance(feature, str) else list(feature)
        missing = [f for f in features if f not in self.df_train.columns]
        if missing:
            raise ValueError(f"Feature(s) not found: {missing}")
        if self.date_col is None and ((reference and any(reference.values())) or (compare and any(compare.values()))):
            raise ValueError("`date_col` is required for vintage filtering.")

        # ------------------------------------------------------------------ #
        # 2 ─ Data extraction                                               #
        # ------------------------------------------------------------------ #
        all_splits = {"train": self.df_train, "test": self.df_test}
        if getattr(self, "df_val", None) is not None:
            all_splits["val"] = self.df_val

        def _prep(mapping, default_split):
            if mapping is None:
                return {default_split: None}
            if isinstance(mapping, (list, tuple)):
                return {default_split: list(mapping)}
            return {k.lower(): (list(v) if v is not None else None) for k, v in mapping.items()}

        reference = _prep(reference, "train")
        default_cmp = "test" if "test" in all_splits else "val"
        compare = _prep(compare, default_cmp)

        def _collect(mapping):
            frames = []
            for split, vint in mapping.items():
                df = all_splits.get(split)
                if df is None:
                    continue
                if vint:
                    df = _filter_by_vintages(df, self.date_col, vint)
                frames.append(df)
            return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()

        df_ref, df_cmp = _collect(reference), _collect(compare)
        if df_ref.empty or df_cmp.empty:
            warnings.warn("Histogram data is empty for the specified vintages.")
            return None

        # ------------------------------------------------------------------ #
        # 3 ─ Layout                                                        #
        # ------------------------------------------------------------------ #
        n = len(features)
        ncols = 1 if n == 1 else 2
        nrows = math.ceil(n / ncols)

        if backend == "plotly":
            fig = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=features,
                vertical_spacing=0.14,
                horizontal_spacing=0.08,
            )
        else:
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(figsize[0] * ncols, figsize[1] * nrows),
                squeeze=False,
            )

        # ------------------------------------------------------------------ #
        # 4 ─ Annotation helper                                             #
        # ------------------------------------------------------------------ #
        def _annotate_metric(ax_or_fig, row_idx, col_idx, metric_str):
            if not show_metric or not metric_str:
                return
            if backend == "plotly":
                fig.add_annotation(
                    text=metric_str,
                    xref=f"x{row_idx*ncols+col_idx}" if n > 1 else "x",
                    yref=f"y{row_idx*ncols+col_idx}" if n > 1 else "y",
                    x=0.98,
                    y=0.98,
                    showarrow=False,
                    align="right",
                    font=dict(size=10, color="black"),
                )
            else:
                ax_or_fig.text(
                    0.98,
                    0.98,
                    metric_str,
                    transform=ax_or_fig.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                )

        # ------------------------------------------------------------------ #
        # 5 ─ Main loop                                                     #
        # ------------------------------------------------------------------ #
        for idx, feat in enumerate(features):
            row, col = (idx // ncols) + 1, (idx % ncols) + 1
            ax = None if backend == "plotly" else axes[row - 1, col - 1]

            ref_series = pd.to_numeric(df_ref[feat], errors="coerce").dropna()
            cmp_series = pd.to_numeric(df_cmp[feat], errors="coerce").dropna()
            if ref_series.empty or cmp_series.empty:
                warnings.warn(f"No data to plot for feature '{feat}'.")
                if backend == "matplotlib":
                    ax.set_visible(False)
                continue

            # Bin edges
            data_ref = ref_series.values
            if bins == "auto":
                edges = np.histogram_bin_edges(data_ref, bins="fd")
                if len(edges) - 1 > 50:
                    edges = np.linspace(data_ref.min(), data_ref.max(), 51)
            else:
                edges = np.histogram_bin_edges(data_ref, bins=bins)
            counts_ref, _ = np.histogram(data_ref, bins=edges)
            counts_cmp, _ = np.histogram(cmp_series.values, bins=edges)

            # Stats conversion
            scale = 1.0
            if stat == "count":
                hist_ref, hist_cmp = counts_ref, counts_cmp
                if normalize_to_reference and hist_cmp.sum() > 0:
                    scale = hist_ref.sum() / hist_cmp.sum()
                    hist_cmp = counts_cmp * scale
            else:
                widths = np.diff(edges)
                p_ref = counts_ref / counts_ref.sum() if counts_ref.sum() else counts_ref
                p_cmp = counts_cmp / counts_cmp.sum() if counts_cmp.sum() else counts_cmp
                if stat == "density":
                    hist_ref, hist_cmp = p_ref / widths, p_cmp / widths
                else:
                    hist_ref, hist_cmp = p_ref, p_cmp

            centres = edges[:-1] + np.diff(edges) / 2
            bar_kwargs = {"width": np.diff(edges)}
            key_opacity = "opacity" if backend == "plotly" else "alpha"
            bar_kwargs[key_opacity] = alpha

            # ----------------------- Bars -----------------------
            if bars:
                if backend == "plotly":
                    fig.add_bar(
                        x=centres,
                        y=hist_ref,
                        name="Reference" if idx == 0 else "Reference",
                        marker=dict(color=cmap_reference),
                        row=row,
                        col=col,
                        **bar_kwargs,
                    )
                else:
                    ax.bar(
                        centres,
                        hist_ref,
                        label="Reference",
                        color=cmap_reference,
                        align="center",
                        **bar_kwargs,
                    )

            # Highlight drift (only when bars & stat='count')
            edge_colours = None
            if bars and highlight_drift and stat == "count":
                scaled_cmp = counts_cmp * scale
                diff = np.abs(scaled_cmp - counts_ref)
                thresh = 2 * np.sqrt(scaled_cmp + counts_ref)
                edge_colours = ["red" if d > t else None for d, t in zip(diff, thresh)]

            if bars:
                if backend == "plotly":
                    fig.add_bar(
                        x=centres,
                        y=hist_cmp,
                        name="Compare" if idx == 0 else "Compare",
                        marker=dict(
                            color=cmap_compare,
                            line=dict(
                                color=edge_colours if edge_colours else cmap_compare,
                                width=1.8 if edge_colours else 0,
                            ),
                        ),
                        row=row,
                        col=col,
                        **bar_kwargs,
                    )
                else:
                    ax.bar(
                        centres,
                        hist_cmp,
                        label="Compare",
                        color=cmap_compare,
                        align="center",
                        edgecolor=edge_colours,
                        linewidth=1.5 if edge_colours else 0,
                        **bar_kwargs,
                    )

            # ----------------------- KDE -----------------------
            def _kde(series, color, label):
                if not kde:
                    return
                kp = dict(
                    ax=ax if backend == "matplotlib" else None,
                    color=color,
                    linewidth=1.4,
                    label=label,
                )
                if kde_fill_alpha is not None and kde_fill_alpha > 0:
                    kp.update(fill=True, alpha=kde_fill_alpha)
                if backend == "matplotlib":
                    sns.kdeplot(series, **kp)

            if backend == "matplotlib":
                _kde(ref_series, cmap_reference, "Reference")
                _kde(cmp_series, cmap_compare, "Compare")

            # ----------------------- Metrics -------------------
            metric_str = ""
            if drift_metric == "psi":
                metric_str = f"PSI={_compute_psi(counts_ref, counts_cmp):.3f}"
            elif drift_metric == "ks":
                ks_stat, p_val = stats.ks_2samp(ref_series, cmp_series)
                metric_str = f"KS={ks_stat:.3f} (p={p_val:.3f})"

            # ----------------------- Labels --------------------
            if backend == "plotly":
                fig.update_xaxes(title=feat, type="log" if log_scale else "linear", row=row, col=col)
                fig.update_yaxes(title=stat.capitalize(), row=row, col=col)
            else:
                ax.set_title(feat)
                ax.set_ylabel(stat.capitalize())
                if log_scale:
                    ax.set_xscale("log")

            # annotation – replaces old table
            _annotate_metric(ax if backend == "matplotlib" else fig, row, col, metric_str)

            # legend
            if backend == "matplotlib" and show_legend:
                ax.legend(loc="best")

        # ------------------------------------------------------------------ #
        # 6 ─ Output                                                        #
        # ------------------------------------------------------------------ #
        if backend == "plotly":
            fig.update_layout(
                template="plotly_white",
                height=figsize[1] * nrows * 110,
                width=figsize[0] * ncols * 110,
                barmode="overlay",
                showlegend=show_legend,
            )
            if save:
                Path(save).with_suffix(".html").write_text(fig.to_html())
            if show:
                fig.show()
        else:
            for j in range(n, nrows * ncols):
                axes.flat[j].set_visible(False)
            fig.tight_layout()
            if save:
                plt.savefig(Path(save))
            if show:
                plt.show()
            else:
                plt.close(fig)
        return None

    def plot_ks(
        self,
        *,
        limite: float = 0.20,
        save: bool = False,
        title: str = "",
        return_table: bool = False,
    ) -> go.Figure | tuple[go.Figure, pd.DataFrame] | None:
        """
        KS por safra para Train/Test/(Val) com cores fixas e hover customizado.
        """

        import plotly.graph_objects as go

        if self.date_col is None:
            raise ValueError("`date_col` é obrigatório para plot_ks().")
        self._validate_predictors()

        # ------- prepara splits disponíveis --------
        dfs = [
            ("Train", self.df_train, "#7f7f7f"),  # TODO(use COLOR_PRIMARY)
            ("Test", self.df_test, "#1f77b4"),  # TODO(use COLOR_PRIMARY)
            *(
                [("Val", self.df_val, "#d62728")] if getattr(self, "df_val", None) is not None else []
            ),  # TODO(use COLOR_PRIMARY)
        ]

        ks_records = []
        for split_name, df, _ in dfs:
            df = df.copy()
            df["Safra"] = pd.to_datetime(df[self.date_col]).dt.to_period("M")
            for safra, grp in df.groupby("Safra", sort=True):
                y_true = grp[self.target_col].values
                score = grp[self.score_col_].values
                ks = _safe_ks_stat(y_true, score)
                metric_status = "ok" if _has_two_classes(y_true) else "single_class"
                ks_records.append(
                    {
                        "Split": split_name,
                        "Safra": safra.to_timestamp(),
                        "Safra_fmt": safra.strftime("%Y%m"),
                        "KS": ks,
                        "Volume": len(grp),
                        "n_events": int(np.sum(y_true == 1)),
                        "n_nonevents": int(np.sum(y_true == 0)),
                        "metric_status": metric_status,
                    }
                )

        ks_df = pd.DataFrame(ks_records)
        self.ks_table_ = ks_df
        if ks_df.empty:
            warnings.warn("KS não pode ser computado (dados insuficientes).")
            empty_fig = go.Figure()
            return (empty_fig, ks_df) if return_table else None

        # ------- plotly --------
        fig = go.Figure()
        for split, color in [
            ("Train", "#7f7f7f"),  # TODO(use COLOR_PRIMARY)
            ("Test", "#d62728"),  # TODO(use COLOR_PRIMARY)
            ("Val", "#1f77b4"),  # TODO(use COLOR_PRIMARY)
        ]:
            grp = ks_df[ks_df["Split"] == split]
            if grp.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=grp["Safra"],
                    y=grp["KS"],
                    mode="lines+markers",
                    name=split,
                    line=dict(color=color),
                    marker=dict(color=color),
                    customdata=np.stack([grp["Safra_fmt"], grp["Volume"]], axis=-1),
                    hovertemplate=("Safra: %{customdata[0]}<br>" "KS: %{y:.4f}<br>" "Volume: %{customdata[1]:,}"),
                )
            )

        # linha de referência 0.20
        fig.add_hline(
            y=limite,
            line=dict(color="gray", dash="dash", width=1),
            annotation_text=f"{limite}",
            annotation_position="top right",
        )

        y_max = ks_df["KS"].dropna().max()
        y_max = 0.05 if pd.isna(y_max) else max(0.05, y_max * 1.1)

        fig.update_layout(
            title=title or "KS por Safra",
            xaxis_title="Safra",
            yaxis_title="KS",
            yaxis_tickformat=".2f",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, range=[0, y_max]),
            template="plotly_white",
        )

        if save and self.save_dir:
            fig.write_image(str(self.save_dir / "ks_evolution.png"))

        if return_table:
            return fig, ks_df
        return fig

    # def plot_group_radar(
    #     self,
    #     features: List[str] | None = None,
    #     *,
    #     scaler: Literal["zscore", "minmax"] = "zscore",
    #     save: bool = False,
    #     title: str = "",
    # ) -> go.Figure:
    #     """Radar chart das médias por grupo, com grupos de maior risco por cima."""
    #     if self.group_ is None:
    #         raise ValueError("Homogeneous groups were not computed.")

    #     self._compute_group_palette()
    #     df = self.data_.copy()

    #     # ---------- seleção de features numéricas ----------
    #     if features is None:
    #         numeric_predictors = [
    #             c for c in self.predictor_cols if pd.api.types.is_numeric_dtype(df[c])
    #         ]
    #         features = numeric_predictors
    #     if not features:
    #         raise ValueError("No numeric features available for radar plot.")

    #     # ---------- escalonamento ----------
    #     if scaler == "zscore":
    #         scaled = df[features].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
    #     else:
    #         scaled = df[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    #     # ---------- média por grupo ----------
    #     mean_by_group = scaled.groupby(df[self.group_col_])[features].mean()

    #     # ---------- ordena grupos por event-rate ----------
    #     event_rate = (
    #         df.groupby(self.group_col_)[self.target_col].mean().sort_values()
    #     )  # ascendente
    #     ordered_groups = list(event_rate.index)  # do menor (azul) ao maior (vermelho)

    #     # ---------- gráfico ----------
    #     fig = go.Figure()
    #     for group_id in ordered_groups:
    #         row = mean_by_group.loc[group_id]
    #         theta = features + [features[0]]             # fecha polígono
    #         r = row.tolist() + [row.iloc[0]]

    #         fig.add_trace(
    #             go.Scatterpolar(
    #                 r=r,
    #                 theta=theta,
    #                 fill="toself",
    #                 name=f"Group {group_id}",
    #                 line=dict(color=_rgba(self.group_palette_.get(group_id), 0.8)),
    #                 fillcolor=_rgba(self.group_palette_.get(group_id), 0.3),
    #             )
    #         )

    #     fig.update_layout(
    #         template="plotly_white",
    #         title=title or "Group Radar",
    #         polar=dict(radialaxis=dict(visible=False)),
    #         showlegend=True,
    #     )

    #     if save and self.save_dir:
    #         fig.write_image(str(self.save_dir / "group_radar.png"))

    #     return fig

    def plot_group_radar(
        self,
        features: list[str] | None = None,
        *,
        groups: list[int] | None = None,
        separated: bool = False,
        splits: list[str] | None = None,
        scaler: Literal["zscore", "minmax"] = "zscore",
        animation: bool = False,
        save: bool = False,
        title: str = "",
    ) -> Union[go.Figure, Dict[int, go.Figure]]:
        """Radar chart das médias das *features* por Grupo Homogêneo (GH).

        Parameters
        ----------
        features : list[str] | None, optional
            Lista de features numéricas para o radar. Se ``None`` (default),
            todas as colunas preditoras numéricas são utilizadas.

        Novidades
        ---------
        • Escala do eixo *radial* agora é **fixa** para todos os gráficos,
        evitando “encolher/esticar” quando GHs mudam.
        • Todos os gráficos usam **height = 800** e **width = 1200**.
        • Hover mostra Split e Volume; eixo radial oculto.
        """

        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # ------------------- validações -------------------
        if self.group_ is None:
            raise ValueError("Homogeneous groups were not computed.")
        if animation and self.date_col is None:
            raise ValueError("`date_col` é necessário para animation=True.")

        # ------------ mapeia GH ↔ número ordenado ----------
        br = self.data_.groupby(self.group_col_)[self.target_col].mean().sort_values(ascending=False)
        gh_order = list(br.index)  # pior → melhor
        gh_to_num = {g: i + 1 for i, g in enumerate(gh_order)}
        num_to_gh = {v: k for k, v in gh_to_num.items()}

        if groups is None:
            groups_num = list(range(1, len(gh_order) + 1))
        else:
            groups_num = [g for g in groups if g in num_to_gh]
        groups_int = [num_to_gh[g] for g in groups_num]

        # -------------------- splits ----------------------
        all_splits = {"train": self.df_train, "test": self.df_test}
        if self.df_val is not None:
            all_splits["val"] = self.df_val
        if splits is None:
            splits = list(all_splits.keys())
        splits = [s.lower() for s in splits if s.lower() in all_splits]

        if features is None:
            cols = self.predictor_cols
        else:
            cols = [c for c in features if c in self.predictor_cols]

        feats = [c for c in cols if pd.api.types.is_numeric_dtype(self.data_[c])]
        if not feats:
            raise ValueError("No numeric features available for radar plot.")

        # ------------- escalonamento helper ---------------
        def _scale(df):
            if scaler == "zscore":
                return df[feats].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
            return df[feats].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # ---------------- escala global -------------------
        scaled_global = _scale(self.data_)
        r_min, r_max = (
            scaled_global[feats].min().min(),
            scaled_global[feats].max().max(),
        )

        # ---------------- volume helper -------------------
        def _vol(df, g_int):
            return int(df[df[self.group_col_] == g_int].shape[0])

        # ==================================================
        #                 animação mensal
        # ==================================================
        if animation:
            periods = pd.to_datetime(self.data_[self.date_col]).dt.to_period("M").sort_values().unique()
            if not len(periods):
                raise ValueError("Nenhum período para animação.")

            n_gh = len(groups_int)
            fig = make_subplots(
                rows=1,
                cols=n_gh,
                specs=[[{"type": "polar"}] * n_gh],
                subplot_titles=[f"GH{n}" for n in groups_num],
            )

            def _frame(per):
                traces = []
                for idx, (g_int, g_num) in enumerate(zip(groups_int, groups_num), 1):
                    theta = feats + [feats[0]]
                    for split in splits:
                        df_s = all_splits[split]
                        mask = pd.to_datetime(df_s[self.date_col]).dt.to_period("M") == per
                        df_s = df_s[mask]
                        if df_s.empty or g_int not in df_s[self.group_col_].values:
                            continue
                        mean_row = _scale(df_s).groupby(df_s[self.group_col_])[feats].mean().loc[g_int]
                        r_vals = mean_row.tolist() + [mean_row[feats[0]]]
                        traces.append(
                            go.Scatterpolar(
                                r=r_vals,
                                theta=theta,
                                subplot=f"polar{idx}",
                                name=f"GH{g_num} ({split})",
                                line=dict(color=_rgba(self.group_palette_[g_int], 0.8)),
                                fill="toself",
                                fillcolor=_rgba(self.group_palette_[g_int], 0.25),
                                customdata=[_vol(df_s, g_int)] * len(r_vals),
                                hovertemplate=(
                                    f"Safra: {per.strftime('%Y%m')}<br>"
                                    f"Split: {split.capitalize()}<br>"
                                    "Volume: %{customdata:,}<extra></extra>"
                                ),
                                showlegend=False,
                            )
                        )
                return traces

            frames = [go.Frame(data=_frame(per), name=per.strftime("%Y-%m")) for per in periods]
            fig.add_traces(frames[0].data)
            fig.frames = frames

            # fixa escala
            fig.update_layout(
                **{f"polar{i}": dict(radialaxis=dict(visible=False, range=[r_min, r_max])) for i in range(1, n_gh + 1)},
                template="plotly_white",
                title=title or "Evolução mensal",
                height=800,
                width=1200,
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        x=0.1,
                        y=1.1,
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 600, "redraw": True},
                                        "fromcurrent": True,
                                    },
                                ],
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {"frame": {"duration": 0}, "mode": "immediate"},
                                ],
                            ),
                        ],
                    )
                ],
            )
            if save and self.save_dir:
                fig.write_html(self.save_dir / "group_radar_animation.html")
            return fig

        # ==================================================
        #              estáticos (combined / sep)
        # ==================================================
        mean_split = {s: _scale(df).groupby(df[self.group_col_])[feats].mean() for s, df in all_splits.items()}

        def _polar_cfg(n):
            return {f"polar{i}": dict(radialaxis=dict(visible=False, range=[r_min, r_max])) for i in range(1, n + 1)}

        # -------- combinado ----------
        if not separated:
            n_cols = len(splits)
            fig = make_subplots(
                rows=1,
                cols=n_cols,
                specs=[[{"type": "polar"}] * n_cols],
                subplot_titles=[s.capitalize() for s in splits],
            )
            for c, split in enumerate(splits, 1):
                for g_int, g_num in zip(reversed(groups_int), reversed(groups_num)):
                    if g_int not in mean_split[split].index:
                        continue
                    theta = feats + [feats[0]]
                    r_vals = mean_split[split].loc[g_int].tolist() + [mean_split[split].loc[g_int, feats[0]]]
                    fig.add_trace(
                        go.Scatterpolar(
                            r=r_vals,
                            theta=theta,
                            subplot=f"polar{c}",
                            name=f"GH{g_num}",
                            line=dict(color=_rgba(self.group_palette_[g_int], 0.8)),
                            fill="toself",
                            fillcolor=_rgba(self.group_palette_[g_int], 0.25),
                            customdata=[_vol(all_splits[split], g_int)] * len(r_vals),
                            hovertemplate=(
                                f"GH{g_num}<br>Split: {split.capitalize()}<br>" "Volume: %{customdata:,}<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=c,
                    )
            fig.update_layout(
                **_polar_cfg(n_cols),
                template="plotly_white",
                title=title or "Radar – GHs combinados",
                height=800,
                width=1200,
                legend_title="Grupos Homogêneos",
            )
            if save and self.save_dir:
                fig.write_image(self.save_dir / "group_radar_combined.png")
            return fig

        # -------- separado ----------
        figs = {}
        for g_int, g_num in zip(groups_int, groups_num):
            n_cols = len(splits)
            fig = make_subplots(
                rows=1,
                cols=n_cols,
                specs=[[{"type": "polar"}] * n_cols],
                subplot_titles=[s.capitalize() for s in splits],
            )
            for c, split in enumerate(splits, 1):
                if g_int not in mean_split[split].index:
                    continue
                theta = feats + [feats[0]]
                r_vals = mean_split[split].loc[g_int].tolist() + [mean_split[split].loc[g_int, feats[0]]]
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_vals,
                        theta=theta,
                        subplot=f"polar{c}",
                        name=f"GH{g_num}",
                        line=dict(color=_rgba(self.group_palette_[g_int], 0.8)),
                        fill="toself",
                        fillcolor=_rgba(self.group_palette_[g_int], 0.25),
                        customdata=[_vol(all_splits[split], g_int)] * len(r_vals),
                        hovertemplate=(f"Split: {split.capitalize()}<br>" "Volume: %{customdata:,}<extra></extra>"),
                        showlegend=False,
                    ),
                    row=1,
                    col=c,
                )
            fig.update_layout(
                **_polar_cfg(n_cols),
                template="plotly_white",
                title=f"{title or 'Radar'} – GH{g_num}",
                height=800,
                width=1200,
            )
            figs[g_num] = fig
            if save and self.save_dir:
                fig.write_image(self.save_dir / f"radar_GH{g_num}.png")
        return figs

    # def plot_group_radar(
    #     self,
    #     *,
    #     groups: list[int] | None = None,
    #     separated: bool = False,
    #     splits: list[str] | None = None,
    #     scaler: Literal["zscore", "minmax"] = "zscore",
    #     animation: bool = False,
    #     save: bool = False,
    #     title: str = "",
    # ) -> Union[go.Figure, Dict[int, go.Figure]]:
    #     """Radar chart das médias de features por GH.

    #     Parâmetros
    #     ----------
    #     groups : list[int] | None
    #         Lista com os números dos GHs desejados (GH1 = maior bad‑rate).
    #         ``None`` => todos.
    #     separated : bool
    #         • False → 1 radar por split (Train/Test/Val).
    #         • True  → 1 radar por GH (cada radar pode conter vários splits).
    #     splits : list[str] | None
    #         Subconjunto de ``["train","test","val"]``. ``None`` => todos.
    #     scaler : {"zscore","minmax"}
    #         Método de normalização das features no radar.
    #     animation : bool
    #         Se ``True``, gera animação mensal (1 subplot por GH) sincronizada
    #         pela coluna de data.
    #     save : bool
    #         Se ``True`` e ``self.save_dir`` definido, grava PNG/HTML (quando animado).
    #     title : str
    #         Título base da figura.
    #     """

    #     import pandas as pd, numpy as np, plotly.graph_objects as go
    #     from plotly.subplots import make_subplots
    #     from math import ceil

    #     # ---------------- validações ----------------
    #     if self.group_ is None:
    #         raise ValueError("Homogeneous groups were not computed.")
    #     if animation and self.date_col is None:
    #         raise ValueError("`date_col` é necessário para animation=True.")

    #     # ---------------- mapeia GH ------------------
    #     br = (
    #         self.data_.groupby(self.group_col_)[self.target_col]
    #         .mean()
    #         .sort_values(ascending=False)          # GH1 = pior
    #     )
    #     gh_order      = list(br.index)             # labels internos
    #     gh_to_num     = {g: i + 1 for i, g in enumerate(gh_order)}
    #     num_to_gh     = {v: k for k, v in gh_to_num.items()}

    #     if groups is None:
    #         groups_num = list(range(1, len(gh_order) + 1))
    #     else:
    #         groups_num = [g for g in groups if g in num_to_gh]
    #     groups_int = [num_to_gh[g] for g in groups_num]

    #     # ---------------- splits ---------------------
    #     all_splits = {"train": self.df_train, "test": self.df_test}
    #     if self.df_val is not None:
    #         all_splits["val"] = self.df_val
    #     if splits is None:
    #         splits = list(all_splits.keys())
    #     splits = [s.lower() for s in splits if s.lower() in all_splits]

    #     # ---------------- features -------------------
    #     feats = [
    #         c for c in self.predictor_cols
    #         if pd.api.types.is_numeric_dtype(self.data_[c])
    #     ]

    #     def _scale(df):
    #         if scaler == "zscore":
    #             return df[feats].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
    #         return df[feats].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    #     def _vol(df, g_int):
    #         return int(df[df[self.group_col_] == g_int].shape[0])

    #     # =============== ANIMAÇÃO ====================
    #     if animation:
    #         periods = (
    #             pd.to_datetime(self.data_[self.date_col])
    #             .dt.to_period("M").sort_values().unique()
    #         )
    #         n_gh, n_cols = len(groups_int), min(3, len(groups_int))
    #         n_rows = ceil(n_gh / n_cols)
    #         specs  = [[{"type": "polar"}]*n_cols for _ in range(n_rows)]
    #         fig = make_subplots(rows=n_rows, cols=n_cols,
    #                             specs=specs,
    #                             subplot_titles=[f"GH{n}" for n in groups_num])

    #         # ---------------- frames ----------------
    #         frames = []
    #         for per in periods:
    #             traces = []
    #             for idx, (g_int, g_num) in enumerate(zip(groups_int, groups_num)):
    #                 r_idx = idx//n_cols + 1; c_idx = idx % n_cols + 1
    #                 theta = feats + [feats[0]]
    #                 for split in splits:
    #                     df_p = all_splits[split]
    #                     mask = pd.to_datetime(df_p[self.date_col]).dt.to_period("M") == per
    #                     df_p = df_p[mask]
    #                     if df_p.empty or g_int not in df_p[self.group_col_].values:
    #                         continue
    #                     mean_row = _scale(df_p).groupby(
    #                         df_p[self.group_col_])[feats].mean().loc[g_int]
    #                     r_vals = mean_row.tolist() + [mean_row[feats[0]]]
    #                     traces.append(go.Scatterpolar(
    #                         r=r_vals, theta=theta,
    #                         subplot=f"polar{r_idx}{c_idx}",
    #                         name=f"GH{g_num} – {split.capitalize()}",
    #                         line=dict(color=_rgba(self.group_palette_[g_int], .8)),
    #                         fill="toself",
    #                         fillcolor=_rgba(self.group_palette_[g_int], .25),
    #                         customdata=[_vol(df_p, g_int)]*len(r_vals),
    #                         hovertemplate=(f"Safra: {per:%Y%m}<br>"
    #                                     f"GH{g_num} – {split.capitalize()}<br>"
    #                                     "Volume: %{customdata:,}<extra></extra>"),
    #                         showlegend=False,
    #                     ))
    #             frames.append(go.Frame(data=traces, name=per.strftime("%Y-%m")))

    #         if frames:
    #             fig.add_traces(frames[0].data); fig.frames = frames
    #         fig.update_layout(
    #             template="plotly_white", title=title or "Evolução mensal",
    #             height=800*n_rows, width=1200*n_cols,
    #             polar=dict(radialaxis=dict(visible=False)),
    #             updatemenus=[dict(type="buttons", buttons=[
    #                 dict(label="Play",  method="animate",
    #                     args=[None, {"frame": {"duration": 600, "redraw": True},
    #                                 "fromcurrent": True}]),
    #                 dict(label="Pause", method="animate",
    #                     args=[[None], {"frame": {"duration": 0},
    #                                     "mode": "immediate"}]),
    #             ])])
    #         if save and self.save_dir:
    #             fig.write_html(self.save_dir/"group_radar_animation.html")
    #         return fig

    #     # =============== DADOS ESTÁTICOS ==============
    #     mean_split = {s: _scale(df).groupby(df[self.group_col_])[feats].mean()
    #                 for s,df in all_splits.items()}

    #     # ---------- combined (1 radar / split) --------
    #     if not separated:
    #         n_cols = len(splits)
    #         fig = make_subplots(rows=1, cols=n_cols,
    #                             specs=[[{"type": "polar"}]*n_cols],
    #                             subplot_titles=[s.capitalize() for s in splits])

    #         for c, split in enumerate(splits, 1):
    #             for g_int, g_num in zip(reversed(groups_int), reversed(groups_num)):
    #                 if g_int not in mean_split[split].index: continue
    #                 theta = feats + [feats[0]]
    #                 r = mean_split[split].loc[g_int].tolist()+[mean_split[split].loc[g_int,feats[0]]]
    #                 fig.add_trace(go.Scatterpolar(
    #                     r=r, theta=theta, subplot=f"polar{c}",
    #                     name=f"GH{g_num}",
    #                     line=dict(color=_rgba(self.group_palette_[g_int], .8)),
    #                     fill="toself", fillcolor=_rgba(self.group_palette_[g_int], .25),
    #                     customdata=[_vol(all_splits[split], g_int)]*len(r),
    #                     hovertemplate=f"GH{g_num}<br>Volume: %{{customdata:,}}<extra></extra>"
    #                 ), row=1, col=c)

    #         fig.update_layout(
    #             template="plotly_white", title=title or "Radar – GHs combinados",
    #             height=800, width=1200*n_cols,
    #             legend_title="Grupos Homogêneos")
    #         if save and self.save_dir:
    #             fig.write_image(self.save_dir/"group_radar_combined.png")
    #         return fig

    #     # ---------- separated (dict GH → Figure) ------
    #     figs = {}
    #     for g_int, g_num in zip(groups_int, groups_num):
    #         fig = make_subplots(rows=1, cols=len(splits),
    #                             specs=[[{"type": "polar"}]*len(splits)],
    #                             subplot_titles=[s.capitalize() for s in splits])
    #         for c, split in enumerate(splits, 1):
    #             if g_int not in mean_split[split].index: continue
    #             theta = feats+[feats[0]]
    #             r = mean_split[split].loc[g_int].tolist()+[mean_split[split].loc[g_int,feats[0]]]
    #             fig.add_trace(go.Scatterpolar(
    #                 r=r, theta=theta, subplot=f"polar{c}",
    #                 name=split.capitalize(),
    #                 line=dict(color=_rgba(self.group_palette_[g_int], .8)),
    #                 fill="toself", fillcolor=_rgba(self.group_palette_[g_int], .25),
    #                 customdata=[_vol(all_splits[split], g_int)]*len(r),
    #                 hovertemplate=(f"GH{g_num} – {split.capitalize()}<br>"
    #                             "Volume: %{customdata:,}<extra></extra>"),
    #             ), row=1, col=c)
    #         fig.update_layout(
    #             template="plotly_white", title=f"{title or 'Radar'} – GH{g_num}",
    #             height=800, width=1200*len(splits),
    #             showlegend=False)
    #         figs[g_num] = fig
    #         if save and self.save_dir:
    #             fig.write_image(self.save_dir/f"radar_GH{g_num}.png")
    #     return figs

    def plot_decile_ks(
        self,
        *,
        splits: list[str] | None = None,
        n_bins: int = 10,
        ascending: bool = True,
        group_id: int | None = None,
        title: str = "",
        **kwargs: Any,
    ) -> go.Figure:
        """Plot decile KS per split without silently mixing train/test/val."""
        from plotly.subplots import make_subplots

        split_names = self._infer_splits(splits)
        available = {"train": self.df_train, "test": self.df_test}
        if self.df_val is not None:
            available["val"] = self.df_val

        bar_color = kwargs.pop("bar_color", "#4527a0")
        line_color = kwargs.pop("line_color", "#fa6300")
        font_family = kwargs.pop("font_family", "Arial Black")

        tables: list[pd.DataFrame] = []
        split_tables: dict[str, tuple[pd.DataFrame, float]] = {}
        for split in split_names:
            df = available[split].copy()
            if group_id is not None:
                if self.group_col_ not in df.columns:
                    raise ValueError("Homogeneous groups were not computed.")
                df = df[df[self.group_col_] == group_id]
            table, ks_value = ks_table(
                df,
                score_col=self.score_col_,
                target_col=self.target_col,
                n_bins=n_bins,
                ascending=ascending,
            )
            table = table.copy()
            table.insert(0, "Split", split.capitalize())
            tables.append(table)
            split_tables[split] = (table, ks_value)

        self.decile_ks_table_ = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

        subplot_titles = []
        for split in split_names:
            _, ks_value = split_tables[split]
            ks_label = "N/A" if pd.isna(ks_value) else f"{ks_value:.2%}"
            subplot_titles.append(f"{split.capitalize()} KS {ks_label}")

        fig = make_subplots(
            rows=1,
            cols=len(split_names),
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True} for _ in split_names]],
        )

        for col_idx, split in enumerate(split_names, start=1):
            table, _ = split_tables[split]
            if table.empty:
                continue
            fig.add_bar(
                x=table["decile"].astype(str),
                y=table["total"],
                name=f"{split.capitalize()} total",
                marker_color=bar_color,
                opacity=0.65,
                showlegend=col_idx == 1,
                row=1,
                col=col_idx,
                secondary_y=False,
            )
            fig.add_scatter(
                x=table["decile"].astype(str),
                y=table["bad_rate"],
                name=f"{split.capitalize()} bad rate",
                mode="lines+markers",
                marker_color=line_color,
                line=dict(color=line_color, width=2),
                showlegend=col_idx == 1,
                row=1,
                col=col_idx,
                secondary_y=True,
            )
            total = table["total"].sum()
            if total > 0:
                mean_rate = table["bad"].sum() / total
                fig.add_hline(
                    y=mean_rate,
                    line=dict(color="gray", dash="dot", width=1),
                    row=1,
                    col=col_idx,
                    secondary_y=True,
                )
            fig.update_xaxes(title_text="Decil", row=1, col=col_idx)
            fig.update_yaxes(title_text="Total", showgrid=False, row=1, col=col_idx, secondary_y=False)
            fig.update_yaxes(
                title_text="Bad rate",
                tickformat=".2%",
                showgrid=False,
                row=1,
                col=col_idx,
                secondary_y=True,
            )

        fig.update_layout(
            title=title or "Ordenacao por decil",
            template="plotly_white",
            barmode="overlay",
            font=dict(family=font_family),
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        )
        return fig

    def plot_sankey_migration(
        self,
        start_period: int | str,
        end_period: int | str,
        *,
        period_to_period: bool = True,
        money_col: str | None = None,
        top_n: int | None = None,
        normalize: bool = False,
        cmap: str = "Blues",
        save: bool = False,
        raise_on_empty: bool = True,
    ) -> tuple[list[go.Figure] | go.Figure, dict[str, Any]] | tuple[None, dict[str, Any]]:
        """Visualize contract migration between GHs using Sankey diagrams.

        Parameters
        ----------
        start_period, end_period
            Periods in ``YYYYMM`` or ``YYYY-MM`` format.
        period_to_period
            If ``True`` generate a diagram for each consecutive period between
            ``start_period`` and ``end_period``. Otherwise a single net diagram
            is returned.
        money_col
            Optional column with monetary value to aggregate. When ``None`` the
            number of contracts is used.
        top_n
            Keep only the ``top_n`` flows by value, grouping the remainder as
            ``Other``.
        normalize
            If ``True`` flows are expressed as percentages of the originating
            group.
        cmap
            Colour map used for the node palette.
        save
            When ``True`` and ``save_dir`` was provided, images are written to
            disk.
        raise_on_empty
            When ``True`` (default) a :class:`ValueError` is raised if no flows
            are found between the selected periods. If ``False`` an empty result
            ``(None, {})`` is returned instead.
        """

        import pandas as pd
        import plotly.graph_objects as go

        logger = logging.getLogger("riskpilot")

        if self.date_col is None:
            raise ValueError("`date_col` é obrigatório para plot_sankey_migration().")

        df_all = self.data_.copy()
        df_all[self.date_col] = pd.to_datetime(df_all[self.date_col]).dt.to_period("M")
        if df_all[self.date_col].isna().any():
            raise ValueError("date_col contains NaT values – check parsing.")

        gh_col = next(
            (c for c in [self.group_col, self.group_col_] if c and c in self.data_.columns),
            None,
        )
        if gh_col is None:
            raise ValueError("Group column não encontrado para Sankey.")

        id_col = self.id_cols[0]
        if df_all[gh_col].isna().any():
            raise ValueError("gh_col has missing values; compute groups first.")
        df_all[id_col] = df_all[id_col].astype(str)

        start_p = pd.Period(str(start_period), freq="M")
        end_p = pd.Period(str(end_period), freq="M")

        periods = df_all[self.date_col].unique()
        if start_p not in periods or end_p not in periods:
            raise ValueError("start_period ou end_period ausente em self.data_.")

        self._compute_group_palette()

        br = df_all.groupby(gh_col)[self.target_col].mean().sort_values(ascending=False)
        gh_to_label = {g: f"GH{i+1}" for i, g in enumerate(br.index)}

        def _historical_means() -> pd.Series:
            flows = []
            ordered = sorted(periods)
            for p1, p2 in zip(ordered[:-1], ordered[1:]):
                d1 = df_all[df_all[self.date_col] == p1]
                d2 = df_all[df_all[self.date_col] == p2][[id_col, gh_col]]
                tmp = (
                    d1.merge(d2, on=id_col, suffixes=("_start", "_end"))
                    .groupby([f"{gh_col}_start", f"{gh_col}_end"], as_index=False)
                    .agg(value=(money_col, "sum") if money_col else (id_col, "count"))
                )
                flows.append(tmp)
            if flows:
                hist = pd.concat(flows, ignore_index=True)
                return hist.groupby([f"{gh_col}_start", f"{gh_col}_end"])["value"].mean()
            return pd.Series(dtype=float)

        hist_means = _historical_means()

        def _one_sankey(p1: pd.Period, p2: pd.Period) -> tuple[go.Figure | None, dict[str, Any]]:
            df_start = df_all[df_all[self.date_col] == p1].copy()
            df_end = df_all[df_all[self.date_col] == p2][[id_col, gh_col]].copy()

            df_start[id_col] = df_start[id_col].astype(str)
            df_end[id_col] = df_end[id_col].astype(str)

            logger.info(
                "Period %s: %d contracts, Period %s: %d contracts",
                p1,
                len(df_start),
                p2,
                len(df_end),
            )

            cross = (
                df_start.merge(df_end, on=id_col, suffixes=("_start", "_end"))
                .groupby([f"{gh_col}_start", f"{gh_col}_end"], as_index=False)
                .agg(value=(money_col, "sum") if money_col else (id_col, "count"))
                .rename(columns={f"{gh_col}_start": "gh_start", f"{gh_col}_end": "gh_end"})
            )

            logger.info("After merge: %d flows", len(cross))

            if cross.empty:
                msg = (
                    f"No migrations found between {p1.strftime('%Y-%m')} and {p2.strftime('%Y-%m')}. "
                    "Verify id_col overlap and homogeneous groups."
                )
                if raise_on_empty:
                    raise ValueError(msg)
                logger.warning(msg)
                return None, {}

            if normalize and not cross.empty:
                cross["value"] = cross["value"] / cross.groupby("gh_start")["value"].transform("sum")

            if top_n is not None and top_n < len(cross):
                cross = cross.sort_values("value", ascending=False)
                other_val = cross.iloc[top_n:]["value"].sum()
                cross = cross.iloc[:top_n]
                if other_val > 0:
                    cross = pd.concat(
                        [
                            cross,
                            pd.DataFrame(
                                {
                                    "gh_start": ["Other"],
                                    "gh_end": ["Other"],
                                    "value": [other_val],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

            nodes = pd.Index(cross["gh_start"].tolist() + cross["gh_end"].tolist()).unique().tolist()
            idx_map = {n: i for i, n in enumerate(nodes)}
            labels = [gh_to_label.get(n, "Other") for n in nodes]
            colors = [self.group_palette_.get(n, "lightgrey") for n in nodes]

            fig = go.Figure(
                go.Sankey(
                    node=dict(label=labels, color=colors),
                    link=dict(
                        source=cross["gh_start"].map(idx_map),
                        target=cross["gh_end"].map(idx_map),
                        value=cross["value"],
                    ),
                )
            )

            title = (
                f"GH Migration – {p1.strftime('%Y-%m')} → {p2.strftime('%Y-%m')}"
                if period_to_period
                else f"GH Migration – {start_p.strftime('%Y-%m')} ⟶ {end_p.strftime('%Y-%m')} (net)"
            )
            fig.update_layout(template="plotly_white", title=title, height=800, width=1200)

            if save and self.save_dir:
                fname = f"sankey_{p1.strftime('%Y%m')}_{p2.strftime('%Y%m')}.png"
                fig.write_image(self.save_dir / fname)
                fig.write_html(self.save_dir / fname.replace(".png", ".html"))

            new_entries = df_end[~df_end[id_col].isin(df_start[id_col])][gh_col]
            exits = df_start[~df_start[id_col].isin(df_end[id_col])][gh_col]

            metrics = {
                "new_entries": (
                    gh_to_label.get(new_entries.value_counts().idxmax(), None) if not new_entries.empty else None
                ),
                "exits": (gh_to_label.get(exits.value_counts().idxmax(), None) if not exits.empty else None),
                "common_flow": None,
                "outlier_flow": None,
            }

            if not cross.empty:
                pair = cross.sort_values("value", ascending=False).iloc[0]
                metrics["common_flow"] = (
                    gh_to_label.get(pair["gh_start"]),
                    gh_to_label.get(pair["gh_end"]),
                )

                if not hist_means.empty:
                    cur = cross.set_index(["gh_start", "gh_end"])["value"]
                    comp = hist_means.reindex(cur.index)
                    diff = (cur - comp).abs()
                    if not diff.isna().all():
                        max_pair = diff.idxmax()
                        metrics["outlier_flow"] = (
                            gh_to_label.get(max_pair[0]),
                            gh_to_label.get(max_pair[1]),
                        )

            return fig, metrics

        if period_to_period:
            period_range = pd.period_range(start_p, end_p, freq="M")
            fig_list: list[go.Figure] = []
            metrics_dict: dict[str, Any] = {}
            for p1, p2 in zip(period_range[:-1], period_range[1:]):
                logger.info("Gerando Sankey %s → %s", p1, p2)
                fig, met = _one_sankey(p1, p2)
                if fig is not None:
                    fig_list.append(fig)
                metrics_dict[f"{p1.strftime('%Y%m')}->{p2.strftime('%Y%m')}"] = met
            return fig_list, metrics_dict

        logger.info("Gerando Sankey %s ⟶ %s", start_p, end_p)
        fig, metrics_single = _one_sankey(start_p, end_p)
        return fig, metrics_single

    ## ---------- helpers ----------
    def _load_model(self, model: Union[str, Path, object]):
        """Load model from path or return object as‑is."""
        if isinstance(model, (str, Path)):
            model_path = Path(model)
            if model_path.suffix in {".joblib", ".jbl"}:
                return joblib.load(model_path)
            elif model_path.suffix in {".pkl", ".pickle"}:
                with open(model_path, "rb") as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported model file extension: {model_path.suffix}")
        else:
            # assume object is already a fitted estimator
            if not hasattr(model, "predict_proba"):
                raise AttributeError("Provided model object lacks predict_proba.")
            return model

    def _validate_data(self) -> None:
        """Basic dataframe validations."""
        for name, df in [
            ("df_train", self.df_train),
            ("df_test", self.df_test),
            ("df_val", self.df_val),
        ]:
            if df is None:
                continue
            if self.target_col not in df.columns:
                raise KeyError(f"{self.target_col} missing in {name}.")
            if df[self.target_col].isna().any():
                raise ValueError(f"NaN detected in target column of {name}.")

            missing_ids = [col for col in self.id_cols if col not in df.columns]
            if missing_ids:
                raise KeyError(f"{name} missing id_cols: {missing_ids}")

        if self.homogeneous_group is None and self.group_col:
            for name, df in [
                ("df_train", self.df_train),
                ("df_test", self.df_test),
                ("df_val", self.df_val),
            ]:
                if df is None:
                    continue
                if self.group_col not in df.columns:
                    raise KeyError(f"{name} missing group_col: {self.group_col}")

    def _parse_date_col(self) -> None:
        """Parse `date_col` to datetime when format yyyymm is detected."""
        if not self.date_col:
            return

        for df in [self.df_train, self.df_test, self.df_val]:
            if df is None or self.date_col not in df.columns:
                continue

            col = df[self.date_col]
            if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
                try:
                    df[self.date_col] = pd.to_datetime(col.astype(int).astype(str), format="%Y%m")
                    continue
                except Exception:
                    pass
            df[self.date_col] = pd.to_datetime(col, errors="coerce")

    def _infer_predictors(self) -> List[str]:
        """Infer intersection of columns across all datasets, excluding id/date/group/target."""
        cols = set(self.df_train.columns)
        cols &= set(self.df_test.columns)
        if self.df_val is not None:
            cols &= set(self.df_val.columns)

        exclude = set(self.id_cols + [self.target_col])
        if self.date_col:
            exclude.add(self.date_col)
        if self.group_col:
            exclude.add(self.group_col)

        predictor_cols = sorted(list(cols - exclude))
        if not predictor_cols:
            raise ValueError("No predictor columns detected after exclusions.")
        return predictor_cols

    def _get_model_feature_names(self) -> Optional[List[str]]:
        """Return feature names used during model training, if available."""
        if hasattr(self.model, "feature_names_in_"):
            return list(getattr(self.model, "feature_names_in_"))
        try:
            booster = self.model.get_booster()
            if hasattr(booster, "feature_names"):
                return list(booster.feature_names)
        except Exception:
            pass
        return None

    def _get_model_n_features(self) -> Optional[int]:
        """Return the number of features the model expects, if available."""
        if hasattr(self.model, "n_features_in_"):
            return int(getattr(self.model, "n_features_in_"))
        try:
            booster = self.model.get_booster()
            if hasattr(booster, "num_features"):
                return int(booster.num_features())
        except Exception:
            pass
        return None

    def _align_predictors_with_model(self, cols: List[str]) -> List[str]:
        """Ensure predictors match model's training features, preserving order."""
        if self.model_feature_names:
            missing = [c for c in self.model_feature_names if c not in cols]
            if missing:
                raise ValueError(f"Model expects columns not present in provided data: {missing}")
            return [c for c in self.model_feature_names if c in cols]

        if self.model_n_features is not None and len(cols) != self.model_n_features:
            raise ValueError(
                f"Number of predictor columns ({len(cols)}) does not match model "
                f"expectation ({self.model_n_features})."
            )
        return cols

    def _validate_predictors(self) -> None:
        """Ensure predictor columns align with model expectations."""
        for name, df in [
            ("df_train", self.df_train),
            ("df_test", self.df_test),
            ("df_val", self.df_val),
        ]:
            if df is None:
                continue
            missing = [c for c in self.predictor_cols if c not in df.columns]
            if missing:
                raise KeyError(f"{name} missing predictor columns: {missing}")

        if self.model_feature_names:
            ordered = [c for c in self.model_feature_names if c in self.predictor_cols]
            if ordered != self.predictor_cols:
                self.predictor_cols = ordered
        elif self.model_n_features is not None and len(self.predictor_cols) != self.model_n_features:
            raise ValueError(f"Model expects {self.model_n_features} features, got {len(self.predictor_cols)}")

    def _score_datasets(self) -> None:
        """Add predicted probabilities and labels to each split."""
        dfs = [("train", self.df_train), ("test", self.df_test)]
        if self.df_val is not None:
            dfs.append(("val", self.df_val))

        for name, df in dfs:
            proba = self.model.predict_proba(df[self.predictor_cols])[:, self._pos_class_idx]
            df[self.score_col_] = proba
            df[self.label_col_] = (proba >= self.threshold).astype(int)
            df["Split"] = name.capitalize()

        self.data_ = pd.concat(
            [self.df_train, self.df_test] + ([self.df_val] if self.df_val is not None else []),
            axis=0,
            ignore_index=True,
        )

    def _assign_groups(self) -> None:
        """Create homogeneous groups according to ``self.homogeneous_group``."""
        if self.homogeneous_group is None:
            return

        self.group_ = {}

        if isinstance(self.homogeneous_group, str):
            if self.homogeneous_group != "auto":
                raise ValueError("Unsupported string for homogeneous_group")

            if OptimalBinning is None:
                raise ImportError(
                    "OptimalBinning is required for `homogeneous_group='auto'`. "
                    "Install with `pip install riskpilot[binning]`."
                )

            optb = OptimalBinning(
                name="y_proba_train",
                dtype="numerical",
                solver="mip",
                min_prebin_size=0.01,
                max_n_bins=5,
                min_bin_size=0.05,
                monotonic_trend="ascending",
            )
            optb.fit(self.df_train[self.score_col_] * 1000, self.df_train[self.target_col])
            self.binning_table_ = optb.binning_table.build()

            for name, df in [
                ("train", self.df_train),
                ("test", self.df_test),
                ("val", self.df_val),
            ]:
                if df is None:
                    continue
                labels = optb.transform(df[self.score_col_] * 1000, metric="bins")
                df[self.group_col_] = labels
                self.group_[name] = labels

        elif isinstance(self.homogeneous_group, int):
            n = int(self.homogeneous_group)
            for name, df in [
                ("train", self.df_train),
                ("test", self.df_test),
                ("val", self.df_val),
            ]:
                if df is None:
                    continue
                bins = pd.qcut(
                    df[self.score_col_].rank(method="first"),
                    q=n,
                    labels=range(1, n + 1),
                ).astype(int)
                df[self.group_col_] = bins
                self.group_[name] = bins
            self.binning_table_ = None

        else:
            groups = pd.Series(self.homogeneous_group)
            if len(groups) != len(self.data_):
                raise ValueError("Length of provided group labels does not match data.")

            self.data_[self.group_col_] = groups.reset_index(drop=True)
            start = 0
            for name, df in [
                ("train", self.df_train),
                ("test", self.df_test),
                ("val", self.df_val),
            ]:
                if df is None:
                    continue
                end = start + len(df)
                df[self.group_col_] = groups.iloc[start:end].values
                self.group_[name] = df[self.group_col_]
                start = end
            self.binning_table_ = None

        self.data_ = pd.concat(
            [self.df_train, self.df_test] + ([self.df_val] if self.df_val is not None else []),
            axis=0,
            ignore_index=True,
        )
        self._compute_group_palette()

    def _compute_group_palette(self) -> None:
        """Create consistent colours per group based on event rate."""
        group_col = None
        for cand in [self.group_col, self.group_col_]:
            if cand and cand in self.data_.columns:
                group_col = cand
                break
        if group_col is None:
            return

        rates = self.data_.groupby(group_col)[self.target_col].mean().sort_values()
        palette = sns.diverging_palette(240, 10, n=len(rates))
        self.group_palette_ = {
            grp: f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for grp, (r, g, b) in zip(rates.index, palette)
        }

    def _psi_variables(self) -> List[str]:
        """Select variables to evaluate for PSI (exclude id/date/target)."""
        exclude = set(self.id_cols + [self.target_col])
        if self.date_col:
            exclude.add(self.date_col)
        vars_ = [c for c in self.df_train.columns if c not in exclude]
        numeric_vars = [v for v in vars_ if pd.api.types.is_numeric_dtype(self.df_train[v])]
        return numeric_vars

    def _infer_splits(self, splits: Sequence[str] | None) -> list[str]:
        """Return valid split names in order.

        Parameters
        ----------
        splits : Sequence[str] | None
            Desired splits (``"train"``, ``"test"``, ``"val"``). ``None`` uses all
            available.

        Returns
        -------
        list[str]
            Validated list of split names.
        """

        available = ["train", "test"] + (["val"] if self.df_val is not None else [])
        if splits is None:
            return available
        result = []
        for s in splits:
            if s not in available:
                raise KeyError(f"split '{s}' not available")
            result.append(s)
        return result

    def _get_shap_explainer(self) -> "shap.explainers._explainer.Explainer":
        """Return and cache the appropriate SHAP explainer for the model.

        The explainer type is determined automatically based on ``self.model``.
        Tree-based models use :class:`shap.TreeExplainer`, linear models use
        :class:`shap.LinearExplainer`, and any other estimator falls back to
        :class:`shap.KernelExplainer` with a k-means background sampled from the
        training data. The resulting explainer is stored on ``self._shap_explainer``
        to avoid recomputation.

        Time complexity and memory usage depend on the chosen explainer. In
        particular, ``KernelExplainer`` can be expensive for many features and
        background samples.

        Returns
        -------
        shap.explainers._explainer.Explainer
            A SHAP explainer instance appropriate for ``self.model``.
        """

        if getattr(self, "_shap_explainer", None) is not None:
            return self._shap_explainer  # type: ignore[return-value]

        if shap is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SHAP visualisations need the optional dependency 'shap'. " "Install with `pip install riskpilot[viz]`."
            )

        model = self.model
        if (
            (XGBModel is not object and isinstance(model, XGBModel))
            or (LGBMModel is not object and isinstance(model, LGBMModel))
            or isinstance(
                model,
                (
                    RandomForestClassifier,
                    GradientBoostingClassifier,
                    DecisionTreeClassifier,
                ),
            )
        ):
            explainer = shap.TreeExplainer(model)
        elif isinstance(
            model,
            (
                LogisticRegression,
                LinearRegression,
                SGDClassifier,
                SGDRegressor,
            ),
        ):
            X_train = self.df_train[self.predictor_cols]
            explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
        else:
            logger = logging.getLogger("riskpilot")
            X_train = self.df_train[self.predictor_cols]
            background = shap.sample(X_train, 100, random_state=42) if len(X_train) > 100 else X_train
            logger.warning(
                "\u26a0\ufe0f Falling back to shap.KernelExplainer. This could be "
                "very slow for large datasets and many features. Consider using "
                "a tree or linear model, or sample your data."
            )
            explainer = shap.KernelExplainer(
                lambda m: model.predict_proba(m)[:, self._pos_class_idx],
                background,
            )

        self._shap_explainer = explainer
        self.feature_names_ = list(self.predictor_cols)
        return explainer

    # ---- SHAP cache helpers ---- #

    def _cache_path(self, split: str) -> Path:
        base = self.save_dir or Path(".")
        return base / "shap_cache" / self._fingerprint / f"{split}.joblib"

    def _save_shap_to_disk(self, split: str, explanation: Any) -> None:
        path = self._cache_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(explanation, path, compress=3)
        except Exception as err:  # pragma: no cover
            logging.getLogger("riskpilot").warning("SHAP cache write failed: %s", err)

    def _load_shap_from_disk(self, split: str) -> Any | None:
        path = self._cache_path(split)
        if not path.is_file():
            return None
        try:
            obj = joblib.load(path)
        except Exception:
            return None
        n_features = getattr(obj, "values", np.array([])).shape[-1]
        if n_features != len(self.predictor_cols):
            return None
        return obj

    def _compute_shap_values_no_cache(self, X: pd.DataFrame) -> "shap.Explanation":
        if shap is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SHAP visualisations need the optional dependency 'shap'. " "Install with `pip install riskpilot[viz]`."
            )

        expected_cols = list(getattr(self, "feature_names_", self.predictor_cols))
        if list(X.columns) != expected_cols:
            raise ValueError("Input features must match model training columns: " f"{expected_cols}")

        explainer = self._get_shap_explainer()
        if isinstance(explainer, shap.TreeExplainer):
            explanation = explainer(X, check_additivity=False)
        else:
            explanation = explainer.shap_values(X)
            explanation = shap.Explanation(
                np.asarray(explanation),
                base_values=explainer.expected_value,
                data=X.values,
                feature_names=expected_cols,
            )
        return explanation

    def _compute_shap_values(self, X: pd.DataFrame, *, split_name: str, use_cache: bool = False) -> "shap.Explanation":
        """Compute SHAP values for a pre-processed feature matrix.

        The columns in ``X`` must match those used during model training. For
        large datasets, consider sampling rows prior to calling this method to
        reduce memory usage.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix with the same columns and order as ``self.predictor_cols``.
        use_cache : bool, default ``False``
            Persist SHAP values in-memory and on-disk. Can be globally enabled by
            setting the environment variable ``BPE_SHAP_CACHE=1``.

        Returns
        -------
        shap.Explanation
            SHAP values and expected value for the provided samples.
        """

        if shap is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SHAP visualisations need the optional dependency 'shap'. " "Install with `pip install riskpilot[viz]`."
            )

        if os.getenv("BPE_SHAP_CACHE", "0") == "1":
            use_cache = True

        logger = logging.getLogger("riskpilot")
        if not use_cache:
            logger.debug("SHAP caching disabled (use_cache=False)")

        expected_cols = list(getattr(self, "feature_names_", self.predictor_cols))
        if list(X.columns) != expected_cols:
            raise ValueError("Input features must match model training columns: " f"{expected_cols}")

        if use_cache and (split_name, self._fingerprint) in self._shap_cache:
            self._cache_hits += 1
            return self._shap_cache[(split_name, self._fingerprint)]

        if use_cache:
            disk = self._load_shap_from_disk(split_name)
            if disk is not None:
                self._disk_hits += 1
                self._shap_cache[(split_name, self._fingerprint)] = disk
                return disk

        explanation = self._compute_shap_values_no_cache(X)

        if use_cache:
            self._cache_misses += 1
            self._shap_cache[(split_name, self._fingerprint)] = explanation
            self._save_shap_to_disk(split_name, explanation)
        return explanation

    def clear_shap_cache(self, *, splits: list[str] | None = None) -> None:
        """Delete cache files for given splits."""

        splits = self._infer_splits(splits)
        for split in splits:
            self._shap_cache.pop((split, self._fingerprint), None)
            path = self._cache_path(split)
            if path.is_file():
                try:
                    path.unlink()
                except Exception:  # pragma: no cover
                    pass

    def cache_stats(self) -> dict[str, Any]:
        """Return cache statistics."""

        mem = sum(getattr(v, "values", np.array([])).nbytes for v in self._shap_cache.values())
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "disk_hits": self._disk_hits,
            "size_mb": mem / (1024 * 1024),
        }

    def _prepare_shap_summary(
        self,
        shap_dict: dict[str, shap.Explanation],
        *,
        max_display: int = 20,
        feature_groups: dict[str, Sequence[str]] | None = None,
    ) -> pd.DataFrame:
        """Aggregate SHAP values across splits into a tidy table.

        Args:
            shap_dict: Mapping of split name to :class:`shap.Explanation`.
            max_display: Number of top features per split after grouping.
            feature_groups: Optional mapping ``{group: [features...]}`` used to
                collapse raw features.

        Returns:
            pandas.DataFrame: Long-format DataFrame with columns ``feature``,
            ``split``, ``importance`` and ``direction``.
        """

        if not shap_dict:
            return pd.DataFrame(columns=["feature", "split", "importance", "direction"])

        ref = next(iter(shap_dict.values()))
        ref_features = list(ref.feature_names)
        n_features = len(ref_features)

        # validate shapes and feature names
        for split, expl in shap_dict.items():
            names = list(expl.feature_names)
            if names != ref_features:
                raise ValueError("All splits must share the same feature order")
            values = np.asarray(expl.values)
            if values.ndim == 3 and values.shape[1] == 1:
                values = values[:, 0, :]
            if values.ndim != 2 or values.shape[1] != n_features:
                raise ValueError("SHAP values must be 2-D with consistent features")

        feature_groups = feature_groups or {}

        group_members = {f for members in feature_groups.values() for f in members}
        for f in group_members:
            if f not in ref_features:
                raise ValueError(f"Feature '{f}' specified in groups not found")

        split_frames = {}
        top_features: list[set[str]] = []

        for split, expl in shap_dict.items():
            values = np.asarray(expl.values)
            if values.ndim == 3 and values.shape[1] == 1:
                values = values[:, 0, :]

            imp = np.abs(values).mean(axis=0)
            dir_ = np.sign(values).mean(axis=0)
            df = pd.DataFrame(
                {
                    "feature": ref_features,
                    "importance": imp,
                    "direction": dir_,
                }
            )

            if feature_groups:
                grouped_imp = {}
                grouped_dir = {}
                for gname, feats in feature_groups.items():
                    g_imp = df.loc[df["feature"].isin(feats), "importance"].sum()
                    weights = df.loc[df["feature"].isin(feats), "importance"]
                    g_dir = 0.0
                    if weights.sum() > 0:
                        g_dir = float(
                            np.average(
                                df.loc[df["feature"].isin(feats), "direction"],
                                weights=weights,
                            )
                        )
                    grouped_imp[gname] = g_imp
                    grouped_dir[gname] = g_dir

                remaining = df[~df["feature"].isin(group_members)].copy()
                grouped_df = pd.DataFrame(
                    {
                        "feature": list(grouped_imp.keys()),
                        "importance": list(grouped_imp.values()),
                        "direction": list(grouped_dir.values()),
                    }
                )
                df = pd.concat([grouped_df, remaining], ignore_index=True)

            df["split"] = split
            split_frames[split] = df
            top = df.sort_values("importance", ascending=False).head(max_display)["feature"].tolist()
            top_features.append(set(top))

        features_keep = set().union(*top_features)

        result_frames = []
        for split, df in split_frames.items():
            df_keep = df[df["feature"].isin(features_keep)].copy()
            df_keep = df_keep.sort_values("importance", ascending=False)
            result_frames.append(df_keep)

        summary_df = pd.concat(result_frames, ignore_index=True)
        summary_df = summary_df[["feature", "split", "importance", "direction"]]
        return summary_df

    def _flag_variations(
        self,
        summary_df: pd.DataFrame,
        *,
        reference_split: str | None,
        variation_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Flag features whose importance varies across splits.

        Args:
            summary_df: Output from :meth:`_prepare_shap_summary`.
            reference_split: Baseline split used for comparison. When ``None``
                the mean importance of other splits is used.
            variation_threshold: Relative difference threshold to trigger the
                flag.

        Returns:
            pandas.DataFrame: ``summary_df`` with an extra ``variation_flag``
            column.
        """

        logger = logging.getLogger("riskpilot")
        eps = np.finfo(float).eps

        df = summary_df.copy()

        if reference_split is not None and reference_split not in df["split"].unique():
            logger.warning(
                "reference_split '%s' not found. Falling back to mean logic.",
                reference_split,
            )
            reference_split = None

        if reference_split is not None:
            ref_imp = df[df["split"] == reference_split].set_index("feature")["importance"]
            df = df.merge(ref_imp.rename("ref_imp"), left_on="feature", right_index=True)
            variation = (df["importance"] - df["ref_imp"]).abs() / np.maximum(np.abs(df["ref_imp"]), eps)
        else:
            count = df.groupby("feature")["importance"].transform("count")
            sum_imp = df.groupby("feature")["importance"].transform("sum")
            others_mean = np.where(
                count > 1,
                (sum_imp - df["importance"]) / (count - 1),
                0.0,
            )
            variation = (df["importance"] - others_mean).abs() / np.maximum(np.abs(others_mean), eps)

        df["variation"] = variation
        df["variation_flag"] = variation >= variation_threshold
        return df

    def _make_shap_bullets(
        self,
        df: pd.DataFrame,
        *,
        reference_split: str,
        top_k: int,
        variation_threshold: float,
    ) -> list[str]:
        """Return executive summary bullet points."""

        if df.empty or reference_split not in df["split"].unique():
            return []

        order = df[df["split"] == reference_split].sort_values("importance", ascending=False)["feature"].tolist()
        top_features = order[:top_k]
        bullets: list[str] = []
        eps = np.finfo(float).eps
        for feat in top_features:
            ref_imp = df[(df["feature"] == feat) & (df["split"] == reference_split)]["importance"].iloc[0]
            parts = []
            for split in df["split"].unique():
                if split == reference_split:
                    continue
                row = df[(df["feature"] == feat) & (df["split"] == split)]
                if row.empty:
                    continue
                imp = row["importance"].iloc[0]
                rel = (imp - ref_imp) / (abs(ref_imp) + eps)
                if abs(rel) < variation_threshold:
                    arrow = "↔"
                else:
                    arrow = "↑" if rel > 0 else "↓"
                parts.append(f"{split} {arrow}")
            if parts:
                bullets.append(f"{feat}: " + ", ".join(parts))
            else:
                bullets.append(feat)
        return bullets

    def _build_shap_bar_plot(
        self,
        summary_df: pd.DataFrame,
        *,
        plot_type: Literal["bar", "layered"] = "bar",
        color_palette: list[str] | None = None,
        annotate_variation: bool = True,
        title: str | None = None,
        directionality: bool = False,
    ) -> go.Figure:
        """Return a Plotly bar chart visualising SHAP summary values.

        Parameters
        ----------
        summary_df:
            Output from :meth:`_flag_variations` with columns ``feature``,
            ``split`` and ``importance``. ``direction`` and ``variation`` are
            optional.
        plot_type:
            ``"bar"`` for side-by-side bars or ``"layered"`` for stacked bars.
        color_palette:
            List of colours to use per split. ``None`` falls back to
            ``px.colors.qualitative.Plotly``.
        annotate_variation:
            When ``True`` adds annotations for rows where ``variation_flag`` is
            ``True``.
        title:
            Custom plot title. If ``None`` a default is used.
        directionality:
            If ``True`` colours are lightened when the mean SHAP value direction
            is negative.

        Returns
        -------
        go.Figure
            Fully styled Plotly Figure ready to display or save.

        Examples
        --------
        >>> summary = evaluator._prepare_shap_summary(shap_dict)
        >>> summary = evaluator._flag_variations(summary, reference_split="train")
        >>> fig = evaluator._build_shap_bar_plot(
        ...     summary,
        ...     plot_type="layered",
        ...     color_palette=["#1f77b4", "#ff7f0e"],  # TODO(use COLOR_PRIMARY)
        ...     directionality=True,
        ...     annotate_variation=True,
        ...     title="SHAP – Train vs Test",
        ... )
        >>> fig.show()
        """

        if summary_df.empty:
            raise ValueError("summary_df must not be empty")

        if plot_type not in {"bar", "layered"}:
            raise ValueError("plot_type must be 'bar' or 'layered'")

        # Pivot to get consistent ordering and handle missing splits
        imp_wide = summary_df.pivot(index="feature", columns="split", values="importance").fillna(0)
        dir_wide = summary_df.pivot(index="feature", columns="split", values="direction").fillna(0)

        feature_order = imp_wide.mean(axis=1).sort_values(ascending=False).index.tolist()
        imp_wide = imp_wide.loc[feature_order]
        dir_wide = dir_wide.loc[feature_order]

        splits = list(imp_wide.columns)

        palette = color_palette or px.colors.qualitative.Plotly
        colors = {s: palette[i % len(palette)] for i, s in enumerate(splits)}

        fig = go.Figure()
        barmode = "group" if plot_type == "bar" else "stack"

        def _lighten(color: str, factor: float = 0.6) -> str:
            import matplotlib.colors as mcolors

            r, g, b = mcolors.to_rgb(color)
            r = 1 - (1 - r) * factor
            g = 1 - (1 - g) * factor
            b = 1 - (1 - b) * factor
            return mcolors.to_hex((r, g, b))

        for split in splits:
            values = imp_wide[split].tolist()
            dirs = dir_wide[split].tolist()
            if directionality:
                bar_colors = [_lighten(colors[split]) if d < 0 else colors[split] for d in dirs]
            else:
                bar_colors = colors[split]

            fig.add_bar(
                y=feature_order,
                x=values,
                orientation="h",
                name=split,
                offsetgroup=split,
                marker_color=bar_colors,
            )

        if annotate_variation and "variation_flag" in summary_df.columns:
            flagged = summary_df[summary_df["variation_flag"]].copy()
            flagged = flagged.set_index(["feature", "split"])
            y_shift: dict[str, int] = {}
            for feature in feature_order:
                cum = 0.0
                for split in splits:
                    val = imp_wide.loc[feature, split]
                    if (feature, split) in flagged.index:
                        var_val = float(flagged.loc[(feature, split), "variation"])
                        text = f"Δ {var_val:+.0%}"
                        x_pos = val / 2 if barmode == "group" else cum + val / 2
                        shift = y_shift.get(feature, 0)
                        fig.add_annotation(
                            x=x_pos,
                            y=feature,
                            text=text,
                            showarrow=False,
                            yshift=shift,
                        )
                        y_shift[feature] = shift + 12
                    cum += val

        fig.update_layout(
            template="simple_white",
            title=title or "SHAP Feature Importance",
            barmode=barmode,
            height=min(50 + 30 * len(feature_order), 900),
            margin=dict(l=150, r=40, t=80, b=60),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(autorange="reversed")

        return fig

    def _build_shap_beeswarm(
        self,
        explanation: "shap.Explanation",
        *,
        max_display: int = 20,
    ) -> go.Figure:
        """Return a Plotly beeswarm plot for SHAP values."""

        if shap is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SHAP visualisations need the optional dependency 'shap'. " "Install with `pip install riskpilot[viz]`."
            )

        values = np.asarray(explanation.values)
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0, :]

        feature_names = list(explanation.feature_names)
        df = pd.DataFrame(values, columns=feature_names)
        df_long = df.melt(var_name="feature", value_name="shap_value")

        mean_imp = df.abs().mean().sort_values(ascending=False)
        top = mean_imp.head(max_display).index.tolist()
        df_long = df_long[df_long["feature"].isin(top)]
        df_long["feature"] = pd.Categorical(df_long["feature"], categories=top[::-1], ordered=True)

        fig = px.strip(df_long, x="shap_value", y="feature", orientation="h")
        fig.update_traces(jitter=0.4)
        fig.update_layout(
            template="simple_white",
            title="SHAP Beeswarm",
            height=min(50 + 20 * len(top), 800),
            margin=dict(l=150, r=40, t=80, b=60),
        )
        return fig

    def _build_shap_dependence(
        self,
        explanation: "shap.Explanation",
        feature: str,
        *,
        color_by: str | None = None,
    ) -> go.Figure:
        """Return a Plotly dependence plot for a given feature."""

        if shap is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SHAP visualisations need the optional dependency 'shap'. " "Install with `pip install riskpilot[viz]`."
            )

        values = np.asarray(explanation.values)
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0, :]

        data = pd.DataFrame(explanation.data, columns=explanation.feature_names)
        shap_df = pd.DataFrame(values, columns=explanation.feature_names)

        df = pd.concat([data, shap_df.add_prefix("shap_")], axis=1)

        if feature not in data.columns:
            raise ValueError(f"feature '{feature}' not found in explanation")

        shap_col = f"shap_{feature}"
        color = df[color_by] if color_by else None

        fig = px.scatter(
            df,
            x=feature,
            y=shap_col,
            color=color,
            color_continuous_scale="RdBu",
        )
        fig.update_layout(
            template="simple_white",
            title=f"SHAP Dependence – {feature}",
        )
        return fig

    def _build_shap_waterfall(
        self,
        explanation: "shap.Explanation",
        *,
        index: int = 0,
        max_display: int = 10,
    ) -> go.Figure:
        """Return a Plotly waterfall plot for one observation."""

        if shap is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SHAP visualisations need the optional dependency 'shap'. " "Install with `pip install riskpilot[viz]`."
            )

        values = np.asarray(explanation.values)
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0, :]

        feature_names = list(explanation.feature_names)
        shap_values = values[index]
        base_value = (
            explanation.base_values[index]
            if isinstance(explanation.base_values, (list, np.ndarray))
            else explanation.base_values
        )

        df = pd.DataFrame({"feature": feature_names, "value": shap_values})
        df = df.reindex(df["value"].abs().sort_values(ascending=False).index)
        df = df.head(max_display)

        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["relative"] * len(df) + ["total"],
                x=df["feature"].tolist() + ["base"],
                y=df["value"].tolist() + [base_value],
            )
        )
        fig.update_layout(
            template="simple_white",
            title=f"SHAP Waterfall – index {index}",
            showlegend=False,
        )
        return fig

    def _prepare_shap_time_series(
        self,
        shap_dict: dict[str, "shap.Explanation"],
        *,
        date_lookup: dict[str, pd.Series],
        freq: str = "M",
        feature_groups: dict[str, Sequence[str]] | None = None,
        max_display: int | None = None,
        min_samples: int = 30,
    ) -> pd.DataFrame:
        """Aggregate SHAP values over time windows per split.

        Parameters
        ----------
        shap_dict:
            Mapping of split name to :class:`shap.Explanation` objects.
        date_lookup:
            Mapping of split name to date series aligned with SHAP rows.
        freq:
            Resample frequency, e.g. ``"M"`` or ``"Q"``.
        feature_groups:
            Optional semantic buckets ``{group: [features...]}``.
        max_display:
            If set, keep only the global top ``k`` features.
        min_samples:
            Drop periods with fewer records than this value.

        Returns
        -------
        pandas.DataFrame
            Long-form table with columns ``period``, ``feature``, ``split`` and
            ``importance``.
        """

        feature_groups = feature_groups or {}
        group_members = {f for fs in feature_groups.values() for f in fs}

        all_frames: list[pd.DataFrame] = []

        for split, expl in shap_dict.items():
            if split not in date_lookup:
                raise KeyError(f"Missing dates for split '{split}'")

            dates = date_lookup[split].reset_index(drop=True)
            values = np.asarray(expl.values)
            if values.ndim == 3 and values.shape[1] == 1:
                values = values[:, 0, :]
            if len(dates) != values.shape[0]:
                raise ValueError("Date vector must align with SHAP rows")

            df = pd.DataFrame(values, columns=expl.feature_names).abs()
            df["__period"] = pd.to_datetime(dates).dt.to_period(freq)

            if feature_groups:
                missing = [f for f in group_members if f not in df.columns]
                if missing:
                    raise ValueError(f"Feature '{missing[0]}' specified in groups not found")
                grouped = {g: df[cols].sum(axis=1) for g, cols in feature_groups.items()}
                remaining = df.drop(columns=set(group_members), errors="ignore")
                df = pd.concat([remaining, pd.DataFrame(grouped)], axis=1)

            long_df = df.melt("__period", var_name="feature", value_name="importance")
            agg = long_df.groupby(["__period", "feature"], observed=True).importance.mean()
            agg = agg.reset_index()

            counts = df["__period"].value_counts()
            valid_periods = counts[counts >= min_samples].index
            agg = agg[agg["__period"].isin(valid_periods)]

            agg["split"] = split
            all_frames.append(agg)

        if not all_frames:
            return pd.DataFrame(columns=["period", "feature", "split", "importance"])

        ts_df = pd.concat(all_frames, ignore_index=True)
        ts_df.rename(columns={"__period": "period"}, inplace=True)

        if max_display is not None:
            totals = ts_df.groupby("feature")["importance"].sum()
            keep = totals.nlargest(max_display).index
            ts_df = ts_df[ts_df["feature"].isin(keep)]

        ts_df["period"] = ts_df["period"].astype(f"period[{freq}]")
        return ts_df[["period", "feature", "split", "importance"]]

    def _build_shap_trend_plot(
        self,
        ts_df: pd.DataFrame,
        *,
        feature: str,
        splits: list[str] | None = None,
        color_palette: list[str] | None = None,
        drift_df: pd.DataFrame | None = None,
        reference_split: str | None = None,
        ref_band: float | None = 0.2,
        title: str | None = None,
    ) -> go.Figure:
        """Return line chart of mean |SHAP| over time with optional drift flags.

        Parameters
        ----------
        ts_df:
            Output from :meth:`_prepare_shap_time_series`.
        feature:
            Feature to plot.
        splits:
            Restrict to these splits. ``None`` uses all available.
        color_palette:
            Optional list of colours per split.
        drift_df:
            Optional DataFrame with drift flags (``period``, ``feature``, ``split``,
            ``flag``, ``metric``, ``value``).
        reference_split:
            Split used as baseline for the reference band.
        ref_band:
            ± percentage width around ``reference_split`` line. ``None`` disables.
        title:
            Custom plot title.

        Returns
        -------
        go.Figure
            Styled Plotly figure with one trace per split.

        Raises
        ------
        KeyError
            If ``feature`` is not present in ``ts_df``.
        ValueError
            If ``ts_df`` is empty.

        Examples
        --------
        >>> ts_df = evaluator._prepare_shap_time_series(shap_dict, date_lookup)
        >>> drift = evaluator._psi_flags
        >>> fig = evaluator._build_shap_trend_plot(
        ...     ts_df,
        ...     feature="util_pct",
        ...     splits=["train", "test"],
        ...     drift_df=drift,
        ...     reference_split="train",
        ... )
        >>> fig.show()
        """

        if ts_df.empty:
            raise ValueError("ts_df must not be empty")
        if feature not in ts_df["feature"].unique():
            raise KeyError(f"feature '{feature}' not found")

        df_feat = ts_df[ts_df["feature"] == feature].copy()
        available_splits = list(df_feat["split"].unique())
        splits = splits or available_splits

        palette = color_palette or px.colors.qualitative.Safe
        colors = {s: palette[i % len(palette)] for i, s in enumerate(splits)}

        fig = go.Figure()

        for split in splits:
            df_split = df_feat[df_feat["split"] == split].sort_values("period")
            fig.add_trace(
                go.Scatter(
                    x=df_split["period"].astype(str),
                    y=df_split["importance"],
                    mode="lines+markers",
                    name=split,
                    marker_color=colors[split],
                )
            )

        if reference_split and ref_band is not None and reference_split in splits:
            df_ref = df_feat[df_feat["split"] == reference_split].sort_values("period")
            upper = df_ref["importance"] * (1 + ref_band)
            lower = df_ref["importance"] * (1 - ref_band)
            x = df_ref["period"].astype(str)
            color = _rgba(colors[reference_split], 0.15)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=upper,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=color,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        if drift_df is not None:
            df_drift = drift_df[(drift_df["feature"] == feature) & drift_df["flag"]].copy()
            for split in splits:
                df_split = df_drift[df_drift["split"] == split]
                if df_split.empty:
                    continue
                y_lookup = df_feat[df_feat["split"] == split].set_index("period")["importance"]
                y_vals = y_lookup.reindex(df_split["period"]).values
                hover = [f"{m}: {v:.3f}" for m, v in zip(df_split["metric"], df_split["value"])]
                fig.add_trace(
                    go.Scatter(
                        x=df_split["period"].astype(str),
                        y=y_vals,
                        mode="markers",
                        marker=dict(symbol="x", color=colors[split], size=10),
                        name=f"{split} drift",
                        hovertext=hover,
                        showlegend=False,
                    )
                )

        _style_plotly(fig, title=title or f"SHAP Trend – {feature}")
        return fig

    # ---- Public wrappers ---- #

    def plot_shap_beeswarm(self, split: str = "train", *, max_display: int = 20) -> go.Figure:
        """Convenience wrapper around :meth:`_build_shap_beeswarm`."""

        if shap is None:
            raise RuntimeError("plot_shap_beeswarm requires 'shap'; install riskpilot[viz] to enable.")

        df_split = getattr(self, f"df_{split}")
        expl = self._compute_shap_values(df_split[self.predictor_cols], split_name=split)
        return self._build_shap_beeswarm(expl, max_display=max_display)

    def plot_shap_dependence(
        self,
        feature: str,
        *,
        split: str = "train",
        color_by: str | None = None,
    ) -> go.Figure:
        """Wrapper for :meth:`_build_shap_dependence`."""

        if shap is None:
            raise RuntimeError("plot_shap_dependence requires 'shap'; install riskpilot[viz] to enable.")

        df_split = getattr(self, f"df_{split}")
        expl = self._compute_shap_values(df_split[self.predictor_cols], split_name=split)
        return self._build_shap_dependence(expl, feature=feature, color_by=color_by)

    def plot_shap_waterfall(
        self,
        index: int = 0,
        *,
        split: str = "train",
        max_display: int = 10,
    ) -> go.Figure:
        """Wrapper for :meth:`_build_shap_waterfall`."""

        if shap is None:
            raise RuntimeError("plot_shap_waterfall requires 'shap'; install riskpilot[viz] to enable.")

        df_split = getattr(self, f"df_{split}")
        expl = self._compute_shap_values(df_split[self.predictor_cols], split_name=split)
        return self._build_shap_waterfall(expl, index=index, max_display=max_display)

    def export_report(
        self,
        *,
        figs: go.Figure | Sequence[go.Figure] | Mapping[str, go.Figure],
        summary_df: pd.DataFrame | None = None,
        bullets: list[str] | None = None,
        export_dir: str | Path | None = None,
        formats: Sequence[str] = ("png",),
        name_prefix: str = "shap",
        dpi: int = 150,
        zip_bundle: bool = True,
    ) -> Path:
        """Persist figures and data to disk.

        Returns
        -------
        Path
            Directory containing the exported files.
        """

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base = Path(export_dir or self.save_dir or ".")
        exp_dir = base / f"report_{ts}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        def _save_fig(name: str, fig: go.Figure) -> None:
            for ext in formats:
                fpath = exp_dir / f"{name}.{ext}"
                try:
                    if ext == "html":
                        fig.write_html(fpath, full_html=False, include_plotlyjs="cdn")
                    else:
                        fig.write_image(fpath, scale=dpi / 72)
                except ValueError as err:
                    if "kaleido" in str(err).lower():
                        raise RuntimeError(
                            "Static export requires the 'kaleido' package. Try `pip install kaleido`."
                        ) from err
                    raise

        if isinstance(figs, go.Figure):
            _save_fig("fig_0", figs)
        elif isinstance(figs, Mapping):
            for name, fig in figs.items():
                _save_fig(name, fig)
        else:
            for i, fig in enumerate(figs):
                _save_fig(f"fig_{i}", fig)

        if summary_df is not None:
            summary_df.to_csv(exp_dir / f"{name_prefix}_summary.csv", index=False)
            summary_df.to_html(exp_dir / f"{name_prefix}_summary.html", index=False)

        if bullets:
            (exp_dir / "insights.txt").write_text("\n".join(bullets), encoding="utf-8")

        if zip_bundle:
            try:
                shutil.make_archive(str(exp_dir), "zip", root_dir=exp_dir)
            except Exception:
                pass

        return exp_dir

    def plot_shap(
        self,
        *,
        splits: list[str] | None = None,
        plot_type: Literal[
            "bar",
            "layered",
            "beeswarm",
            "dependence",
            "trend",
            "waterfall",
        ] = "bar",
        reference_split: str | None = "train",
        feature_groups: dict[str, Sequence[str]] | None = None,
        focus_feature: str | None = None,
        record_index: int | None = None,
        variation_threshold: float = 0.50,
        annotate_variation: bool = True,
        summary: bool = False,
        save: bool = False,
        save_format: str | Sequence[str] = "png",
        zip_bundle: bool = False,
        return_data: bool = False,
        **kwargs,
    ) -> go.Figure | list[go.Figure] | dict[str, Any]:
        """High-level SHAP visualisation interface."""

        if shap is None:
            raise RuntimeError("plot_shap requires 'shap'; install riskpilot[viz] to enable.")

        splits = self._infer_splits(splits)
        shap_dict = {
            s: self._compute_shap_values(getattr(self, f"df_{s}")[self.predictor_cols], split_name=s) for s in splits
        }

        max_display = int(kwargs.pop("max_display", 20))
        summary_df = self._prepare_shap_summary(shap_dict, max_display=max_display, feature_groups=feature_groups)
        summary_df = self._flag_variations(
            summary_df,
            reference_split=reference_split,
            variation_threshold=variation_threshold,
        )

        figs: list[go.Figure]
        if plot_type in {"bar", "layered"}:
            params = {k: kwargs.pop(k) for k in ["color_palette", "title", "directionality"] if k in kwargs}
            fig = self._build_shap_bar_plot(
                summary_df,
                plot_type=plot_type,
                annotate_variation=annotate_variation,
                **params,
            )
            figs = [fig]
        elif plot_type == "beeswarm":
            bee_params = {k: kwargs.pop(k) for k in ["max_display"] if k in kwargs}
            figs = [self._build_shap_beeswarm(shap_dict[s], **bee_params) for s in splits]
        elif plot_type == "dependence":
            if not focus_feature:
                raise ValueError("focus_feature is required for plot_type='dependence'")
            dep_params = {k: kwargs.pop(k) for k in ["color_by"] if k in kwargs}
            figs = [self._build_shap_dependence(shap_dict[s], feature=focus_feature, **dep_params) for s in splits]
        elif plot_type == "waterfall":
            if record_index is None:
                raise ValueError("record_index is required for plot_type='waterfall'")
            wf_params = {k: kwargs.pop(k) for k in ["max_display"] if k in kwargs}
            figs = [self._build_shap_waterfall(shap_dict[s], index=record_index, **wf_params) for s in splits]
        elif plot_type == "trend":
            if not focus_feature:
                raise ValueError("focus_feature is required for plot_type='trend'")
            if not self.date_col:
                raise ValueError("date_col must be set for plot_type='trend'")
            date_lookup = {s: getattr(self, f"df_{s}")[self.date_col] for s in splits}
            ts_params = {k: kwargs.pop(k) for k in ["freq", "min_samples"] if k in kwargs}
            ts_df = self._prepare_shap_time_series(
                shap_dict,
                date_lookup=date_lookup,
                feature_groups=feature_groups,
                **ts_params,
            )
            trend_params = {k: kwargs.pop(k) for k in ["color_palette", "drift_df", "ref_band", "title"] if k in kwargs}
            fig = self._build_shap_trend_plot(
                ts_df,
                feature=focus_feature,
                splits=splits,
                reference_split=reference_split,
                **trend_params,
            )
            figs = [fig]
        else:
            raise ValueError("invalid plot_type")

        bullets = (
            self._make_shap_bullets(
                summary_df,
                reference_split=reference_split or "train",
                top_k=max_display,
                variation_threshold=variation_threshold,
            )
            if summary
            else None
        )

        if save:
            self.export_report(
                figs=figs,
                summary_df=summary_df,
                bullets=bullets,
                formats=(save_format,) if isinstance(save_format, str) else save_format,
                zip_bundle=zip_bundle,
            )

        result: go.Figure | list[go.Figure] | dict[str, Any]
        if summary or return_data or save:
            result = {"figures": figs}
            if return_data:
                result["data"] = summary_df
            if summary:
                result["summary"] = bullets or []
            return result

        return figs[0] if len(figs) == 1 else figs
