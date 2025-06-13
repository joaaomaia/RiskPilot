
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
from binary_performance_evaluator import BinaryPerformanceEvaluator

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
evaluator.plot_psi()
evaluator.plot_ks()

print(evaluator.report)           # full dict of results
"""

from __future__ import annotations
import warnings
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from optbinning import OptimalBinning
from decile_plot import decile_analysis_plot
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
)

sns.set(style='whitegrid')  # consistent style throughout

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
        Categorical column for group analyses.
    save_dir : Optional[str|Path], default=None
        If provided, figures are saved to this directory in PNG format.
    threshold : float, default 0.5
        Probability cutoff used to convert scores into class labels.
    homogeneous_group : str | int | pd.Series | np.ndarray | None, default "auto"
        Strategy to create homogeneous groups. See :meth:`plot_group_radar`.

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
        homogeneous_group: Optional[Union[str, int, pd.Series, np.ndarray]] = "auto",
    ) -> None:
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

        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

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
        self.group_col_ = "homogeneous_group"
        self.group_: Dict[str, pd.Series] | None = None
        self.binning_table_: Any | None = None

        self._score_datasets()
        self._assign_groups()

    ## ---------- public API ----------
    def compute_metrics(self) -> None:
        """Compute numeric metrics for train, test and (optional) validation."""
        self._validate_predictors()
        splits = {'train': self.df_train, 'test': self.df_test}
        if self.df_val is not None:
            splits['val'] = self.df_val

        for split_name, df in splits.items():
            y_true = df[self.target_col].values
            y_pred_proba = df[self.score_col_].values

            metrics_dict = {
                'MCC': matthews_corrcoef(y_true, (y_pred_proba >= self.threshold).astype(int)),
                'AUC_ROC': roc_auc_score(y_true, y_pred_proba),
                'AUC_PR': average_precision_score(y_true, y_pred_proba),
                'Precision': precision_score(y_true, (y_pred_proba >= self.threshold).astype(int)),
                'Recall': recall_score(y_true, (y_pred_proba >= self.threshold).astype(int)),
                'Brier': brier_score_loss(y_true, y_pred_proba),
            }
            self.report[split_name] = metrics_dict

    def plot_confusion(self, *, save: bool = False) -> None:
        """Plot confusion matrices (absolute + %)."""
        if not self.report:
            warnings.warn('compute_metrics() has not been called; metrics may be missing.')
        self._validate_predictors()

        fig, axes = plt.subplots(1, 3 if self.df_val is not None else 2, figsize=(18, 5))
        split_dfs = {'Train': self.df_train, 'Test': self.df_test}
        if self.df_val is not None:
            split_dfs['Val'] = self.df_val

        for ax, (title, df) in zip(axes, split_dfs.items()):
            y_true = df[self.target_col]
            y_pred = df[self.label_col_]
            cm = confusion_matrix(y_true, y_pred)
            cm_perc = cm / cm.sum()

            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xticklabels(['0', '1'])
            ax.set_yticklabels(['0', '1'], rotation=0)

            # annotate abs + %
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    abs_val = cm[i, j]
                    perc_val = cm_perc[i, j]
                    text_color = 'white' if cm_perc[i, j] > 0.35 else 'black'
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f'{abs_val}\n{perc_val:.1%}',
                        ha='center',
                        va='center',
                        color=text_color,
                        fontsize=10,
                        fontweight='bold',
                    )
            ax.set_title(f'{title} Confusion Matrix', fontsize=12)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')

        fig.tight_layout()
        if save and self.save_dir:
            plt.savefig(self.save_dir / 'confusion_matrices.png', dpi=200, bbox_inches='tight')
        plt.show()

    def plot_calibration(self, *, n_bins: int = 10, save: bool = False) -> go.Figure:
        """Reliability diagram for test split using Plotly."""
        self._validate_predictors()
        y_true = self.df_test[self.target_col].values
        y_pred_proba = self.model.predict_proba(self.df_test[self.predictor_cols])[:, 1]
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_pred_proba,
            n_bins=n_bins,
            strategy='uniform',
        )

        brier = brier_score_loss(y_true, y_pred_proba)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Model'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Ideal'))
        fig.update_layout(
            title=f'Calibration Curve – Test (Brier = {brier:.4f})',
            xaxis_title='Predicted probability',
            yaxis_title='Observed frequency',
            template='plotly_white',
        )

        if save and self.save_dir:
            fig.write_image(str(self.save_dir / 'calibration_curve.png'))
        return fig

    def plot_event_rate(self, *, save: bool = False) -> go.Figure:
        """Trend of event (target=1) rate over time by group using Plotly."""
        if self.date_col is None or self.group_col is None:
            raise ValueError('Both `date_col` and `group_col` are required for plot_event_rate().')

        df_all = pd.concat(
            [
                self.df_train.assign(Split='Train'),
                self.df_test.assign(Split='Test'),
                *( [self.df_val.assign(Split='Val')] if self.df_val is not None else [] )
            ],
            axis=0,
        )
        df_all[self.date_col] = pd.to_datetime(df_all[self.date_col])

        pivot = (
            df_all.groupby([self.date_col, self.group_col])[self.target_col]
            .mean()
            .unstack(self.group_col)
            .sort_index()
        )

        fig = go.Figure()
        for col in pivot.columns:
            fig.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[col],
                    mode='lines+markers',
                    name=str(col),
                )
            )
        fig.update_layout(
            title='Event Rate by Group over Time',
            xaxis_title=self.date_col,
            yaxis_title='Event rate',
            template='plotly_white',
        )
        if save and self.save_dir:
            fig.write_image(str(self.save_dir / 'event_rate.png'))
        return fig

    def plot_psi(self, *, bins: int = 10, save: bool = False) -> go.Figure:
        """Compute & plot PSI per variable through time using Plotly."""
        if self.date_col is None:
            raise ValueError('`date_col` is required for plot_psi().')

        date_series = pd.to_datetime(self.df_test[self.date_col]).dt.to_period('M')
        unique_periods = sorted(date_series.unique())

        psi_records = []
        baseline_df = self.df_train
        for var in self._psi_variables():
            if not pd.api.types.is_numeric_dtype(baseline_df[var]):
                continue

            base_series = pd.to_numeric(baseline_df[var], errors="coerce").dropna()
            if base_series.empty:
                continue

            baseline_counts, _ = np.histogram(
                base_series, bins=bins, range=(base_series.min(), base_series.max())
            )
            baseline_pct = baseline_counts / baseline_counts.sum()

            for period in unique_periods:
                subset = self.df_test[date_series == period]
                if subset.empty:
                    continue
                sub_series = pd.to_numeric(subset[var], errors="coerce").dropna()
                if sub_series.empty:
                    counts = np.zeros(bins)
                else:
                    counts, _ = np.histogram(
                        sub_series, bins=bins, range=(base_series.min(), base_series.max())
                    )
                pct = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
                psi = np.where(
                    (baseline_pct > 0) & (pct > 0),
                    (pct - baseline_pct) * np.log(pct / baseline_pct),
                    0,
                ).sum()
                psi_records.append({'Variable': var, 'Period': period.to_timestamp(), 'PSI': psi})

        psi_df = pd.DataFrame(psi_records)
        if psi_df.empty:
            warnings.warn('PSI could not be computed (insufficient data).')
            return

        fig = go.Figure()
        for var, grp in psi_df.groupby('Variable'):
            fig.add_trace(
                go.Scatter(
                    x=grp['Period'],
                    y=grp['PSI'],
                    mode='lines+markers',
                    name=str(var),
                )
            )
        fig.add_hline(y=0.1, line=dict(color='orange', dash='dash'), annotation_text='0.10')
        fig.add_hline(y=0.25, line=dict(color='red', dash='dash'), annotation_text='0.25')
        fig.update_layout(
            title='PSI over Time by Variable',
            xaxis_title='Period',
            yaxis_title='Population Stability Index',
            template='plotly_white',
        )
        if save and self.save_dir:
            fig.write_image(str(self.save_dir / 'psi_over_time.png'))
        return fig

    def plot_ks(self, *, save: bool = False) -> go.Figure:
        """KS statistic over time for each split using Plotly."""
        if self.date_col is None:
            raise ValueError('`date_col` is required for plot_ks().')
        self._validate_predictors()

        def ks_stat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            return np.max(np.abs(tpr - fpr))

        dfs = [
            ('Train', self.df_train),
            ('Test', self.df_test),
            *([('Val', self.df_val)] if self.df_val is not None else []),
        ]

        ks_records = []
        for split_name, df in dfs:
            df = df.copy()
            df['Period'] = pd.to_datetime(df[self.date_col]).dt.to_period('M')
            for period, grp in df.groupby('Period'):
                y_true = grp[self.target_col].values
                y_pred = self.model.predict_proba(grp[self.predictor_cols])[:, 1]
                ks = ks_stat(y_true, y_pred)
                ks_records.append({'Split': split_name, 'Period': period.to_timestamp(), 'KS': ks})

        ks_df = pd.DataFrame(ks_records)
        if ks_df.empty:
            warnings.warn('KS could not be computed (insufficient data).')
            return

        fig = go.Figure()
        for split, grp in ks_df.groupby('Split'):
            fig.add_trace(
                go.Scatter(
                    x=grp['Period'],
                    y=grp['KS'],
                    mode='lines+markers',
                    name=str(split),
                )
            )
        fig.add_hline(y=0.05, line=dict(color='gray', dash='dash'), annotation_text='0.05')
        fig.update_layout(
            title='KS Evolution over Time',
            xaxis_title='Period',
            yaxis_title='Kolmogorov–Smirnov',
            template='plotly_white',
        )
        if save and self.save_dir:
            fig.write_image(str(self.save_dir / 'ks_evolution.png'))
        return fig

    def plot_group_radar(
        self,
        features: List[str] | None = None,
        *,
        scaler: Literal["zscore", "minmax"] = "zscore",
        save: bool = False,
    ) -> go.Figure:
        """Return radar chart of average feature values per homogeneous group."""
        if self.group_ is None:
            raise ValueError("Homogeneous groups were not computed.")

        df = self.data_.copy()
        if features is None:
            numeric_predictors = [
                c for c in self.predictor_cols if pd.api.types.is_numeric_dtype(df[c])
            ]
            features = numeric_predictors
        if not features:
            raise ValueError("No numeric features available for radar plot.")

        if scaler == "zscore":
            scaled = df[features].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
        else:
            scaled = df[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        mean_by_group = scaled.groupby(df[self.group_col_])[features].mean()

        fig = go.Figure()
        for group_id, row in mean_by_group.iterrows():
            fig.add_trace(
                go.Scatterpolar(
                    r=row.values,
                    theta=features,
                    fill="toself",
                    name=f"Group {group_id}",
                )
            )

        fig.update_layout(template="plotly_white")
        if save and self.save_dir:
            fig.write_image(str(self.save_dir / "group_radar.png"))
        return fig

    def plot_decile_ks(
        self,
        *,
        n_bins: int = 10,
        ascending: bool = True,
        group_id: int | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Wrapper around :func:`decile_analysis_plot` respecting groups."""
        df = self.data_.copy()
        if group_id is not None:
            if self.group_ is None:
                raise ValueError("Homogeneous groups were not computed.")
            df = df[df[self.group_col_] == group_id]

        fig, _, _ = decile_analysis_plot(
            df,
            score_col=self.score_col_,
            target_col=self.target_col,
            n_bins=n_bins,
            ascending=ascending,
            **kwargs,
        )
        return fig

    ## ---------- helpers ----------
    def _load_model(self, model: Union[str, Path, object]):
        """Load model from path or return object as‑is."""
        if isinstance(model, (str, Path)):
            model_path = Path(model)
            if model_path.suffix in {'.joblib', '.jbl'}:
                return joblib.load(model_path)
            elif model_path.suffix in {'.pkl', '.pickle'}:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f'Unsupported model file extension: {model_path.suffix}')
        else:
            # assume object is already a fitted estimator
            if not hasattr(model, 'predict_proba'):
                raise AttributeError('Provided model object lacks predict_proba.')
            return model

    def _validate_data(self) -> None:
        """Basic dataframe validations."""
        for name, df in [('df_train', self.df_train), ('df_test', self.df_test), ('df_val', self.df_val)]:
            if df is None:
                continue
            if self.target_col not in df.columns:
                raise KeyError(f'{self.target_col} missing in {name}.')
            if df[self.target_col].isna().any():
                raise ValueError(f'NaN detected in target column of {name}.')

            missing_ids = [col for col in self.id_cols if col not in df.columns]
            if missing_ids:
                raise KeyError(f'{name} missing id_cols: {missing_ids}')

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
            raise ValueError('No predictor columns detected after exclusions.')
        return predictor_cols

    def _get_model_feature_names(self) -> Optional[List[str]]:
        """Return feature names used during model training, if available."""
        if hasattr(self.model, 'feature_names_in_'):
            return list(getattr(self.model, 'feature_names_in_'))
        try:
            booster = self.model.get_booster()
            if hasattr(booster, 'feature_names'):
                return list(booster.feature_names)
        except Exception:
            pass
        return None

    def _get_model_n_features(self) -> Optional[int]:
        """Return the number of features the model expects, if available."""
        if hasattr(self.model, 'n_features_in_'):
            return int(getattr(self.model, 'n_features_in_'))
        try:
            booster = self.model.get_booster()
            if hasattr(booster, 'num_features'):
                return int(booster.num_features())
        except Exception:
            pass
        return None

    def _align_predictors_with_model(self, cols: List[str]) -> List[str]:
        """Ensure predictors match model's training features, preserving order."""
        if self.model_feature_names:
            missing = [c for c in self.model_feature_names if c not in cols]
            if missing:
                raise ValueError(
                    f'Model expects columns not present in provided data: {missing}'
                )
            return [c for c in self.model_feature_names if c in cols]

        if self.model_n_features is not None and len(cols) != self.model_n_features:
            raise ValueError(
                f'Number of predictor columns ({len(cols)}) does not match model '
                f'expectation ({self.model_n_features}).'
            )
        return cols

    def _validate_predictors(self) -> None:
        """Ensure predictor columns align with model expectations."""
        for name, df in [('df_train', self.df_train), ('df_test', self.df_test), ('df_val', self.df_val)]:
            if df is None:
                continue
            missing = [c for c in self.predictor_cols if c not in df.columns]
            if missing:
                raise KeyError(f'{name} missing predictor columns: {missing}')

        if self.model_feature_names:
            ordered = [c for c in self.model_feature_names if c in self.predictor_cols]
            if ordered != self.predictor_cols:
                self.predictor_cols = ordered
        elif self.model_n_features is not None and len(self.predictor_cols) != self.model_n_features:
            raise ValueError(
                f'Model expects {self.model_n_features} features, got {len(self.predictor_cols)}'
            )

    def _score_datasets(self) -> None:
        """Add predicted probabilities and labels to each split."""
        dfs = [('train', self.df_train), ('test', self.df_test)]
        if self.df_val is not None:
            dfs.append(('val', self.df_val))

        for name, df in dfs:
            proba = self.model.predict_proba(df[self.predictor_cols])[:, 1]
            df[self.score_col_] = proba
            df[self.label_col_] = (proba >= self.threshold).astype(int)
            df['Split'] = name.capitalize()

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
            if self.homogeneous_group != 'auto':
                raise ValueError("Unsupported string for homogeneous_group")

            optb = OptimalBinning(
                name='y_proba_train',
                dtype='numerical',
                solver='mip',
                min_prebin_size=0.01,
                max_n_bins=5,
                min_bin_size=0.05,
                monotonic_trend='ascending',
            )
            optb.fit(self.df_train[self.score_col_] * 1000, self.df_train[self.target_col])
            self.binning_table_ = optb.binning_table.build()

            for name, df in [('train', self.df_train), ('test', self.df_test), ('val', self.df_val)]:
                if df is None:
                    continue
                labels = optb.transform(df[self.score_col_] * 1000, metric='bins')
                df[self.group_col_] = labels
                self.group_[name] = labels

        elif isinstance(self.homogeneous_group, int):
            n = int(self.homogeneous_group)
            for name, df in [('train', self.df_train), ('test', self.df_test), ('val', self.df_val)]:
                if df is None:
                    continue
                bins = pd.qcut(
                    df[self.score_col_].rank(method='first'),
                    q=n,
                    labels=range(1, n + 1),
                ).astype(int)
                df[self.group_col_] = bins
                self.group_[name] = bins
            self.binning_table_ = None

        else:
            groups = pd.Series(self.homogeneous_group)
            if len(groups) != len(self.data_):
                raise ValueError('Length of provided group labels does not match data.')

            self.data_[self.group_col_] = groups.reset_index(drop=True)
            start = 0
            for name, df in [('train', self.df_train), ('test', self.df_test), ('val', self.df_val)]:
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


    def _psi_variables(self) -> List[str]:
        """Select variables to evaluate for PSI (exclude id/date/target)."""
        exclude = set(self.id_cols + [self.target_col])
        if self.date_col:
            exclude.add(self.date_col)
        vars_ = [c for c in self.df_train.columns if c not in exclude]
        numeric_vars = [v for v in vars_ if pd.api.types.is_numeric_dtype(self.df_train[v])]
        return numeric_vars
