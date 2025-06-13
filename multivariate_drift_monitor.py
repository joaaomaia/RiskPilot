# multivariate_drift_monitor.py
"""
MultivariateDriftMonitor
========================
Monitoramento multivariado de deriva em produção via erro de reconstrução.

Métodos suportados
------------------
- Autoencoder (não‑linear, Keras)
- Kernel PCA   (linear ou kernelizado)

O monitor:
1. Ajusta um *DynamicScaler* por coluna numérica.
2. Treina o modelo de compressão/reconstrução no dataset de referência (`fit`).
3. Calcula o erro de reconstrução em novos lotes e sinaliza deriva quando
   o erro excede ``mean + k·std`` do erro de treinamento (`score`).
4. Gera gráficos interativos em Plotly (`plot_drift`) e um relatório JSON
   consolidado (`get_report`).

Requisitos
----------
```
pip install tensorflow scikit-learn pandas numpy plotly joblib
```

Exemplo rápido
--------------
```python
monitor = MultivariateDriftMonitor(
    features_cols=feat_cols,
    date_col='date',
    method='autoencoder',          # ou 'kpca'
    variance_retained=0.65,
    alert_sigma=3.0,
)
monitor.fit(df_train)
df_scores = monitor.score(df_test)
monitor.plot_drift()
report = monitor.get_report()
```
"""

from __future__ import annotations

import json
import logging
import pathlib
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import KernelPCA, PCA
from sklearn.utils.validation import check_is_fitted

import plotly.graph_objects as go

try:
    # Keras pode não estar instalado — falhar graciosamente
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
except Exception as e:  # pragma: no cover
    keras = None
    layers = None

try:
    from .dynamic_scaler import DynamicScaler  # type: ignore
except ImportError:  # monolítico – definido no mesmo arquivo
    from dynamic_scaler import DynamicScaler  # noqa


class MultivariateDriftMonitor:
    """
    Parâmetros principais
    ---------------------
    features_cols : list[str]
        Colunas numéricas a monitorar.

    date_col : str
        Nome da coluna temporal (qualquer tipo compatível com Pandas).

    method : {'autoencoder', 'kpca'}
        Técnica de compressão/reconstrução.

    variance_retained : float, default depende do método
        Fração desejada da variância original representada no espaço latente.
        - Autoencoder → define a dimensão latente = ceil(variance * n_features)
        - (K)PCA      → `n_components` aproximado = ceil(variance * n_features)

    alert_sigma : float, default 3.0
        Número de desvios‑padrão acima da média do erro base para disparar alerta.

    hidden_layers : list[int] | None
        Arquitetura escondida do Autoencoder. Se None, uma arquitetura simples
        decrescente é construída automaticamente.

    scaler_strategy : str, default 'auto'
        Estratégia para o **DynamicScaler** (ver classe DynamicScaler).

    rolling_window : int | None
        Se informado, calcula métricas em janela móvel (n observações).
    """

    _METHODS = {'autoencoder', 'kpca'}

    def __init__(
        self,
        *,
        features_cols: List[str],
        date_col: str,
        method: str = 'autoencoder',
        variance_retained: Optional[float] = None,
        alert_sigma: float = 3.0,
        hidden_layers: Optional[List[int]] = None,
        scaler_strategy: str = 'auto',
        rolling_window: Optional[int] = None,
        random_state: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if method not in self._METHODS:
            raise ValueError(f"method deve ser um de {self._METHODS}")

        self.features_cols = features_cols
        self.date_col = date_col
        self.method = method
        self.variance_retained = (
            variance_retained if variance_retained is not None
            else (0.65 if method == 'autoencoder' else 0.70)
        )
        self.alert_sigma = alert_sigma
        self.hidden_layers = hidden_layers
        self.scaler_strategy = scaler_strategy
        self.rolling_window = rolling_window
        self.random_state = random_state

        self.scaler_: DynamicScaler | None = None
        self.model_: Any = None  # keras.Model | KernelPCA
        self.err_mean_: float | None = None
        self.err_std_: float | None = None
        self.err_threshold_: float | None = None
        self.history_: pd.DataFrame | None = None  # erros já avaliados

        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s - %(message)s")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, df_base: pd.DataFrame) -> "MultivariateDriftMonitor":
        """
        Ajusta scaler + modelo no dataset de referência.
        """
        self.logger.info("Fit: iniciando scaler (%s)", self.scaler_strategy)
        self.scaler_ = DynamicScaler(strategy=self.scaler_strategy,
                                     random_state=self.random_state)
        X_base = df_base[self.features_cols]
        self.scaler_.fit(X_base)
        X_scaled = self.scaler_.transform(X_base, return_df=False, keep_other_cols=False)

        self.logger.info("Fit: treinando modelo (%s)", self.method)
        if self.method == 'kpca':
            n_comp = max(1, int(np.ceil(self.variance_retained * X_scaled.shape[1])))
            kpca = KernelPCA(
                n_components=n_comp,
                kernel='rbf',
                fit_inverse_transform=True,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model_ = kpca.fit(X_scaled)
        else:  # autoencoder
            if keras is None:
                raise ImportError("TensorFlow/Keras não instalados para Autoencoder.")
            latent_dim = max(1, int(np.ceil(self.variance_retained * X_scaled.shape[1])))
            input_dim = X_scaled.shape[1]
            hlayers = self._build_hidden_layers(input_dim, latent_dim)
            self.model_ = self._build_autoencoder(input_dim, latent_dim, hlayers)

            self.model_.compile(optimizer='adam', loss='mse')
            self.model_.fit(X_scaled, X_scaled,
                            epochs=50,
                            batch_size=256,
                            verbose=0)

        # Erro de reconstrução no base
        base_err = self._reconstruction_error(X_scaled)
        self.err_mean_ = float(base_err.mean())
        self.err_std_ = float(base_err.std())
        self.err_threshold_ = self.err_mean_ + self.alert_sigma * self.err_std_

        self.logger.info(
            "Erro médio base %.5f (std %.5f) | Threshold = %.5f",
            self.err_mean_, self.err_std_, self.err_threshold_
        )

        # Histórico inicial
        self.history_ = pd.DataFrame({
            self.date_col: pd.to_datetime(df_base[self.date_col]),
            'recon_error': base_err,
            'is_drift': base_err > self.err_threshold_,
            'origin': 'base',
        })

        return self

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def score(self, df_new: pd.DataFrame, *, update_history: bool = True) -> pd.DataFrame:
        """
        Calcula erro de reconstrução em novos dados e retorna DataFrame
        com métricas agregadas por período (mesmo índice que df_new).

        Se ``update_history=True``, concatena resultados ao histórico interno.
        """
        self._check_is_fitted()

        X_new = df_new[self.features_cols]
        X_scaled = self.scaler_.transform(X_new, return_df=False, keep_other_cols=False)
        rec_err = self._reconstruction_error(X_scaled)

        df_res = pd.DataFrame({
            self.date_col: pd.to_datetime(df_new[self.date_col]),
            'recon_error': rec_err,
            'is_drift': rec_err > self.err_threshold_,
            'origin': 'new',
        })

        if update_history:
            self.history_ = pd.concat([self.history_, df_res], ignore_index=True)

        return df_res

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot_drift(self, *, show: bool = True, **kwargs) -> go.Figure:
        """
        Gera gráfico Plotly com erros no tempo + faixa de alerta.
        """
        self._check_is_fitted()
        df = self.history_.copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[self.date_col],
            y=df['recon_error'],
            mode='markers',
            name='Recon. error',
            marker=dict(
                color=np.where(df['is_drift'], 'crimson', 'steelblue'),
                size=6,
                opacity=0.7))
        )
        fig.add_hline(
            y=self.err_threshold_,
            line=dict(dash='dash'),
            annotation_text=f"Threshold ({self.alert_sigma}σ)",
            annotation_position="top left"
        )
        fig.update_layout(
            title="Multivariate Drift Monitor — Reconstruction Error",
            xaxis_title=self.date_col,
            yaxis_title="Erro de reconstrução",
            template="plotly_white",
            **kwargs,
        )
        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def get_report(self) -> Dict[str, Any]:
        """Retorna relatório consolidado."""
        self._check_is_fitted()

        df = self.history_.copy()
        if self.rolling_window:
            df['rolling_rmse'] = (
                df['recon_error']
                .rolling(self.rolling_window, min_periods=1)
                .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
            )

        grp = df.groupby(pd.to_datetime(df[self.date_col]).dt.date)['recon_error']
        metrics = pd.DataFrame({
            'RMSE': grp.apply(lambda x: np.sqrt(np.mean(x**2))),
            'MAE': grp.apply(lambda x: np.mean(np.abs(x))),
            'MSE': grp.apply(lambda x: np.mean(x**2)),
        })

        return {
            'params': {
                'method': self.method,
                'variance_retained': self.variance_retained,
                'alert_sigma': self.alert_sigma,
                'hidden_layers': self.hidden_layers,
                'rolling_window': self.rolling_window,
            },
            'base_error': {
                'mean': self.err_mean_,
                'std': self.err_std_,
                'threshold': self.err_threshold_,
            },
            'metrics': metrics.to_dict(orient='index'),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _build_hidden_layers(self, input_dim: int, latent_dim: int) -> List[int]:
        if self.hidden_layers:
            return self.hidden_layers
        # Construção piramidal simples
        hlayers = []
        dim = input_dim
        while dim // 2 > latent_dim:
            dim = dim // 2
            hlayers.append(dim)
        return hlayers

    def _build_autoencoder(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layers: List[int],
    ):
        input_layer = layers.Input(shape=(input_dim,))
        x = input_layer
        for h in hidden_layers:
            x = layers.Dense(h, activation='relu')(x)

        latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)

        y = latent
        for h in reversed(hidden_layers):
            y = layers.Dense(h, activation='relu')(y)

        output
