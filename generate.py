from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class _VarMeta:
    dtype: str                    # 'cont', 'disc', 'cat', 'bool', 'date'
    categories: Optional[np.ndarray] = None
    quantiles: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None    # para PMF discreta


class SyntheticVintageGenerator:
    """
    Gera safras sintéticas de contratos de empréstimo para testes de stress-PD.
    """

    def __init__(
        self,
        id_cols: Sequence[str],
        date_cols: Sequence[str],
        keep_cols: Optional[Sequence[str]] = None,
        ignore_cols: Optional[Sequence[str]] = None,
        random_state: int | None = None,
        custom_noise: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.id_cols = set(id_cols)
        self.date_cols = set(date_cols)
        self.keep_cols = set(keep_cols or [])
        self.ignore_cols = set(ignore_cols or [])
        self.random_state = np.random.RandomState(random_state)
        self.custom_noise = custom_noise or {}

        self._meta: Dict[str, _VarMeta] = {}
        self._corr: Optional[np.ndarray] = None
        self._order: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "SyntheticVintageGenerator":
        """
        Aprende o *shape* estatístico do dataframe histórico.

        1. Detecta tipo de cada coluna (numérica contínua, discreta, categórica…)
        2. Armazena QFs (quantile functions) ou PMFs.
        3. Calcula correlação de postos (Spearman) para cópula gaussiana.
        """
        self._order = [c for c in df.columns
                       if c not in self.ignore_cols and c not in self.date_cols]
        for col in self._order:
            s = df[col].dropna()
            if col in self.id_cols:
                continue  # não gera, só cria novos IDs
            if col in self.date_cols:
                self._meta[col] = _VarMeta("date")
            elif pd.api.types.is_bool_dtype(s):
                self._meta[col] = _VarMeta("bool", values=s.values)
            elif pd.api.types.is_numeric_dtype(s):
                uniq = np.unique(s)
                if len(uniq) < 20:
                    self._meta[col] = _VarMeta("disc", values=uniq,
                                               categories=np.bincount(s.astype(int)))
                else:
                    qs = np.quantile(s, np.linspace(0, 1, 1001))
                    self._meta[col] = _VarMeta("cont", quantiles=qs)
            else:  # categórico
                cats, counts = np.unique(s.astype(str), return_counts=True)
                self._meta[col] = _VarMeta("cat", categories=cats,
                                           values=counts / counts.sum())

        # correlações (apenas variáveis geradas)
        cont_cols = [c for c, m in self._meta.items() if m.dtype in {"cont", "disc"}]
        if len(cont_cols) >= 2:
            rank_df = df[cont_cols].rank(method="average") / (len(df) + 1)
            self._corr = rank_df.corr(method="spearman").values
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # GERAÇÃO
    # ------------------------------------------------------------------
    def generate(
        self,
        n_periods: int,
        freq: str = "M",
        start_vintage: str | pd.Timestamp | None = None,
        scenario: str = "base",
        shocks: Optional[Dict[str, Dict[str, Any]]] = None,
        n_per_vintage: Optional[int] = None,
        preserve_corr: bool = True,
    ) -> pd.DataFrame:
        """
        Produz um dataframe concatenado com `n_periods` safras.

        Parameters
        ----------
        n_periods : int
            Número de períodos futuros.
        freq : str
            Frequência de cada safra (ex.: "M", "Q", "A").
        start_vintage : str | pd.Timestamp | None
            Primeiro período; default = mês seguinte ao último da amostra.
        scenario : {"base", "stress"}
            Controla intensidade do ruído.
        shocks : dict, opcional
            Choques direcionados por coluna.
        n_per_vintage : int, opcional
            Tamanho fixo; se None → bootstrap do histórico.
        preserve_corr : bool
            Usa cópula gaussiana para manter correlações numéricas.
        """
        if not self._fitted:
            raise RuntimeError("Execute .fit() antes de .generate().")

        shocks = shocks or {}
        df_list: List[pd.DataFrame] = []

        # tabela de datas-safra
        if start_vintage is None:
            start_vintage = pd.Timestamp.today().normalize() + pd.tseries.frequencies.to_offset(freq)
        vint_dates = pd.date_range(start=start_vintage, periods=n_periods, freq=freq)

        for vint in vint_dates:
            n_rows = n_per_vintage or self.random_state.choice([500, 1000, 2000])
            df_v = self._generate_one_vintage(n_rows, vint, scenario,
                                              shocks, preserve_corr)
            df_list.append(df_v)

        return pd.concat(df_list, ignore_index=True)

    # ------------------------------------------------------------------
    # AUXILIARES DE UMA SAFRA
    # ------------------------------------------------------------------
    def _generate_one_vintage(
        self,
        n_rows: int,
        vint_date: pd.Timestamp,
        scenario: str,
        shocks: Dict[str, Dict[str, Any]],
        preserve_corr: bool,
    ) -> pd.DataFrame:
        # 1. gera base vazia
        data = {}
        for col in self.id_cols:
            data[col] = [uuid.uuid4().hex for _ in range(n_rows)]
        for col in self.date_cols:
            data[col] = np.full(n_rows, vint_date)

        # 2. gera numéricas/categóricas independentes
        indep_vars = {}
        for col, meta in self._meta.items():
            indep_vars[col] = self._sample_marginal(col, meta, n_rows, scenario)

        df_num = pd.DataFrame(indep_vars)

        # 3. aplica cópula (só numéricas) se desejado
        if preserve_corr and self._corr is not None:
            df_num = self._apply_copula(df_num)

        # 4. une tudo
        df_synth = pd.concat([pd.DataFrame(data), df_num], axis=1)

        # 5. choques
        df_synth = self._apply_shocks(df_synth, shocks)

        # 6. mantém keep_cols originais como NaN (ou valor default)
        for col in self.keep_cols:
            if col not in df_synth.columns:
                df_synth[col] = np.nan

        return df_synth

    # ------------------------------------------------------------------
    # AMOSTRAGEM DE UMA MARGINAL
    # ------------------------------------------------------------------
    def _sample_marginal(
        self,
        col: str,
        meta: _VarMeta,
        n: int,
        scenario: str,
    ) -> np.ndarray:
        """Retorna vetor sintético seguindo regra de ruído."""
        cust = self.custom_noise.get(col)
        if cust is not None:
            return cust["func"](n, **cust.get("kwargs", {}))

        if meta.dtype == "cont":
            u = self.random_state.uniform(0, 1, size=n)
            synt = np.interp(u, np.linspace(0, 1, len(meta.quantiles)), meta.quantiles)
            sigma = synt.std(ddof=0)
            mu_shift = 0.2 * sigma if scenario == "stress" else 0.0
            noise = self.random_state.normal(mu_shift, 0.2 * sigma, size=n)
            return synt + noise

        if meta.dtype == "disc":
            probs = meta.categories / meta.categories.sum()
            synt = self.random_state.choice(meta.values, size=n, p=probs)
            jitter = self.random_state.poisson(1, size=n) - 1  # média 0
            if scenario == "stress":
                jitter *= 2
            return np.maximum(0, synt + jitter)

        if meta.dtype == "cat":
            synt = self.random_state.choice(meta.categories, size=n, p=meta.values)
            if scenario == "stress":
                # amplifica menor categoria
                min_idx = np.argmin(meta.values)
                flip_mask = self.random_state.rand(n) < 0.05
                synt[flip_mask] = meta.categories[min_idx]
            return synt

        if meta.dtype == "bool":
            flip_p = 0.10 if scenario == "stress" else 0.05
            base = self.random_state.choice(meta.values, size=n)
            flips = self.random_state.rand(n) < flip_p
            return np.where(flips, ~base, base)

        if meta.dtype == "date":
            # usa distribuição de delays históricas (dias entre data_col e vint_date)
            delays = self.random_state.choice(np.arange(-30, 31), size=n)
            return pd.to_datetime(vint_date) + pd.to_timedelta(delays, unit="D")

        raise ValueError(f"Tipo desconhecido para {col}")

    # ------------------------------------------------------------------
    # CÓPULA GAUSSIANA
    # ------------------------------------------------------------------
    def _apply_copula(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df_raw.columns if self._meta[c].dtype in {"cont", "disc"}]
        if len(cols) < 2:
            return df_raw

        n = len(df_raw)
        # gera amostra normal multivariada
        L = np.linalg.cholesky(self._corr + 1e-6 * np.eye(len(cols)))
        z = self.random_state.normal(size=(n, len(cols)))
        x = z @ L.T  # correlação desejada

        # converte para uniformes
        u = stats.norm.cdf(x)

        # aplica inversa empírica para cada coluna
        df_new = df_raw.copy()
        for j, col in enumerate(cols):
            meta = self._meta[col]
            if meta.dtype == "cont":
                df_new[col] = np.interp(u[:, j],
                                        np.linspace(0, 1, len(meta.quantiles)),
                                        meta.quantiles)
            else:  # disc
                bins = np.cumsum(meta.categories) / meta.categories.sum()
                df_new[col] = np.searchsorted(bins, u[:, j])
        return df_new

    # ------------------------------------------------------------------
    # CHOQUES DIRECIONADOS
    # ------------------------------------------------------------------
    def _apply_shocks(
        self,
        df_synth: pd.DataFrame,
        shocks: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        for col, cfg in shocks.items():
            if col not in df_synth.columns:
                warnings.warn(f"Choque ignorado: coluna {col} inexistente.")
                continue
            if "Δμ" in cfg:
                df_synth[col] += cfg["Δμ"]
            if "stretch" in cfg:
                mu = df_synth[col].mean()
                df_synth[col] = mu + cfg["stretch"] * (df_synth[col] - mu)
            if "quantile_fn" in cfg and callable(cfg["quantile_fn"]):
                q = df_synth[col].rank(pct=True).values
                df_synth[col] = cfg["quantile_fn"](q)
            if "transition_matrix" in cfg:
                tm = cfg["transition_matrix"]
                uniq = sorted(tm.keys())
                mapping = {}
                for k in uniq:
                    mapping[k] = np.random.choice(
                        uniq, p=[tm[k][u] for u in uniq], size=len(df_synth)
                    )
                df_synth[col] = mapping
        return df_synth

    # ------------------------------------------------------------------
    # VALIDAÇÃO BÁSICA
    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame, raise_err: bool = False) -> bool:
        """
        Checa unicidade de IDs e domínios válidos.
        """
        ok = True
        for col in self.id_cols:
            if df[col].duplicated().any():
                msg = f"IDs duplicados em {col}"
                if raise_err:
                    raise ValueError(msg)
                warnings.warn(msg)
                ok = False
        for col, meta in self._meta.items():
            if meta.dtype == "cat" and not df[col].isin(meta.categories).all():
                msg = f"Valor fora de domínio em {col}"
                if raise_err:
                    raise ValueError(msg)
                warnings.warn(msg)
                ok = False
        return ok
