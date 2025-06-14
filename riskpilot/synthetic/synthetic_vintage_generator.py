from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.tseries import frequencies, offsets
from scipy import stats


@dataclass
class _VarMeta:
    dtype: str  # 'cont', 'disc', 'cat', 'bool', 'date'
    categories: Optional[np.ndarray] = None
    quantiles: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None  # para PMF discreta


class LookAhead:
    """Gera safras sintéticas de contratos de empréstimo para testes de stress-PD.

    Parameters
    ----------
    id_cols : Sequence[str]
        Colunas que identificam cada linha.
    date_cols : Sequence[str]
        Colunas de data. Aceita ``datetime`` ou inteiros ``yyyymm``/``yyyymmdd``.
    """

    def __init__(
        self,
        id_cols: Sequence[str],
        date_cols: Sequence[str],
        keep_cols: Optional[Sequence[str]] = None,
        ignore_cols: Optional[Sequence[str]] = None,
        random_state: int | None = None,
        custom_noise: Optional[Dict[str, Dict[str, Any]]] = None,
        unclear_date_strategy: str = "start",
        int_date_format: Optional[Dict[str, str]] = None,
    ):
        self.id_cols = set(id_cols)
        self.date_cols = set(date_cols)
        self._date_cols = list(date_cols)
        self._date_has_time: Dict[str, bool] = {}
        self._max_dates: Dict[str, pd.Timestamp] = {}
        self._date_month_alignment: Dict[str, str] = {}
        self._has_time_component = False
        self._max_date = pd.Timestamp.min
        self.keep_cols = set(keep_cols or [])
        self.ignore_cols = set(ignore_cols or [])
        self.random_state = np.random.RandomState(random_state)
        self.custom_noise = custom_noise or {}
        if unclear_date_strategy not in {"start", "end"}:
            raise ValueError("unclear_date_strategy must be 'start' or 'end'")
        self.unclear_date_strategy = unclear_date_strategy
        self.int_date_format = int_date_format or {}
        self._date_int_format: Dict[str, str] = {}
        self._date_int_dtype: Dict[str, Any] = {}

        self._meta: Dict[str, _VarMeta] = {}
        self._corr: Optional[np.ndarray] = None
        self._order: List[str] = []
        self._fitted = False

    def _normalize_freq(self, freq: str | pd.DateOffset) -> pd.DateOffset:
        """Normaliza frequências legadas para evitar avisos futuros."""
        if isinstance(freq, str):
            f = freq.upper()
            if f == "M":
                return offsets.MonthEnd()
            if f == "BM":
                return offsets.BusinessMonthEnd()
        return frequencies.to_offset(freq)

    def _parse_intlike_dates(
        self, series: pd.Series
    ) -> Tuple[pd.DatetimeIndex, Optional[str]]:
        """Detect and parse integer-like date columns."""
        if series.empty:
            return pd.to_datetime(series), None
        if (
            pd.api.types.is_integer_dtype(series) or series.dtype == "object"
        ) and series.astype(str).str.isdigit().all():
            s = series.astype(str)
            parsed = pd.to_datetime(s, format="%Y%m", errors="coerce")
            if not parsed.isna().any():
                return parsed, "yyyymm"
            parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            if not parsed.isna().any():
                return parsed, "yyyymmdd"
        return pd.to_datetime(series), None

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "LookAhead":
        """Learns marginal distributions and rank correlations from historical data."""
        self._order = [
            c
            for c in df.columns
            if c not in self.ignore_cols and c not in self.date_cols
        ]

        for col in self._date_cols:
            raw = df[col].dropna()
            fmt_hint = self.int_date_format.get(col)
            if fmt_hint == "yyyymm":
                ser = pd.to_datetime(raw.astype(str), format="%Y%m")
                self._date_int_format[col] = "yyyymm"
                self._date_int_dtype[col] = df[col].dtype
            elif fmt_hint == "yyyymmdd":
                ser = pd.to_datetime(raw.astype(str), format="%Y%m%d")
                self._date_int_format[col] = "yyyymmdd"
                self._date_int_dtype[col] = df[col].dtype
            else:
                ser, detected = self._parse_intlike_dates(raw)
                if detected:
                    self._date_int_format[col] = detected
                    self._date_int_dtype[col] = df[col].dtype
            self._date_has_time[col] = (ser.dt.floor("s") != ser.dt.normalize()).any()
            if not ser.empty:
                self._max_dates[col] = ser.max()
                start_p = ser.dt.is_month_start.mean()
                end_p = ser.dt.is_month_end.mean()
                if max(start_p, end_p) < 0.6:
                    self._date_month_alignment[col] = self.unclear_date_strategy
                else:
                    self._date_month_alignment[col] = (
                        "start" if start_p >= end_p else "end"
                    )
        if self._date_cols:
            self._has_time_component = self._date_has_time[self._date_cols[0]]
            self._max_date = self._max_dates.get(self._date_cols[0], pd.Timestamp.min)

        for col in self._order:
            if col in self.id_cols:
                continue
            s = df[col].dropna()

            if col in self.date_cols:
                self._meta[col] = _VarMeta("date")
            elif pd.api.types.is_bool_dtype(s):
                self._meta[col] = _VarMeta("bool", values=s.values)
            elif pd.api.types.is_numeric_dtype(s):
                uniq, counts = np.unique(s, return_counts=True)
                if len(uniq) < 20:
                    # *disc*: store unique values and their empirical counts (pmf)
                    self._meta[col] = _VarMeta(
                        "disc",
                        values=uniq,
                        categories=counts.astype(float),
                    )
                else:
                    qs = np.quantile(s, np.linspace(0, 1, 1001))
                    self._meta[col] = _VarMeta("cont", quantiles=qs)
            else:
                cats, counts = np.unique(s.astype(str), return_counts=True)
                self._meta[col] = _VarMeta(
                    "cat",
                    categories=cats,
                    values=(counts / counts.sum()).astype(float),
                )

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
        n_periods: int | None = None,
        freq: str = "M",
        start_vintage: str | pd.Timestamp | None = None,
        end_vintage: str | pd.Timestamp | None = None,
        align_with_history: bool = True,
        skip_train_overlap: bool = True,
        scenario: str = "base",
        shocks: Optional[Dict[str, Dict[str, Any]]] = None,
        n_per_vintage: Optional[int] = None,
        preserve_corr: bool = True,
        date_offsets: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Gera múltiplas safras sintéticas.

        Parameters
        ----------
        n_periods : int, optional
            Número de períodos a gerar. Necessário quando ``end_vintage`` não é
            informado.
        freq : str
            Frequência de cada safra (ex.: "M", "Q", "A").
        start_vintage : str | pd.Timestamp | None
            Período inicial. Se ``align_with_history=True`` e ``start_vintage``
            for ``None`` utiliza ``max(date_cols) + freq``.
        end_vintage : str | pd.Timestamp | None
            Data final desejada; sobrescreve ``n_periods`` se fornecida.
        align_with_history : bool, default True
            Garante continuidade temporal usando a última data observada.
        skip_train_overlap : bool, default True
            Evita sobreposição com o histórico ajustando ``start_vintage``.
        scenario : {"base", "stress"}
            Controla intensidade do ruído.
        shocks : dict, opcional
            Choques direcionados por coluna.
        n_per_vintage : int, opcional
            Tamanho fixo; se ``None`` → bootstrap do histórico.
        preserve_corr : bool
            Usa cópula gaussiana para manter correlações numéricas.
        date_offsets : dict, opcional
            Deslocamentos específicos para cada coluna de data.
        """
        if not self._fitted:
            raise RuntimeError("Execute .fit() antes de .generate().")

        shocks = shocks or {}
        df_list: List[pd.DataFrame] = []

        offset = self._normalize_freq(freq)

        if start_vintage is not None:
            start_vintage = pd.to_datetime(start_vintage)
        elif align_with_history:
            start_vintage = pd.to_datetime(self._max_date) + offset
            if not self._has_time_component:
                start_vintage = start_vintage.normalize()
            if skip_train_overlap:
                candidate = start_vintage
                align = self._date_month_alignment.get(
                    self._date_cols[0], self.unclear_date_strategy
                )
                if isinstance(
                    offset,
                    (
                        offsets.MonthBegin,
                        offsets.MonthEnd,
                        offsets.BusinessMonthBegin,
                        offsets.BusinessMonthEnd,
                        offsets.QuarterBegin,
                        offsets.QuarterEnd,
                        offsets.BQuarterBegin,
                        offsets.BQuarterEnd,
                        offsets.YearBegin,
                        offsets.YearEnd,
                        offsets.BYearBegin,
                        offsets.BYearEnd,
                    ),
                ):
                    if align == "start":
                        candidate = pd.Timestamp(
                            candidate.year,
                            candidate.month,
                            1,
                            candidate.hour,
                            candidate.minute,
                            candidate.second,
                            candidate.microsecond,
                            tzinfo=candidate.tzinfo,
                        )
                    elif align == "end":
                        base = pd.Timestamp(candidate.year, candidate.month, 1)
                        base += pd.offsets.MonthEnd(1)
                        candidate = base.replace(
                            hour=candidate.hour,
                            minute=candidate.minute,
                            second=candidate.second,
                            microsecond=candidate.microsecond,
                            tzinfo=candidate.tzinfo,
                        )
                if not self._has_time_component:
                    candidate = pd.to_datetime(candidate).normalize()
                if candidate <= self._max_date:
                    start_vintage += offset
        else:
            start_vintage = pd.Timestamp.today().normalize() + offset

        if end_vintage is not None:
            end_vintage = pd.to_datetime(end_vintage)
            vint_dates = pd.date_range(
                start=start_vintage,
                end=end_vintage,
                freq=offset,
                normalize=not self._has_time_component,
            )
        else:
            if n_periods is None:
                raise ValueError("n_periods or end_vintage must be specified")
            vint_dates = pd.date_range(
                start=start_vintage,
                periods=n_periods,
                freq=offset,
                normalize=not self._has_time_component,
            )

        expected = pd.date_range(
            start=vint_dates[0],
            end=vint_dates[-1],
            freq=offset,
            normalize=not self._has_time_component,
        )
        if len(expected) != len(vint_dates):
            raise ValueError(
                "Generated dates are not continuous with the given frequency"
            )

        for vint in vint_dates:
            n_rows = n_per_vintage or self.random_state.choice([500, 1000, 2000])
            df_v = self._generate_one_vintage(
                n_rows,
                vint,
                scenario,
                shocks,
                preserve_corr,
                date_offsets,
                offset,
            )
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
        date_offsets: Optional[Dict[str, Any]],
        freq_offset: pd.DateOffset,
    ) -> pd.DataFrame:
        # 1. gera base vazia
        data = {}
        for col in self.id_cols:
            data[col] = [uuid.uuid4().hex for _ in range(n_rows)]
        for col in self._date_cols:
            col_date = vint_date
            if date_offsets and col in date_offsets:
                col_date = col_date + self._normalize_freq(date_offsets[col])
            align = self._date_month_alignment.get(col, self.unclear_date_strategy)
            if isinstance(
                freq_offset,
                (
                    offsets.MonthBegin,
                    offsets.MonthEnd,
                    offsets.BusinessMonthBegin,
                    offsets.BusinessMonthEnd,
                    offsets.QuarterBegin,
                    offsets.QuarterEnd,
                    offsets.BQuarterBegin,
                    offsets.BQuarterEnd,
                    offsets.YearBegin,
                    offsets.YearEnd,
                    offsets.BYearBegin,
                    offsets.BYearEnd,
                ),
            ):
                if align == "start":
                    col_date = pd.Timestamp(
                        col_date.year,
                        col_date.month,
                        1,
                        col_date.hour,
                        col_date.minute,
                        col_date.second,
                        col_date.microsecond,
                        tzinfo=col_date.tzinfo,
                    )
                elif align == "end":
                    base = pd.Timestamp(col_date.year, col_date.month, 1)
                    base += pd.offsets.MonthEnd(1)
                    col_date = base.replace(
                        hour=col_date.hour,
                        minute=col_date.minute,
                        second=col_date.second,
                        microsecond=col_date.microsecond,
                        tzinfo=col_date.tzinfo,
                    )
            if not self._date_has_time.get(col, True):
                col_date = pd.to_datetime(col_date).normalize()
            data[col] = np.full(n_rows, col_date)

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

        for col, fmt in self._date_int_format.items():
            if col in df_synth.columns:
                if fmt == "yyyymm":
                    df_synth[col] = (
                        df_synth[col].dt.year * 100 + df_synth[col].dt.month
                    ).astype(self._date_int_dtype[col])
                elif fmt == "yyyymmdd":
                    df_synth[col] = (
                        df_synth[col].dt.year * 10000
                        + df_synth[col].dt.month * 100
                        + df_synth[col].dt.day
                    ).astype(self._date_int_dtype[col])

        return df_synth

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample_marginal(
        self, col: str, meta: _VarMeta, n: int, scenario: str
    ) -> np.ndarray:
        cust = self.custom_noise.get(col)
        if cust is not None:
            func: Callable = cust["func"]
            kwargs = dict(cust.get("kwargs", {}))
            kwargs.setdefault("size", n)
            return func(**kwargs)

        if meta.dtype == "cont":
            u = self.random_state.uniform(0, 1, size=n)
            synt = np.interp(u, np.linspace(0, 1, len(meta.quantiles)), meta.quantiles)
            sigma = synt.std(ddof=0)
            mu_shift = 0.2 * sigma if scenario == "stress" else 0.0
            noise = self.random_state.normal(mu_shift, 0.2 * sigma, size=n)
            return synt + noise

        if meta.dtype == "disc":
            probs = meta.categories / meta.categories.sum()
            if len(probs) != len(meta.values):
                raise ValueError(
                    f"Length mismatch for disc variable '{col}':"
                    f" len(values)={len(meta.values)} len(categories)={len(probs)}"
                )
            synt = self.random_state.choice(meta.values, size=n, p=probs)
            jitter = self.random_state.poisson(1, size=n) - 1
            if scenario == "stress":
                jitter *= 2
            return np.maximum(0, synt + jitter)

        if meta.dtype == "cat":
            synt = self.random_state.choice(meta.categories, size=n, p=meta.values)
            if scenario == "stress":
                min_idx = np.argmin(meta.values)
                flip_mask = self.random_state.rand(n) < 0.05
                synt[flip_mask] = meta.categories[min_idx]
            return synt

        if meta.dtype == "bool":
            flip_p = 0.10 if scenario == "stress" else 0.05
            base = self.random_state.choice(meta.values, size=n)
            flips = self.random_state.rand(n) < flip_p
            return np.where(flips, ~base, base)

        raise ValueError(f"Unknown dtype '{meta.dtype}' for column '{col}'.")

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
                df_new[col] = np.interp(
                    u[:, j], np.linspace(0, 1, len(meta.quantiles)), meta.quantiles
                )
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
