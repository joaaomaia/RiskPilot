import logging
from typing import Literal, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # ajuste o nível conforme necessidade


class TemporalIDSampler:
    """
    Amostrador de contratos/IDs para criar splits de treino e teste
    com suportes a *snapshot* e *full history*.

    Parameters
    ----------
    df : pd.DataFrame
        Base completa contendo as colunas de data e id.
    date_col : str
        Nome da coluna de data (tipo datetime64[ns] ou yyyymm_int etc.).
    id_col : str
        Nome da coluna de identificação do contrato.
    start_train, end_train, start_test, end_test : hashable
        Delimitadores temporais para cada split (mesma granularidade do
        `date_col`). Se `mode="snapshot"`, eles definem *janelas* de onde
        as amostras serão tiradas.
    mode : {"snapshot", "full"}, default "snapshot"
        • "snapshot": amostra até `snapshot_counts` registros aleatórios
          **por id** dentro da janela.  
        • "full": devolve todo o histórico (todas as datas) do id
          pertencente ao período de cada split.
    snapshot_counts : int, default 1
        Quantidade máxima de linhas aleatórias por contrato em "snapshot".
    force_gap : bool, default False
        Se True, garante que **nenhum** id presente no train apareça no test.
    force_continuous : bool, default False
        Se True, garante que **todo** id presente no train tenha
        **pelo menos um** registro no test e vice-versa.

    Notes
    -----
    • `force_gap` e `force_continuous` são mutuamente excludentes.  
    • Todas as verificações de consistência disparam AssertionError.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str,
        id_col: str,
        *,
        start_train,
        end_train,
        start_test,
        end_test,
        mode: Literal["snapshot", "full"] = "snapshot",
        snapshot_counts: int = 1,
        force_gap: bool = False,
        force_continuous: bool = False,
    ):
        # ---------------- assertions ---------------- #
        assert mode in {"snapshot", "full"}, "`mode` deve ser 'snapshot' ou 'full'."
        assert not (force_gap and force_continuous), (
            "Não é permitido definir force_gap=True e force_continuous=True simultaneamente."
        )
        assert date_col in df.columns, f"{date_col!r} não encontrado no DataFrame."
        assert id_col in df.columns, f"{id_col!r} não encontrado no DataFrame."
        assert snapshot_counts >= 1, "snapshot_counts precisa ser >= 1."

        self.df = df.copy()
        self.date_col = date_col
        self.id_col = id_col
        self.mode = mode
        self.snapshot_counts = snapshot_counts
        self.force_gap = force_gap
        self.force_continuous = force_continuous

        # converte coluna de data se for int-like yyyymm
        if pd.api.types.is_integer_dtype(self.df[date_col]):
            self.df[date_col] = pd.to_datetime(
                self.df[date_col].astype(str), format="%Y%m"
            )

        self._window_train = (start_train, end_train)
        self._window_test = (start_test, end_test)

        logger.info(
            "TemporalIDSampler inicializado: mode=%s, snapshot_counts=%s, force_gap=%s, "
            "force_continuous=%s",
            mode,
            snapshot_counts,
            force_gap,
            force_continuous,
        )

    # ----------------------------------------------------------------- #
    def _to_dt(self, val):
        """Converte limites em datetime compatível com date_col."""
        if isinstance(val, (int, np.integer, str)) and len(str(val)) in (6, 8):
            # 6 dígitos = yyyymm   | 8 dígitos = yyyymmdd
            fmt = "%Y%m" if len(str(val)) == 6 else "%Y%m%d"
            return pd.to_datetime(str(val), format=fmt)
        return pd.to_datetime(val)

    def _subset_window(self, start, end) -> pd.DataFrame:
        start_dt, end_dt = self._to_dt(start), self._to_dt(end)
        mask = (self.df[self.date_col] >= start_dt) & (self.df[self.date_col] <= end_dt)
        subset = self.df.loc[mask].copy()
        logger.info(
            "Seleção janela %s → %s: %d registros", start_dt.date(), end_dt.date(), len(subset)
        )
        return subset


    # --------------------------------------------------------------------- #
    def _sample_snapshot(self, df_win: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df_win)
        sampled = (
            df_win.groupby(self.id_col, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), self.snapshot_counts), random_state=42))
        )
        logger.info(
            "Snapshot de %d → %d linhas (snapshot_counts=%d)",
            n_before,
            len(sampled),
            self.snapshot_counts,
        )
        return sampled

    # --------------------------------------------------------------------- #
    def sample(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retorna (df_train, df_test) de acordo com as regras."""
        # ---------- gerar splits brutos pela janela -----------
        df_train = self._subset_window(*self._window_train)
        df_test = self._subset_window(*self._window_test)

        # ---------- aplicar modo snapshot se necessário -------
        if self.mode == "snapshot":
            df_train = self._sample_snapshot(df_train)
            df_test = self._sample_snapshot(df_test)

        # ---------- checagens force_gap / force_continuous ----
        ids_train: set = set(df_train[self.id_col])
        ids_test: set = set(df_test[self.id_col])

        if self.force_gap:
            overlap = ids_train & ids_test
            assert (
                len(overlap) == 0
            ), f"force_gap=True, mas {len(overlap)} ids aparecem em train e test."

        if self.force_continuous:
            missing_train = ids_train - ids_test
            missing_test = ids_test - ids_train
            assert (
                len(missing_train) == 0 and len(missing_test) == 0
            ), (
                f"force_continuous=True, mas {len(missing_train)} ids do train "
                f"não aparecem no test e {len(missing_test)} ids do test "
                f"não aparecem no train."
            )

        logger.info(
            "Amostragem finalizada: train=%d linhas (%d ids) | test=%d linhas (%d ids)",
            len(df_train),
            len(ids_train),
            len(df_test),
            len(ids_test),
        )
        return df_train.reset_index(drop=True), df_test.reset_index(drop=True)
