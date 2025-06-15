<p align="center">
  <h1 align="center">RiskPilot</h1>
</p>

[![CI](https://github.com/joaaomaia/riskpilot/actions/workflows/python-package.yml/badge.svg)](https://github.com/joaaomaia/riskpilot/actions/workflows/python-package.yml)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](#)

Ferramentas para avaliação de modelos, geração de carteiras sintéticas e monitoramento de drift.

---

## 📌 Objetivo

O **Binary Performance Evaluator** é uma classe Python que permite avaliar facilmente a performance de classificadores binários já treinados, carregando-os de arquivos `.joblib` / `.pkl` ou passando diretamente o objeto em memória.

Ele compara desempenho em **treino**, **teste** e, opcionalmente, **validação**, com geração de métricas, gráficos de calibração, matrizes de confusão e muito mais.

---

## ⚙️ Instalação

```bash
pip install -e .[dev]
# ou
pip install riskpilot
```

```bash
# Optional visualization
pip install riskpilot[viz]
pip install kaleido  # required for saving Plotly figures as images
```

---

## 🚀 Exemplo Rápido

```python
from riskpilot.evaluation import BinaryPerformanceEvaluator

evaluator = BinaryPerformanceEvaluator(
    model="modelo_treinado.pkl",   # caminho .pkl/.joblib ou objeto já carregado
    df_train=df_train,
    df_test=df_test,
    df_val=df_val,                 # opcional
    target_col="default_90d",
    id_cols=["contract_id"],
    date_col="snapshot_date",     # opcional
    group_col="product_type",     # opcional
    save_dir="resultados",        # opcional, salva gráficos como PNG
    threshold=0.5                 # opcional
)

# Calcula métricas
metrics = evaluator.compute_metrics()

# Gráficos
evaluator.plot_confusion(save=True)
evaluator.plot_calibration()
evaluator.plot_event_rate()
evaluator.plot_psi(reference_last_period=True)
evaluator.plot_ks()

# Visualizar resultados numéricos
print(metrics)

# Gerar cenário de stress
stress = evaluator.run_stress_test(
    n_periods=36,
    freq="ME",
    scenario="stress",
)
print(stress["metrics"])
```

---

## 🧠 Entradas Esperadas

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `model` | `str` / `Path` / objeto | Caminho para `.joblib` ou `.pkl`, ou modelo em memória |
| `df_train` | `pd.DataFrame` | Base de treino |
| `df_test` | `pd.DataFrame` | Base de teste |
| `df_val` | `pd.DataFrame` | (Opcional) base de validação |
| `target_col` | `str` | Nome da coluna alvo (0 = negativo, 1 = positivo) |
| `id_cols` | `list[str]` | Colunas que identificam cada linha |
| `date_col` | `str` | (Opcional) Coluna de data para análises temporais |
| `group_col` | `str` | (Opcional) Coluna categórica para agrupamentos |
| `save_dir` | `str` / `Path` | (Opcional) Diretório para salvar gráficos PNG |
| `threshold` | `float` | (Opcional) Probabilidade limite para classificação |

---

## 📊 Funcionalidades
- Todos os gráficos aceitam parâmetro ``title`` para personalização

### ✅ Métricas Automáticas
- MCC (Matthews Correlation Coefficient)
- AUC ROC e AUC PR
- Precision, Recall
- Brier Score

### 🧱 Matriz de Confusão
- Gráfico Plotly com contraste automático de texto
- Suporta threshold otimizado (``"ks"``/``"youden"``)
- Pode gerar matrizes por grupo (``group_col``)
- Mostra valores absolutos e percentuais

### 🎯 Curva de Calibração
- Com Brier Score no título
- Compara previsão × observação

### 📈 Evolução de Eventos
- Mostra taxa de evento (target=1) por grupo ao longo do tempo
- Inclui barra empilhada com % de IDs por grupo

### 🧪 PSI por Variável
- PSI por variável ao longo do tempo (usando `date_col`)
- Bins por quantis com base em dataset de referência
- Possibilidade de usar o período imediatamente anterior como referência
- Tolerante a valores fora do intervalo e períodos com poucos dados
- É possível ajustar `min_obs` quando um período tem poucos registros
  (ex.: `evaluator.plot_psi(reference_last_period=True, min_obs=1)`)
- Indicação visual de faixas:
  - PSI ≤ 0.10 (aceitável)
  - PSI 0.10–0.25 (monitorar)
  - PSI > 0.25 (alerta)

### 🧭 KS Temporal
- Mostra evolução do KS (Kolmogorov–Smirnov) no tempo para treino/teste/validação

---

## 📤 Saídas

- `.report` — dicionário Python contendo todas as métricas numéricas organizadas por split.
- `compute_metrics()` retorna um `DataFrame` com essas métricas.
- Gráficos: podem ser exibidos na tela ou salvos em `save_dir`.
- `evaluator.binning_table()` retorna a tabela de binning (se houver).

---

## 📁 Organização dos Gráficos

Se `save_dir="resultados"`:

```
resultados/
├── confusion_matrices.png
├── calibration_curve.png
├── event_rate.png
├── psi_over_time.png
└── ks_evolution.png
```

---

## LookAhead Generator

```python
from riskpilot.synthetic import LookAhead

gen = LookAhead(
    id_cols=["contract_id"],
    date_cols=["snapshot_date"],
    random_state=42,
).fit(df_hist)

df_future = gen.generate(n_periods=36, freq="M")
```

Date columns automatically keep their month-start or month-end alignment based on
the historical data. Mixed patterns default to start-of-month alignment.

### Date granularity & plotting

Datas sem componente de hora permanecem como datas puras, evitando rótulos
``00:00:00`` em gráficos. Se as colunas de data incluírem horas, elas são
mantidas na geração e nas visualizações.

---

## 🧪 Testado com

- scikit-learn 1.4+
- pandas 2.2+
- numpy 1.26+
- matplotlib 3.8+
- seaborn 0.13+
- plotly 5+

---

## ✍️ Licença

MIT License.
