# Comparative Time Series Analysis and Forecasting of Mobile Network Traffic

Formative assignment for **Machine Learning Techniques I** — analysis and one-step-ahead forecasting of mobile internet traffic across 10,000 geographic areas in Milan, using the Telecom Italia Big Data Challenge dataset.

---

## Overview

The dataset records internet activity (CDR counts) for a 100×100 grid of city areas at 10-minute intervals over two months (November–December 2013). The notebook covers three tasks:

- **Task 1** — Efficient loading and memory optimisation of ~5 GB of raw data
- **Task 2** — Exploratory data analysis: distribution, stationarity, decomposition, autocorrelation, spatial patterns, anomalies
- **Task 3** — Design, training, and comparison of three forecasting models (SARIMA, LSTM, Vanilla RNN) on three target areas, with predictions for December 16–22

---

## Repository Structure

```
├── Milan_Traffic_Forecasting.ipynb   # Main notebook — all tasks
├── README.md
```

---

## Requirements

### Hardware
- Google Colab (recommended) — the notebook is written for Colab with Google Drive
- GPU runtime recommended for LSTM and RNN training (Runtime → Change runtime type → T4 GPU)
- ~12 GB RAM (standard Colab session is sufficient after memory optimisation)

### Python dependencies

All installed automatically in the first notebook cell:

```bash
pip install pyarrow statsmodels scikit-learn tensorflow matplotlib seaborn scipy psutil
```

Key libraries:

| Library | Purpose |
|---|---|
| `pyarrow` | Parquet caching and filter pushdown |
| `statsmodels` | SARIMA, ADF test, ACF/PACF, decomposition |
| `tensorflow` / `keras` | LSTM and Vanilla RNN models |
| `scikit-learn` | MinMaxScaler, MAE, RMSE |
| `psutil` | Memory usage tracking |
| `scipy` | KDE for PDF plot |

---

## Dataset

The dataset is the **Telecom Italia Big Data Challenge** — Milan telecommunications activity.

- **Download:** [Harvard Dataverse — Telecommunications activity](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
- **Grid reference:** [Harvard Dataverse — Grid dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU)
- **Size:** ~5 GB across 62 daily `.txt` files
- **Format:** Tab-separated, 8 columns per row

> **Field-order note:** The original Barlacchi et al. paper has a column ordering error. The correct order is:
> `square_id | time_interval | country_code | sms_in | sms_out | call_in | call_out | internet_activity`
> Only `square_id`, `time_interval`, and `internet_activity` are used.

---

## How to Run

### 1. Set up Google Drive

Upload all daily `.txt` files to a folder in your Google Drive, e.g.:

```
MyDrive/milan_traffic/raw/
```

### 2. Update the data path

In the **Setup** cell, set `DATA_DIR` to your folder:

```python
DATA_DIR     = Path('/content/drive/MyDrive/milan_traffic/raw')
PARQUET_PATH = Path('/content/drive/MyDrive/milan_traffic/milan_traffic.parquet')
```

### 3. Run cells top to bottom

- **First run:** Builds a snappy-compressed Parquet cache (~200 MB) from all raw `.txt` files. This takes 10–20 minutes depending on connection speed and only happens once.
- **Subsequent runs:** Load directly from Parquet in seconds — raw CSV parsing is skipped entirely.

> The notebook also includes an automated download cell that fetches files from Harvard Dataverse via the API, if you prefer not to download manually.

---

## Memory Optimisation (Task 1)

Three strategies reduce the 5 GB raw dataset to a manageable in-memory footprint:

| Strategy | Detail |
|---|---|
| Column selection | `usecols` keeps only 3 of 8 columns — SMS and call fields never enter memory |
| Chunked reading | `chunksize=200_000` — no single read call holds an entire daily file |
| Dtype downcasting | `square_id`: `int64` → `int16`; `internet_activity`: `float64` → `float32` |

**Result:** ~41.7% memory reduction (7,209 MB → 4,205 MB theoretical baseline).

---

## Models (Task 3)

All three models are trained and evaluated independently on each of three target areas:

| Area | Description |
|---|---|
| Area 5161 | Highest total traffic — city centre commercial hotspot |
| Area 4159 | Mid-traffic area, downward trend through late November |
| Area 4556 | Stable, moderate traffic, higher noise |

**Test period:** December 16–22, 2013  
**Input window:** 1,008 steps (7 days × 144 steps/day at 10-minute resolution)

### SARIMA
- Order `(2, 0, 2)(1, 0, 1, 24)` derived from ADF test (d=0) and ACF/PACF analysis (p=2, q=2)
- Resampled to hourly resolution before fitting (s=24 instead of s=144) — roughly 36× cheaper
- Each hourly forecast repeated 6 times to restore 10-minute resolution

### LSTM
- Two stacked LSTM layers, `hidden_size=64`, `dropout=0.2`, `lr=5e-4`
- `return_sequences=True` on first layer passes full hidden state to second layer
- Trained with early stopping (patience=5) and ReduceLROnPlateau

### Vanilla RNN
- Identical architecture to LSTM — only `SimpleRNN` replaces `LSTM` layers
- Deliberate controlled experiment: one variable changed (gating), everything else held constant
- Best config: `hidden_size=64`, `dropout=0.2`, `lr=1e-3` (selected by validation RMSE)

---

## Key Results

SARIMA outperforms both neural models on MAE, MAPE, and RMSE across all three areas. The dataset's strong, stable daily seasonality and confirmed stationarity (ADF) favour a well-specified statistical model over deep learning.

The RNN's errors are roughly twice the size of LSTM's on the high-traffic area (Area 5161 MAE: RNN 1,288 vs LSTM 629) — consistent with vanishing gradient degradation over the 1,008-step window.

**One exception:** Area 4159 (lowest traffic, smooth signal) — RNN beats LSTM on MAE (119 vs 173). Short-range memory suffices for a low-amplitude series.

---

## Failure Case

Neural models produce near-flat predictions for Area 4556 despite actual traffic swinging 200–900 CDR daily. The cause is autoregressive inference: each prediction is fed back as input. Once predictions drift flat, the context window fills with the model's own history — the overnight trough signal disappears and predictions never recover. December 18–19 is the worst period across all models (traffic peaks above the training distribution).

---

## AI Tool Usage

AI assistance (Claude, Anthropic) was used as a guide in Task 1 (memory optimisation strategy), Task 2 (interpreting ADF and ACF/PACF results), and Task 3 (articulating SARIMA parameter justification).  Full details are documented in the report. 

---

## References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," *Sci Data* 2, 150055 (2015). https://doi.org/10.1038/sdata.2015.55

[2] Telecommunications activity dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

[3] Grid dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

[11] GitHub repository: https://github.com/PapiWinnie/Time_Series_Forecasting_Formative_1

[12] Demo video: https://drive.google.com/file/d/1mB7luuVoLzzC6o5Lo6tDix35pQ1FGlHJ/view?usp=sharing
