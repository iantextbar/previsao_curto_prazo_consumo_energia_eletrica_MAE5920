# Project context: short-term household energy consumption forecasting

_Last updated: 2026-11-03_

## Overview

Course project for **MAE5920 – Deep Learning** (USP, Prof. Dr. Hamed Yazdanpanah), by Ian Teixeira Barreiro and Ornella Scardua Ferreira.

- **Task:** forecast appliance energy consumption (`Appliances`, in Wh) at **10-minute and 60-minute horizons**.
- **Dataset:** UCI *Appliances Energy Prediction* (https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), file `src/data/energydata_complete.csv` (committed). It has 19,735 rows at 10-minute resolution from 2016-01-11 17:00 to 2016-05-27 18:00, with no gaps and no missing values.
- **Who does which model:**
  - Ian: autoregressive baseline (AR with exogenous variables) and a recurrent network (GRU).
  - Ornella: MLP and 1D CNN.
  - Everything is trained in **PyTorch**, and both people use the same train/val/test sets.
- **Evaluation:** MAE, RMSE, MASE (persistence forecast as the reference) and performance by hour of day.
- **Deliverables:** a reproducible notebook, a data dictionary plus a data-leakage analysis, baseline and deep models, a results table for train/val/test, residual diagnostics, a computational cost summary, a 6–8 page report, a 10–12 minute presentation, and a final section titled "Quando o modelo de aprendizagem profunda falha?".

## Repository layout

- `src/notebooks/previsao_curto_prazo_consumo_energia_eletrica.ipynb` is the main notebook. Its markdown is in **Portuguese**.
- `src/data/energydata_complete.csv` is the raw data.
- `pyproject.toml` / `uv.lock` are managed with **uv**. Dependencies: numpy, pandas, plotly, scikit-learn, nbformat. PyTorch is not added yet.
- The virtual environment is `.venv/`, with Python 3.12.

## Notebook structure (current state)

1. **Imports and loading.** `DATAPATH` resolves to `src/data` when the notebook runs from `src/notebooks`.
2. **EDA** (plotly). This section still uses the **original column names** (`T1`, `RH_6`, `T_out`, …), because Ian's written observations refer to them.
   - `date` is converted to datetime and used as the index (the column is kept too, with `drop=False`).
   - Plots: appliances and lights over time (raw and daily mean), hour-of-day profile and day × hour heatmap, a histogram, temperature and humidity per room (hourly lines plus box plots via `plot_rooms`), outside conditions (`T_out`/`RH_out` vs `T6`/`RH_6`), the house mean temperature minus `T_out`, and a correlation matrix.
   - Ian's findings:
     - Consumption is strongly right-skewed with outliers.
     - Consumption peaks around 18:00 every day.
     - Mondays and Fridays have higher afternoon consumption (possibly home-office days).
     - The bathroom is the most humid room and the ironing room the driest.
     - The house is on average about 13 °C warmer than outside.
     - `RH_6` correlates negatively with indoor temperatures.
3. **Preprocessing.**
   - **Column renaming** (`COLUMN_NAMES`). The pattern is `<room>_temp` / `<room>_hum` (kitchen, living_room, laundry, office, bathroom, ironing_room, teen_room, parents_room), `outside_north_*` (T6/RH_6) and `weather_*` (Chièvres station: temp, hum, pressure, windspeed, visibility, dewpoint). `Appliances` becomes `appliances`.
   - **Dropped columns:** `rv1` and `rv2`. These are random noise added on purpose, and they are identical to each other.
   - **Added time features:** `hour_sin` and `hour_cos` (cyclic minute-of-day), and `dow_0` to `dow_6` (one-hot day of week, 0 = Monday).
   - **Targets:** `y_10` is `appliances` shifted −1 step and `y_60` is `appliances` shifted −6 steps (`HORIZONS`). Targets are built **inside each split**, so the last 6 rows of each split are dropped and no target crosses into the next split.
   - **Split:** chronological, **70/15/15**, with cuts rounded to midnight (`TRAIN_FRAC`, `VAL_FRAC`).

     | Split | Dates | Rows |
     |---|---|---|
     | train | 2016-01-11 → 2016-04-15 (96 days) | 13,716 |
     | val | 2016-04-16 → 2016-05-06 (21 days) | 3,018 |
     | test | 2016-05-07 → 2016-05-27 (21 days) | 2,983 |

     70/15/15 was chosen over 70/10/20 because a 2-week validation set is noisy (weekly mean consumption ranges from about 75 to about 120 Wh).
   - **Data structures:** `splits[name]` holds the unscaled DataFrames, indexed by date. They are kept for the dates, for the persistence metrics and for evaluation by hour of day.
   - **Standardization (sklearn):**
     - `x_scaler` is a `ColumnTransformer(StandardScaler on SCALE_COLS, remainder='passthrough')`, fit on train only.
     - `y_scaler` is a `StandardScaler` fit on train `appliances`. The helpers `scale_y` / `unscale_y` work on targets of shape `(n, k)`.
   - **Model inputs:** `X[name]` (n × 35) and `y[name]` (n × 2, columns `TARGETS = ['y_10', 'y_60']`) are **numpy float32** arrays, ready for `torch.from_numpy`. The column order of `X` is `FEATURE_NAMES`.
   - **No files are saved.** The splits are deterministic and cheap, so they are rebuilt by running the notebook.
   - **Persistence baseline** (`persistence_metrics`, `MASE_SCALE` = train persistence MAE per horizon), in Wh:

     | Split | MAE y_10 | RMSE y_10 | MAE y_60 | RMSE y_60 |
     |---|---|---|---|---|
     | train | 30.83 | 74.12 | 60.29 | 123.95 |
     | val | 26.85 | 67.60 | 53.29 | 113.80 |
     | test | 26.51 | 66.46 | 46.02 | 101.29 |

## Key observations for the report

- **Distribution shift:** the test period (May) is much warmer than train (January–April). Outside temperature averages 14.5 °C in test vs 5.7 °C in train, and the standardized `kitchen_temp` averages about +2.5 in test. The models must extrapolate, which is material for the "Quando o modelo … falha?" section.
- **Possible leakage, already handled:**
  - The split is chronological with no shuffling.
  - Scalers are fit on train only.
  - Targets are built within each split.
  - Windows and lags must also be built within each split.

## Open decisions and next steps

- **Window length L and lags:** the RNN and CNN need windows of length L, and the AR model and MLP need lagged `appliances`. Build them per split; the first L rows of each split lack full history.
- **Target transform and loss:** `appliances` is heavily skewed. The options discussed were (1) `log1p` before scaling, or (2) keeping the scale and training with MAE/Huber loss instead of MSE. The suggestion was to start with (2) and try (1) as a variant. Ian and Ornella should use the same choice.
- **Models:** add PyTorch to the dependencies (`uv add torch`), then implement the AR baseline and the GRU.
- **Evaluation table:** MAE, RMSE and MASE per split and horizon, plus metrics by hour of day (use `splits[name].index.hour`).

## Working notes

- The repo lives on the Windows drive (`/mnt/c/...`) and is opened in VS Code through WSL. When the notebook is edited from outside VS Code, the open tab does **not** refresh, and even "Revert File" doesn't help. The fix is to close the tab **without saving** and reopen it, or run "Developer: Reload Window". Saving the stale tab would overwrite the external edits.
- Before editing the notebook, check its modification time and current cell contents, because Ian may have run or edited it in the meantime.
- The notebook outputs contain plotly figures, so the file is about 17 MB.
