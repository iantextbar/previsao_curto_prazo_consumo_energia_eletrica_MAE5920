# Project context: short-term household energy consumption forecasting

_Last updated: 2026-11-10 (branch `refactor/validacao-temporal`)_

## Overview

Course project for **MAE5920 – Deep Learning** (USP, Prof. Dr. Hamed Yazdanpanah), by Ian Teixeira Barreiro and Ornella Scardua Ferreira.

- **Task:** forecast appliance energy consumption (`Appliances`, in Wh) at **10-minute and 60-minute horizons**.
- **Dataset:** UCI *Appliances Energy Prediction* (https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), file `src/data/energydata_complete.csv` (committed). It has 19,735 rows at 10-minute resolution from 2016-01-11 17:00 to 2016-05-27 18:00, with no gaps and no missing values.
- **Who does which model:**
  - Ian: the ARX baseline and the GRU (main model).
  - Ornella: the MLP and the 1D CNN.
  - Everything is trained in **PyTorch** on the same splits.
- **Evaluation:** MAE, RMSE, MASE (persistence as the reference) and performance by hour of day.
- **Deliverables:** a reproducible notebook, a data dictionary plus a data-leakage analysis, baseline and deep models, a results table for train/val/test, residual diagnostics, a computational cost summary, a 6–8 page report, a 10–12 minute presentation, and a final section titled "Quando o modelo de aprendizagem profunda falha?".

## Repository layout

- `src/notebooks/previsao_curto_prazo_consumo_energia_eletrica.ipynb` is the main notebook. Its markdown is in **Portuguese**.
- `src/data/energydata_complete.csv` is the raw data.
- `outputs/` (prediction CSVs) and `artifacts/` (scaler `.joblib` files) are written by the notebook and git-ignored.
- `pyproject.toml` / `uv.lock` are managed with **uv**. Dependencies: numpy, pandas, plotly, scikit-learn, statsmodels, torch, ipykernel, nbformat.
- The virtual environment is `.venv/`, with Python 3.12.

## Notebook structure

0. **Optional Colab setup cell.** On Colab only, it clones the repo (branch in `REPO_BRANCH`), changes into `src/notebooks` and installs missing dependencies. Locally it does nothing.
1. **Imports and loading.** All imports are in the first cell. `DATAPATH` resolves to `src/data` when the notebook runs from `src/notebooks`.
2. **EDA** (plotly). This section uses the **original column names** (`T1`, `RH_6`, `T_out`, …), because Ian's written observations refer to them.
   - Ian's findings:
     - Consumption is strongly right-skewed and peaks around 18:00.
     - Mondays and Fridays have higher afternoon use.
     - The bathroom is the most humid room.
     - The house is about 13 °C warmer than outside.
     - `RH_6` correlates negatively with indoor temperatures.
3. **Preprocessing (shared by all models).**
   - **Renaming:** `COLUMN_NAMES`, with `<room>_temp` / `<room>_hum`, `outside_north_*` and `weather_*`.
   - **Dropped columns:** `rv1` / `rv2`.
   - **Time features:** `hour_sin` / `hour_cos` and `dow_0..6`.
   - **Weather leak fix:** the dataset linearly interpolates the hourly `weather_*` readings to 10 minutes, so the value at 17:10 already contains part of the 18:00 reading. All `weather_*` columns use the **last full-hour value** (forward fill). Ornella spotted this.
   - **Targets and split:** `y_10` / `y_60` are built inside each split (`make_targets`), and the chronological 70/15/15 split is cut at midnight.

     | Split | Dates | Rows |
     |---|---|---|
     | train | 2016-01-11 → 2016-04-15 | 13,716 |
     | val | 2016-04-16 → 2016-05-06 | 3,018 |
     | test | 2016-05-07 → 2016-05-27 | 2,983 |
   - **Standardization:** `standardize(frame, train_index)` uses a StandardScaler on the continuous columns, fit on the given training rows. Each network fits its own scaler on training rows.
   - **Representations of the 21 temperature/humidity columns:** `apply_representation(frame, 'niveis' | 'variacoes')`. `variacoes` replaces each column with its 10-minute change, 60-minute change and deviation from the past-24 h mean.
   - **Shared evaluation:**
     - `regression_metrics` computes MAE, RMSE and MASE, where MASE uses `MASE_SCALE` = training-set persistence MAE (30.83 / 60.29 Wh).
     - `long_predictions` builds a long table, and the dict `PREDICTIONS[(model, round)]` holds one table per model and round.
     - `predictions_table`, `metrics_by_seed` and `summarize_seeds` filter and summarize it.
     - `export_predictions` and `export_scaler` write the files.
     - Plot and diagnostic helpers: `plot_metric_bars`, `plot_by_hour`, `mean_bias`, `residual_diagnostics` (residual ACF + Ljung–Box on spaced origins), `plot_residuals_vs_fitted`, `cost_summary`.
     - Model keys and labels are in `MODEL_LABELS` / `MODEL_COLORS`, including variants such as `mlp_res` and `cnn_1d_res_lag24`.
   - **Convention for lags and windows (all models):** build them on the full contiguous series, then select each split's rows. Val/test rows may use history from the previous split, so every model is scored on exactly the same val/test rows.
4. **ARX.**
   - ACF/PACF on train.
   - **Helpers:** `arx_inputs(representation, train_index)`, `build_arx(inputs, parts, horizon, p, q, lag24)`, `arx_grid`, `best_arx`, `fit_arx`, `plot_arx_search`.
   - Ridge is fit directly in Wh. Its predictions don't depend on the target's scale, so this is equivalent to fitting on a standardized target.
   - **Best configuration (unchanged):** p = 12, q = 0, with the 24 h lag; alpha 100 for 10 min and 1000 for 60 min.
5. **MLP and CNN 1D (Ornella's models, refactored).**
   - **Inputs:**
     - `build_features(frame, L, representation, lag24)`;
     - `prepare_mlp(L, representation, parts, lag24, residual)`;
     - `prepare_cnn(...)`, which uses the same origins as the MLP.
     - With the default options, the inputs are **bit-identical** to Ornella's original code.
   - **Generic network helpers:**
     - `train_network`: tuple inputs, per-epoch history in `model.history_`, `loss='huber'|'l1'`, `grad_clip`, `device`, `model_factory(inputs, n_out, **kwargs)`;
     - `ResidualOutput`, `predict_raw` / `predict_wh`;
     - `nn_predictions(..., output_steps)`, `network_grid`, `best_config`, `train_seeds`, `seed_predictions`, `plot_learning_curves`.
   - **Round-1 configurations:** MLP L = 12, hidden 128, dropout 0.2, lr 1e-3; CNN L = 6, lr 3e-4. Both use Huber loss on `log1p`, with 6 outputs (t+1 … t+6).
6. **Round-1 comparison.** Metrics, hourly MAE/bias, residual diagnostics and cost.
7. **"Mudança de distribuição e representação das variáveis" (forward-chaining study).**
   - **Evidence on train only:** KPSS rejects stationarity for all 21 level series and none of their changes.
   - **Folds:** 3 expanding-window folds with 21-day blocks (Mar 5–25, Mar 26–Apr 15, Apr 16–May 6).
     - The scaler is refit per fold.
     - Networks early-stop on the last 7 days of fold training (`early_stopping_split`).
     - Fold blocks never reach the test period.
   - **Pre-registered rule:** lowest mean MASE (folds, horizons, ARX + MLP). It chose **`niveis`** (0.906 vs 1.174), so the change-based representation **did not validate**.
   - Inside train + val the shift is mild (5–10% of level values out of the fold's training range), so the folds give little reward for robustness to shift.
8. **Round 2 (audit fixes).**
   - **Variants:** residual head (`residual`) × 24 h lag (`lag24`) for the MLP and CNN, chosen by the same folds and rule.
     - Chosen: **`mlp_res`** (fold MASE 0.770 against 0.971 for the base) and **`cnn_1d_res`** (0.792 against 1.454).
     - The 24 h lag doesn't help.
     - All variants are trained with 5 seeds on the official split and kept in `PREDICTIONS` with round = 2.
   - **ARX + MLP on out-of-sample ARX residuals:**
     - Training residuals come from monthly expanding-window ARX refits (`arx_oos_predictions`, `prepare_arx_residuals`, `arx_mlp_predictions`).
     - It uses L1 loss with residuals scaled by their training std.
     - Fold MASE 0.831 vs 0.841 for the ARX alone, so it counts as useful under the rule.
9. **GRU (main model).** See the open items.
   - **Inputs:** `prepare_gru(L, representation, parts, output_steps)` builds windows `(n, L, 30)`, a vector with the target-time calendar sin/cos, and `log1p(y_t)`.
   - **Model:** `RecurrentForecaster` (GRU or LSTM; unidirectional, stateless, last hidden state concatenated with the vector) inside a `ResidualOutput`, built by `build_recurrent_model`.
   - **Training:** L1 loss, gradient clipping 1.0, AdamW, `DEVICE` = cuda/mps/cpu with cuDNN set to deterministic.
   - **Grid:** set by `GRU_LOOKBACKS` (36, 72, 144), `GRU_HIDDEN` (32, 64, 128), `GRU_LAYERS` (1, 2) and `GRU_DROPOUT` (0.1, 0.2).
   - **Then:** shared vs separate models per horizon (`GRU_SHARED_MODEL`), 5 seeds, and an LSTM ablation (`train_recurrent`).
10. **Final comparison** (both rounds) and **"Quando o modelo de aprendizagem profunda falha?"**

## Results (test MAE, Wh; mean ± std over 5 seeds for networks; local CPU run)

| Model | 10 min | 60 min | RMSE 10 / 60 |
|---|---|---|---|
| Persistence | 26.51 | 46.02 | 66.5 / 101.3 |
| ARX | 27.08 | 40.27 | 59.4 / 78.4 |
| MLP, round 1 | 36.53 ± 3.28 | 52.38 ± 5.59 | 66.1 / 86.4 |
| CNN 1D, round 1 | 83.78 ± 18.53 | 91.37 ± 11.91 | 122.6 / 130.6 |
| **MLP (residual)** | **24.19 ± 0.16** | **35.35 ± 0.83** | 60.1 / 81.1 |
| CNN 1D (residual) | 24.95 ± 0.24 | 37.42 ± 0.58 | 60.8 / 82.6 |
| ARX + MLP | 24.55 ± 0.40 | 35.55 ± 0.75 | 59.3 / 81.7 |

- **Round 1:** the networks matched or beat the ARX on validation, but failed on test because of **covariate shift**. May is much warmer than training, the networks extrapolate the consumption level, and the seeds diverge.
- **Residual head:** it neutralizes this. Residual-model test MAE is close to its validation MAE (MLP residual val 24.6 / 37.6).
- **Remaining failure: spikes.**
  - Readings above 300 Wh are about 4% of the test set. On those, the residual MLP's MAE at 60 min is 320 Wh, against 296 for the ARX.
  - On readings up to 300 Wh it's 22.4, against 28.6 for the ARX.
  - The worst hours are 17–18 h. The residual nets underpredict on average (bias +6 / +12 Wh).
- **Hardware:** Ornella's original MLP numbers (37.5 ± 2.5 at 10 min) differ slightly from this run. The inputs are proven identical, so the difference comes from CPU hardware.

## Open items

- **Run the full GRU/LSTM search on Colab** (GPU). Locally only a tiny smoke grid has run. The "Resultados da GRU" markdown cell and the GRU rows of the final comparison are to be written from that run.
- Possible next steps:
  - a transformer with matched inputs, lookback, splits and parameter count (`prepare_gru` and the parameter counts can be reused);
  - something targeting the spikes (e.g. a quantile loss);
  - re-running the full notebook end-to-end on Colab for a single consistent set of outputs.

## Working notes

- **The notebook is generated** from builder scripts in the session scratchpad (`build2.py`, `v2_*.py`, `results_text.py`). It's the committed file that counts; edit the notebook directly from now on.
- The repo lives on the Windows drive (`/mnt/c/...`) and is opened in VS Code through WSL. When the notebook is edited from outside VS Code, the open tab does **not** refresh, and even "Revert File" doesn't help. Close the tab **without saving** and reopen it, or run "Developer: Reload Window". Saving the stale tab would overwrite the external edits.
- Before editing the notebook, check its modification time and current cell contents.
- **Runtime of a full run on this 2-core CPU:** about 31 minutes without the full GRU search.
