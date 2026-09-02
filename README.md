# Rossmann Store Sales Forecasting

In this project I forecast daily sales for 1,115 Rossmann drugstores across Germany from
historical store/date features, so store managers can plan staffing,
delivery, and promotions ahead of time.

I solved this task with XGBoost trained on pandas features, validated
chronologically, and served behind a FastAPI + Gradio app.

---

## The business problems I solved

Rossmann's daily sales are driven by many interacting factors: promotions,
competition, school and state holidays, seasonality, and store location.
With ~1,115 stores and six weeks of forecasts needed at a time, manually
finding these patterns is imposible — too many factors for a human to weigh
at once.

A reliable model lets managers to:
- **Schedule staff** to match expected traffic, not guess it.
- **Plan deliveries and stock** for high- and low-sales days.
- **Run promotions** at the times they move the needle most.

The deliverable is a 6-week daily sales forecast per store.

---

## Key decisions & The problems faced

### 1. Target leakage — `Customers` feature had to go.
`Customers` (same-day visitor count) correlated `~0.90–0.996` with `Sales`
— in log space it was essentially the target itself, and it isn't known at
prediction time. Keeping it would let the model "cheat" and inflate every
metric. It was `dropped in cleaning` so the dataset is final before
modeling.

### 2. The structural-zero problem — `Open == 0`
Closed stores have **zero sales deterministically** (`Open == 0 ⇒ Sales == 0`).
Training the regressor on those rows forced the model to learn a structural
zero it could never predict well, inflating error and distorting metrics.

**Fix:** a two-stage split — `train the regressor on open rows only`,
predict closed rows as exactly `0`, and clip all predictions at `0`.

### 3. Honest chronological validation

The data is time-ordered, and the dataset has ~1,115 rows per date, so a
row-index `TimeSeriesSplit` would mix dates across folds. Validation switched
to **date-based expanding windows with a real 7-day gap** between train and
validation, so every fold is a clean time window with no future leakage and
CV reflects real production behavior.

### 4. Feature engineering (each signal justified)

- **Promo2 activity** — raw columns only said *when promo2 started*, not
  whether it was running on a given date. Added `promo2_active` and
  `promo2_age_months`.
- **Store-level aggregates** — `store_mean_sales` and `store_mean_sales_dow`
  give the model a direct measure of each store's typical sales level.
- **Payday effect** — `is_payday` (1st and 15th) plus `day_of_month`.
- **Holiday proximity** — `days_to_holiday` / `days_since_holiday`.
- **Lags** — `sales_lag_1`, `sales_lag_7`, `sales_rolling_7`.

### 5. Missing values

- **Promo2** columns → `0` (store never ran the second promo).
- **CompetitionDistance** → median fill (outlier-robust) **plus a missingness
  indicator** so the model can learn the "no competitor" case.
- Competition open date → derived into `competition_age_months` + `has_competition`.

---

## Trade-offs

| Choice | What I gained | What it cost |
|---|---|---|
| **XGBoost over Ridge** | Crushed the linear baseline on every metric; comparable to LightGBM but better on the primary metric (RMSLE) and ~23% faster to train | Slower to train than Ridge (though still seconds) |
| **Two-stage Open handling** | Removed the deterministic-zero noise; the single biggest metric gain | Model can't learn *why* a store closed — must rely on the `Open` flag |
| **RMSLE as primary metric** | Rewards relative accuracy (100→200 is treated as worse than 1000→1100) | Less intuitive to stakeholders than raw MAE |
| **Chronological date-based CV** | Honest, leak-free validation that mirrors production | Slightly more complex than row-index CV; fewer effective training windows |
| **Store-level aggregates & lags** | Big predictive lift from each store's history | Lag features drop the first ~7 days per store from training |
| **Multicollinearity left in** | Features like `year/month/week` and `Promo2` groups are near-duplicates (VIF up to ~1.5M) | Harmless for tree models, so not worth the effort of removing |
| **Stop at RMSLE 0.116** | Clean, low-complexity model with no overfitting left to fix | Left ensembling / store-grouped models (est. 1–3% each) on the table |

---

## Results

The last meaningful gains came from **structural fixes, not tuning harder**:

| Run | test_RMSLE | gap_RMSLE |
|---|---|---|
| First model | 1.073 | +0.057 |
| **Final model** | **0.116** | **−0.011** |

Final metrics on the held-out test set (chronological, after the 2015 cutoff):

| Metric | Train | Test | Gap |
|---|---|---|---|
| MAE | 457.1 | 508.5 | +51.3 |
| RMSE | 730.2 | 805.5 | +75.3 |
| RMSLE | 0.1278 | **0.1164** | **−0.0114** |

The **negative RMSLE gap** is the key signal: the model generalizes *better*
than it fits — it is not overfitting, so adding regularization would only hurt.

---

## The pipeline

```
raw train.csv + store.csv
      │  merge on Store
      ▼
Cleaning & feature engineering   (notebooks/Cleaning.ipynb, src/data/loaders.py)
      │  fill missing, derive features, one-hot encode, drop leakage
      ▼
rossmann.csv (model-ready)
      │  chronological split (cutoff 2015-01-01)
      ▼
Two-stage modeling               (src/models/train.py)
      │  regressor on open rows; closed rows → 0
      ▼
XGBoost  →  MLflow registry  →  FastAPI + Gradio serving (src/app/, docker)
```

---

## Repo layout
- `notebooks/` — `EDA.ipynb` → `Cleaning.ipynb` → `Modeling.ipynb` (exploration, cleaning, and modeling).
- `src/data/` — loaders, preprocessing, chronological splits.
- `src/features/` — feature engineering logic.
- `src/models/` — training, tuning, and prediction pipelines.
- `src/evaluation/` — MAE / RMSE / RMSLE metrics and cross-validation.
- `src/tracking/` — MLflow experiment/run logging.
- `src/app/` — FastAPI + Gradio serving app.
- `tests/` — unit tests for the pipeline.
- `dockerfile` / `docker/` — containerized serving.

Environment is managed with **`uv`** (`uv sync`, `uv run`); models are
registered in **MLflow**.

---

## Getting started

If you want to try it by yourself do:

```bash
uv sync                       # install deps from uv.lock
uv run jupyter notebook       # explore notebooks/ (EDA → Cleaning → Modeling)
uv run uvicorn src.app.main:app --reload   # run the serving app
```