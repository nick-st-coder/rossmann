from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.app import inference
from src.app.ui import build_ui

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Load the model once at startup; fail fast if it cannot load."""
    inference.load()
    yield


app = FastAPI(title="Rossmann Sales Forecasting", version="0.1.0", lifespan=lifespan)

class PredictRequest(BaseModel):
    Store: int = Field(..., ge=1, description="Store id.")
    DayOfWeek: int = Field(..., ge=1, le=7, description="Day of week (1-7).")
    Promo: int = Field(..., ge=0, le=1, description="Promo running (0/1).")
    SchoolHoliday: int = Field(..., ge=0, le=1, description="School holiday (0/1).")
    CompetitionDistance: float = Field(..., ge=0, description="Distance to nearest competitor.")
    Promo2: int = Field(..., ge=0, le=1, description="Store participates in promo2 (0/1).")
    Promo2SinceWeek: float = Field(..., ge=0, description="Promo2 start week.")
    Promo2SinceYear: float = Field(..., ge=0, description="Promo2 start year.")
    CompetitionDistance_missing: int = Field(..., ge=0, le=1, description="CompetitionDistance was missing (0/1).")
    competition_age_months: float = Field(..., ge=0, description="Months since competition opened.")
    has_competition: int = Field(..., ge=0, le=1, description="Store has a competitor (0/1).")
    year: int = Field(..., ge=2013, le=2015, description="Prediction year.")
    month: int = Field(..., ge=1, le=12, description="Prediction month.")
    day: int = Field(..., ge=1, le=31, description="Prediction day.")
    week_of_year: int = Field(..., ge=1, le=53, description="ISO week of year.")
    promo2_active: int = Field(..., ge=0, le=1, description="Promo2 active this month (0/1).")
    promo2_age_months: int = Field(..., ge=0, description="Months since promo2 started.")
    store_mean_sales: float = Field(..., ge=0, description="Store historical mean sales.")
    store_mean_sales_dow: float = Field(..., ge=0, description="Store mean sales for this day-of-week.")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month.")
    is_payday: int = Field(..., ge=0, le=1, description="Payday (1st/15th) flag (0/1).")
    days_to_holiday: int = Field(..., ge=0, description="Days until next state holiday.")
    days_since_holiday: int = Field(..., ge=0, description="Days since last state holiday.")
    sales_lag_1: float = Field(..., ge=0, description="Sales one day ago.")
    sales_lag_7: float = Field(..., ge=0, description="Sales seven days ago.")
    sales_rolling_7: float = Field(..., ge=0, description="7-day rolling mean sales.")
    StateHoliday_0: int = Field(..., ge=0, le=1, description="StateHoliday == 0 (0/1).")
    StateHoliday_a: int = Field(..., ge=0, le=1, description="StateHoliday == a (0/1).")
    StateHoliday_b: int = Field(..., ge=0, le=1, description="StateHoliday == b (0/1).")
    StateHoliday_c: int = Field(..., ge=0, le=1, description="StateHoliday == c (0/1).")
    StoreType_a: int = Field(..., ge=0, le=1, description="StoreType == a (0/1).")
    StoreType_b: int = Field(..., ge=0, le=1, description="StoreType == b (0/1).")
    StoreType_c: int = Field(..., ge=0, le=1, description="StoreType == c (0/1).")
    StoreType_d: int = Field(..., ge=0, le=1, description="StoreType == d (0/1).")
    Assortment_a: int = Field(..., ge=0, le=1, description="Assortment == a (0/1).")
    Assortment_b: int = Field(..., ge=0, le=1, description="Assortment == b (0/1).")
    Assortment_c: int = Field(..., ge=0, le=1, description="Assortment == c (0/1).")
    PromoInterval_Feb_May_Aug_Nov: int = Field(..., ge=0, le=1, description="PromoInterval Feb/May/Aug/Nov (0/1).")
    PromoInterval_Jan_Apr_Jul_Oct: int = Field(..., ge=0, le=1, description="PromoInterval Jan/Apr/Jul/Oct (0/1).")
    PromoInterval_Mar_Jun_Sept_Dec: int = Field(..., ge=0, le=1, description="PromoInterval Mar/Jun/Sept/Dec (0/1).")
    PromoInterval_None: int = Field(..., ge=0, le=1, description="No promo interval (0/1).")

class PredictResponse(BaseModel):
    # Response body for a prediction.
    sales: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = inference.predict(req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return PredictResponse(**result)


# Mount the Gradio UI in the same process/port.
blocks = build_ui()
app = gr.mount_gradio_app(app, blocks, path="/ui")
