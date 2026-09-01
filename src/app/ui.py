"""Gradio UI for interactively testing the Rossmann model.

Uses ``gr.Blocks`` with raw, human-friendly inputs (store id, date, store
type, promo flags, etc.) and a Predict button that calls the same
``inference.predict_raw`` used by the FastAPI route. The confusing derived
features (one-hot encodings, lags, store aggregates) are computed
server-side from history.
"""

from __future__ import annotations

import gradio as gr

from src.app import inference
from src.app.features import ASSORTMENTS, PROMO_INTERVALS, STATE_HOLIDAYS, STORE_TYPES


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Rossmann Sales Forecasting") as blocks:
        gr.Markdown(
            """
            # Rossmann Store Sales Forecasting

            Predict daily sales for a store from its **raw attributes** —
            no need to fill in derived features like sales lags or one-hot
            encodings; those are computed automatically from store history.

            **Input ranges** (from training data): Store 1-1115, dates
            between 2013-01-08 and 2015-07-31.

            *Predictions are model output, not ground truth.*
            """
        )

        with gr.Row():
            with gr.Column():
                store = gr.Number(label="Store ID", value=1, minimum=1, maximum=1115, step=1)
                pred_date = gr.DateTime(
                    label="Prediction Date",
                    value="2015-01-01",
                    include_time=False,
                    type="string",
                )
                store_type = gr.Dropdown(
                    label="Store Type", choices=STORE_TYPES, value="b"
                )
                assortment = gr.Dropdown(
                    label="Assortment", choices=ASSORTMENTS, value="b"
                )
                state_holiday = gr.Dropdown(
                    label="State Holiday",
                    choices=STATE_HOLIDAYS,
                    value="0",
                    info="'0' = no state holiday",
                )
                school_holiday = gr.Checkbox(label="School Holiday", value=False)
                promo = gr.Checkbox(label="Promo running", value=False)

            with gr.Column():
                gr.Markdown("### Competition")
                comp_dist = gr.Number(
                    label="Competition Distance (m)",
                    value=None,
                    minimum=0,
                    info="Leave blank if there is no competitor",
                )
                comp_open_month = gr.Dropdown(
                    label="Competition opened (month)",
                    choices=list(range(1, 13)),
                    value=None,
                    info="Leave blank if there is no competitor",
                )
                comp_open_year = gr.Dropdown(
                    label="Competition opened (year)",
                    choices=[2013, 2014, 2015],
                    value=None,
                    info="Leave blank if there is no competitor",
                )

                gr.Markdown("### Promo 2 (continuous promo)")
                promo2 = gr.Checkbox(label="Store runs Promo 2", value=False)
                promo2_week = gr.Number(
                    label="Promo 2 start (ISO week)",
                    value=1,
                    minimum=1,
                    maximum=53,
                    step=1,
                    info="Leave blank if the store does not run Promo 2",
                )
                promo2_year = gr.Number(
                    label="Promo 2 start (year)",
                    value=2013,
                    minimum=2013,
                    maximum=2015,
                    step=1,
                    info="Leave blank if the store does not run Promo 2",
                )
                promo_interval = gr.Dropdown(
                    label="Promo 2 interval (months)",
                    choices=PROMO_INTERVALS,
                    value="None",
                    info="Which months Promo 2 runs",
                )

                predict_btn = gr.Button("Predict", variant="primary")
                output = gr.Number(label="Predicted Sales", interactive=False)

        def _predict(
            store: float,
            pred_date: str,
            store_type: str,
            assortment: str,
            state_holiday: str,
            school_holiday: bool,
            promo: bool,
            comp_dist: float | None,
            comp_open_month: int | None,
            comp_open_year: int | None,
            promo2: bool,
            promo2_week: float | None,
            promo2_year: float | None,
            promo_interval: str,
        ) -> float:
            from datetime import date

            pred = date.fromisoformat(pred_date)
            return inference.predict_raw(
                store=int(store),
                pred_date=pred,
                store_type=store_type,
                assortment=assortment,
                state_holiday=state_holiday,
                school_holiday=school_holiday,
                promo=promo,
                competition_distance=comp_dist,
                competition_open_since_month=comp_open_month,
                competition_open_since_year=comp_open_year,
                promo2=promo2,
                promo2_since_week=int(promo2_week) if promo2_week else None,
                promo2_since_year=int(promo2_year) if promo2_year else None,
                promo_interval=promo_interval,
            )["sales"]

        predict_btn.click(
            fn=_predict,
            inputs=[
                store,
                pred_date,
                store_type,
                assortment,
                state_holiday,
                school_holiday,
                promo,
                comp_dist,
                comp_open_month,
                comp_open_year,
                promo2,
                promo2_week,
                promo2_year,
                promo_interval,
            ],
            outputs=output,
        )

    return blocks
