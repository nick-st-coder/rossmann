"""Gradio UI for interactively testing the Rossmann model.

Uses ``gr.Blocks`` with one input widget per feature and a Predict button
that calls the same ``inference.predict`` used by the FastAPI route.
"""

from __future__ import annotations

import gradio as gr

from src.app import inference


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Rossmann Sales Forecasting") as blocks:
        gr.Markdown(
            """
            # Rossmann Store Sales Forecasting

            Predict daily sales for a store from its features. This model
            predicts **sales for open stores**; closed stores (Open == 0)
            are predicted as 0.

            **Input ranges** (from training data): Store 1-1115, DayOfWeek
            1-7, CompetitionDistance 0-75860, year 2013-2015.

            *Predictions are model output, not ground truth.*
            """
        )

        with gr.Row():
            with gr.Column():
                store = gr.Number(label="Store", value=1, minimum=1, maximum=1115, step=1)
                dow = gr.Slider(label="DayOfWeek", value=4, minimum=1, maximum=7, step=1)
                promo = gr.Radio(label="Promo", choices=[0, 1], value=0)
                school_holiday = gr.Radio(label="SchoolHoliday", choices=[0, 1], value=0)
                comp_dist = gr.Number(label="CompetitionDistance", value=720.0, minimum=0)
                promo2 = gr.Radio(label="Promo2", choices=[0, 1], value=0)
                promo2_week = gr.Number(label="Promo2SinceWeek", value=0.0, minimum=0)
                promo2_year = gr.Number(label="Promo2SinceYear", value=0.0, minimum=0)
                comp_missing = gr.Radio(label="CompetitionDistance_missing", choices=[0, 1], value=0)
                comp_age = gr.Number(label="competition_age_months", value=154.0, minimum=0)
                has_comp = gr.Radio(label="has_competition", choices=[0, 1], value=1)
                year = gr.Slider(label="year", value=2015, minimum=2013, maximum=2015, step=1)
                month = gr.Slider(label="month", value=1, minimum=1, maximum=12, step=1)
                day = gr.Slider(label="day", value=1, minimum=1, maximum=31, step=1)
                week = gr.Slider(label="week_of_year", value=1, minimum=1, maximum=53, step=1)
                promo2_active = gr.Radio(label="promo2_active", choices=[0, 1], value=0)
                promo2_age = gr.Number(label="promo2_age_months", value=0, minimum=0, step=1)
                store_mean = gr.Number(label="store_mean_sales", value=9744.6, minimum=0)
                store_mean_dow = gr.Number(label="store_mean_sales_dow", value=9829.86, minimum=0)
                day_of_month = gr.Slider(label="day_of_month", value=1, minimum=1, maximum=31, step=1)
                is_payday = gr.Radio(label="is_payday", choices=[0, 1], value=1)
                days_to_hol = gr.Number(label="days_to_holiday", value=0, minimum=0, step=1)
                days_since_hol = gr.Number(label="days_since_holiday", value=6, minimum=0, step=1)
                lag1 = gr.Number(label="sales_lag_1", value=7470.0, minimum=0)
                lag7 = gr.Number(label="sales_lag_7", value=9430.0, minimum=0)
                roll7 = gr.Number(label="sales_rolling_7", value=9405.57, minimum=0)

            with gr.Column():
                sh_0 = gr.Radio(label="StateHoliday_0", choices=[0, 1], value=0)
                sh_a = gr.Radio(label="StateHoliday_a", choices=[0, 1], value=1)
                sh_b = gr.Radio(label="StateHoliday_b", choices=[0, 1], value=0)
                sh_c = gr.Radio(label="StateHoliday_c", choices=[0, 1], value=0)
                st_a = gr.Radio(label="StoreType_a", choices=[0, 1], value=0)
                st_b = gr.Radio(label="StoreType_b", choices=[0, 1], value=1)
                st_c = gr.Radio(label="StoreType_c", choices=[0, 1], value=0)
                st_d = gr.Radio(label="StoreType_d", choices=[0, 1], value=0)
                as_a = gr.Radio(label="Assortment_a", choices=[0, 1], value=0)
                as_b = gr.Radio(label="Assortment_b", choices=[0, 1], value=1)
                as_c = gr.Radio(label="Assortment_c", choices=[0, 1], value=0)
                pi_feb = gr.Radio(label="PromoInterval_Feb_May_Aug_Nov", choices=[0, 1], value=0)
                pi_jan = gr.Radio(label="PromoInterval_Jan_Apr_Jul_Oct", choices=[0, 1], value=0)
                pi_mar = gr.Radio(label="PromoInterval_Mar_Jun_Sept_Dec", choices=[0, 1], value=0)
                pi_none = gr.Radio(label="PromoInterval_None", choices=[0, 1], value=1)

                predict_btn = gr.Button("Predict", variant="primary")
                output = gr.Number(label="Predicted Sales", interactive=False)

        def _predict(*values: float) -> float:
            names = [
                "Store", "DayOfWeek", "Promo", "SchoolHoliday", "CompetitionDistance",
                "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "CompetitionDistance_missing",
                "competition_age_months", "has_competition", "year", "month", "day",
                "week_of_year", "promo2_active", "promo2_age_months", "store_mean_sales",
                "store_mean_sales_dow", "day_of_month", "is_payday", "days_to_holiday",
                "days_since_holiday", "sales_lag_1", "sales_lag_7", "sales_rolling_7",
                "StateHoliday_0", "StateHoliday_a", "StateHoliday_b", "StateHoliday_c",
                "StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d",
                "Assortment_a", "Assortment_b", "Assortment_c",
                "PromoInterval_Feb_May_Aug_Nov", "PromoInterval_Jan_Apr_Jul_Oct",
                "PromoInterval_Mar_Jun_Sept_Dec", "PromoInterval_None",
            ]
            features = dict(zip(names, values, strict=True))
            return inference.predict(features)["sales"]

        predict_btn.click(
            fn=_predict,
            inputs=[
                store, dow, promo, school_holiday, comp_dist, promo2, promo2_week,
                promo2_year, comp_missing, comp_age, has_comp, year, month, day, week,
                promo2_active, promo2_age, store_mean, store_mean_dow, day_of_month,
                is_payday, days_to_hol, days_since_hol, lag1, lag7, roll7,
                sh_0, sh_a, sh_b, sh_c, st_a, st_b, st_c, st_d,
                as_a, as_b, as_c, pi_feb, pi_jan, pi_mar, pi_none,
            ],
            outputs=output,
        )

    return blocks
