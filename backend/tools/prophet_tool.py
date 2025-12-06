"""
from prophet import Prophet

class ProphetTool:
    \"""Fits Prophet models and generates forecasts.\"""

    def __init__(self):
        self.model = None

    def run(self, df, horizon: int = 3):
        if "ds" not in df or "y" not in df:
            raise ValueError("DataFrame must have columns 'ds' and 'y'")
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        return model, forecast


# backend/tools/prophet_tool.py
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except Exception:
    _PROPHET_AVAILABLE = False

class ProphetTool:
    """
    Prophet wrapper for monthly forecasting (no logistic).
    - Infers datetime/value cols
    - Resamples to monthly sums ('MS')
    - Drops partial current month
    - Detects & drops a crater last month (optional)
    - Requires >= 12 monthly points
    - Linear growth + multiplicative seasonality
    """

    _TIME_CANDIDATES = [
        "ds", "date", "Date",
        "invoice_date", "InvoiceDate",
        "timestamp", "Timestamp",
        "order_date", "OrderDate",
    ]

    def __init__(self):
        pass

    # -------------------- PUBLIC API --------------------

    def run(
        self,
        df: pd.DataFrame,
        periods: int = 3,
        time_col: Optional[str] = None,
        value_col: Optional[str] = None,
        freq: str = "MS",
        drop_partial_current_month: bool = True,
        drop_crater_last_bucket: bool = True,
        crater_ratio_threshold: float = 0.65,   # last point < 65% of median(last 12) → drop
        clip_nonnegative: bool = True,
        # compatibility with forecast_agent.py
        horizon: Optional[int] = None,
    ) -> Tuple[Prophet, pd.DataFrame]:

        if horizon is not None:
            periods = int(horizon)

        if df is None or df.empty:
            raise ValueError("ProphetTool: empty input DataFrame.")

        if not _PROPHET_AVAILABLE:
            raise RuntimeError("Prophet is not installed. Please add 'prophet' to requirements.txt and install.")

        history = self._prepare_monthly_series(
            df=df,
            time_col=time_col,
            value_col=value_col,
            drop_partial_current_month=drop_partial_current_month,
            drop_crater_last_bucket=drop_crater_last_bucket,
            crater_ratio_threshold=crater_ratio_threshold,
        )

        if history["y"].dropna().shape[0] < 12:
            raise ValueError(f"ProphetTool: insufficient history ({history['y'].dropna().shape[0]} points). Need >= 12 months.")

        model, forecast = self._fit_predict_linear(history, periods=periods, freq=freq)

        if clip_nonnegative and not forecast.empty:
            cols = [c for c in ["yhat", "yhat_lower", "yhat_upper"] if c in forecast.columns]
            if cols:
                forecast[cols] = forecast[cols].clip(lower=0)

        return model, forecast

    # -------------------- PREP HELPERS --------------------

    def _infer_time_col(self, df: pd.DataFrame, time_col: Optional[str]) -> str:
        if time_col and time_col in df.columns:
            return time_col
        for c in self._TIME_CANDIDATES:
            if c in df.columns:
                return c
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                continue
        raise ValueError("ProphetTool: no datetime-like column found (expected ds/date/invoice_date/timestamp etc.).")

    def _pick_value_col(self, df: pd.DataFrame, time_col: str, value_col: Optional[str]) -> str:
        if value_col and value_col in df.columns:
            return value_col
        # If already shaped as ds/y, honor it
        if {"ds", "y"}.issubset(df.columns):
            return "y"
        numeric = list(df.select_dtypes(include=["number"]).columns)
        if time_col in numeric:
            numeric.remove(time_col)
        if not numeric:
            raise ValueError("ProphetTool: no numeric column to forecast.")
        return numeric[0]

    def _prepare_monthly_series(
        self,
        df: pd.DataFrame,
        time_col: Optional[str],
        value_col: Optional[str],
        drop_partial_current_month: bool,
        drop_crater_last_bucket: bool,
        crater_ratio_threshold: float,
    ) -> pd.DataFrame:

        tcol = self._infer_time_col(df, time_col)
        vcol = self._pick_value_col(df, tcol, value_col)

        work = df[[tcol, vcol]].copy()
        if not pd.api.types.is_datetime64_any_dtype(work[tcol]):
            work[tcol] = pd.to_datetime(work[tcol], errors="coerce")
        work = work.dropna(subset=[tcol]).set_index(tcol).sort_index()

        monthly = work.resample("MS").sum()
        monthly = monthly.rename_axis("ds").reset_index().rename(columns={vcol: "y"})
        monthly["y"] = pd.to_numeric(monthly["y"], errors="coerce").fillna(0.0)

        # Drop partial current month (prevents the “crater” when you run mid-month)
        if drop_partial_current_month and not monthly.empty:
            this_month_start = pd.Timestamp.today().to_period("M").start_time
            monthly = monthly[monthly["ds"] < this_month_start]

        # Optional: detect & drop a crater final bucket (e.g., last month << typical level)
        if drop_crater_last_bucket and monthly.shape[0] >= 13:
            last_idx = monthly.index.max()
            last_val = float(monthly.loc[last_idx, "y"])
            recent = monthly.iloc[-13:-1]["y"]  # previous 12 months
            med = float(recent.median()) if not recent.empty else None
            if med and med > 0 and last_val < crater_ratio_threshold * med:
                monthly = monthly.iloc[:-1]  # drop last anomalous point

        # no negatives for business series
        monthly.loc[monthly["y"] < 0, "y"] = 0.0

        return monthly.sort_values("ds")

    # -------------------- MODEL --------------------

    def _fit_predict_linear(self, history: pd.DataFrame, periods: int, freq: str) -> Tuple[Prophet, pd.DataFrame]:
        """
        Linear trend, multiplicative seasonality, tighter changepoint prior to avoid over-steep trends.
        Suitable for monthly business series without going negative.
        """
        m = Prophet(
            growth="linear",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,   # slightly conservative trend changes
            seasonality_prior_scale=10.0,   # default; ok for monthly
            interval_width=0.95,
        )
        m.fit(history)

        future = m.make_future_dataframe(periods=periods, freq=freq)
        fcst = m.predict(future)
        return m, fcst
