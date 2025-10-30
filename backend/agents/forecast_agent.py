
from backend.config import Config
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    # Prophet was renamed from fbprophet; in requirements it's 'prophet'
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except Exception:
    _PROPHET_AVAILABLE = False


class ForecastAgent:
    """Time-series forecasting using Prophet with uncertainty intervals.

    Expected input:
        - data: Pandas DataFrame that includes a datetime column and one or more numeric series
        - periods: number of future periods to forecast (default: 1)

    Behavior:
        - Auto-detects a datetime column if not provided (tries: 'ds', 'date', 'Date', 'invoice_date', 'InvoiceDate')
        - Resamples to monthly frequency (start of month) unless data already looks monthly
        - Requires at least 12 historical points; otherwise returns an explanatory message
        - Returns yhat, yhat_lower, yhat_upper for each value column
    """
    def __init__(self):
        print("🤖 FORECAST_AGENT: Initialized (Prophet-ready)")

    def _infer_time_col(self, df: pd.DataFrame, time_col: Optional[str] = None) -> Optional[str]:
        if time_col and time_col in df.columns:
            return time_col
        candidates = ["ds", "date", "Date", "invoice_date", "InvoiceDate", "timestamp", "Timestamp"]
        for c in candidates:
            if c in df.columns:
                return c
        # fallback: try first datetime-like column
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                continue
        return None

    def _ensure_datetime(self, s: pd.Series) -> pd.Series:
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce")
        return s

    def _resample_monthly(self, df: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
        ts = df[[time_col, value_col]].copy()
        ts[time_col] = self._ensure_datetime(ts[time_col])
        ts = ts.dropna(subset=[time_col])
        ts = ts.set_index(time_col).sort_index()
        # If index is not monthly, resample to month start using sum (typical for revenue)
        monthly = ts.resample('MS').sum()
        monthly = monthly.rename_axis("ds").reset_index().rename(columns={value_col: "y"})
        return monthly

    def _is_enough_history(self, df: pd.DataFrame) -> bool:
        # Require at least 12 non-null points
        return df['y'].dropna().shape[0] >= 12

    def _fit_forecast(self, history: pd.DataFrame, periods: int, freq: str = 'MS') -> pd.DataFrame:
        m = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(history)
        future = m.make_future_dataframe(periods=periods, freq=freq)
        fcst = m.predict(future)
        # Return only the forecast horizon rows
        return fcst.loc[fcst['ds'] > history['ds'].max(), ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def generate_forecast(
        self,
        data: pd.DataFrame,
        periods: int = 1,
        time_col: Optional[str] = None,
        value_cols: Optional[List[str]] = None,
        freq: str = 'MS'
    ):
        print(f"🤖 FORECAST_AGENT: Prophet forecast for {periods} period(s)")
        try:
            if data is None or data.empty:
                return {"status": "error", "message": "Not enough data for forecasting"}

            if not _PROPHET_AVAILABLE:
                return {"status": "error", "message": "Prophet library is not available. Please install 'prophet' per requirements.txt."}

            # Detect time column
            tcol = self._infer_time_col(data, time_col)
            if tcol is None:
                return {"status": "error", "message": "No datetime column found (expected one of: ds, date, Date, invoice_date, InvoiceDate)."}

            # Choose value columns
            if value_cols is None:
                numeric_cols = list(data.select_dtypes(include=['number']).columns)
                if tcol in numeric_cols:
                    numeric_cols.remove(tcol)
                if not numeric_cols:
                    return {"status": "error", "message": "No numeric columns found to forecast."}
                value_cols = numeric_cols

            results: Dict[str, dict] = {}
            for col in value_cols:
                # Prepare monthly series
                monthly = self._resample_monthly(data, tcol, col)
                if not self._is_enough_history(monthly):
                    results[col] = {
                        "status": "insufficient_data",
                        "message": "Not enough history to forecast reliably (need >= 12 monthly points).",
                        "history_points": int(monthly['y'].dropna().shape[0])
                    }
                    continue

                # Fit Prophet and forecast
                fcst = self._fit_forecast(monthly, periods=periods)
                # Extract last actual
                last_actual_ds = monthly['ds'].max()
                last_actual_y = float(monthly.loc[monthly['ds'] == last_actual_ds, 'y'].values[0])

                # Package
                points = [
                    {
                        "ds": r['ds'].strftime('%Y-%m-%d'),
                        "yhat": float(r['yhat']),
                        "yhat_lower": float(r['yhat_lower']),
                        "yhat_upper": float(r['yhat_upper'])
                    }
                    for _, r in fcst.iterrows()
                ]

                # Trend signal: compare last forecast to last actual
                final = points[-1] if points else None
                trend = None
                if final is not None:
                    if final['yhat'] > last_actual_y:
                        trend = 'increasing'
                    elif final['yhat'] < last_actual_y:
                        trend = 'decreasing'
                    else:
                        trend = 'flat'

                results[col] = {
                    "status": "ok",
                    "freq": "MS",
                    "history_points": int(monthly.shape[0]),
                    "last_actual": {"ds": last_actual_ds.strftime('%Y-%m-%d'), "y": last_actual_y},
                    "forecast_horizon": periods,
                    "forecast": points,
                    "trend": trend
                }

            print(f"✅ FORECAST_AGENT: Forecast generated for {len([k for k,v in results.items() if v.get('status')=='ok'])} series")
            return results

        except Exception as e:
            print(f"❌ FORECAST_AGENT Error: {e}")
            return {"status": "error", "message": f"Forecasting failed: {str(e)}"}
