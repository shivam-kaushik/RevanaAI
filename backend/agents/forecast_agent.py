from backend.config import Config
import pandas as pd
from datetime import datetime, timedelta

class ForecastAgent:
    def __init__(self):
        print("🤖 FORECAST_AGENT: Initialized")
    
    def generate_forecast(self, data, periods=6):
        """Generate simple forecasts using moving averages"""
        print(f"🤖 FORECAST_AGENT: Generating forecast for {periods} periods")
        
        try:
            # Simple forecasting logic (can be enhanced with Prophet later)
            if data is None or data.empty:
                return "Not enough data for forecasting"
            
            # Check if we have time series data
            numeric_columns = data.select_dtypes(include=['number']).columns
            
            if len(numeric_columns) == 0:
                return "No numeric data available for forecasting"
            
            # Simple moving average forecast
            forecast_results = {}
            for col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                series = data[col].dropna()
                if len(series) > 1:
                    # Simple moving average
                    forecast_value = series.rolling(window=min(3, len(series))).mean().iloc[-1]
                    forecast_results[col] = {
                        'current': series.iloc[-1] if len(series) > 0 else 0,
                        'forecast': forecast_value,
                        'trend': 'increasing' if forecast_value > series.iloc[-1] else 'decreasing'
                    }
            
            print(f"✅ FORECAST_AGENT: Forecast generated for {len(forecast_results)} series")
            return forecast_results
            
        except Exception as e:
            print(f"❌ FORECAST_AGENT Error: {e}")
            return f"Forecasting failed: {str(e)}"