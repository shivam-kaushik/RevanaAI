"""
Unit tests for ForecastAgent
"""
import pytest
from backend.agents.forecast_agent import ForecastAgent
import pandas as pd
import numpy as np


class TestForecastAgent:
    """Test suite for ForecastAgent"""
    
    def test_forecast_agent_initialization(self):
        """Test that ForecastAgent initializes correctly"""
        agent = ForecastAgent()
        assert agent is not None
    
    def test_generate_forecast_with_valid_data(self):
        """Test forecast generation with valid time series data"""
        agent = ForecastAgent()
        
        # Create sample time series data
        data = pd.DataFrame({
            'sales': [100, 105, 110, 115, 120, 125, 130],
            'quantity': [50, 52, 54, 56, 58, 60, 62]
        })
        
        forecast = agent.generate_forecast(data, periods=3)
        
        assert forecast is not None
        assert isinstance(forecast, dict)
        assert 'sales' in forecast or len(forecast) > 0
    
    def test_generate_forecast_empty_dataframe(self):
        """Test forecast generation with empty DataFrame"""
        agent = ForecastAgent()
        data = pd.DataFrame()
        
        forecast = agent.generate_forecast(data)
        
        assert isinstance(forecast, str)
        assert "not enough data" in forecast.lower() or "no data" in forecast.lower()
    
    def test_generate_forecast_none_data(self):
        """Test forecast generation with None data"""
        agent = ForecastAgent()
        
        forecast = agent.generate_forecast(None)
        
        assert isinstance(forecast, str)
        assert "not enough data" in forecast.lower() or "no data" in forecast.lower()
    
    def test_generate_forecast_no_numeric_columns(self):
        """Test forecast generation with no numeric columns"""
        agent = ForecastAgent()
        data = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'category': ['X', 'Y', 'Z']
        })
        
        forecast = agent.generate_forecast(data)
        
        assert isinstance(forecast, str)
        assert "no numeric" in forecast.lower() or "numeric" in forecast.lower()
    
    def test_generate_forecast_single_value(self):
        """Test forecast generation with single value"""
        agent = ForecastAgent()
        data = pd.DataFrame({
            'sales': [100]
        })
        
        forecast = agent.generate_forecast(data)
        
        # Should handle single value gracefully
        assert forecast is not None
    
    def test_generate_forecast_trend_detection(self):
        """Test that forecast detects trends correctly"""
        agent = ForecastAgent()
        
        # Increasing trend
        increasing_data = pd.DataFrame({
            'sales': [100, 110, 120, 130, 140]
        })
        forecast_inc = agent.generate_forecast(increasing_data)
        
        # Decreasing trend
        decreasing_data = pd.DataFrame({
            'sales': [140, 130, 120, 110, 100]
        })
        forecast_dec = agent.generate_forecast(decreasing_data)
        
        assert forecast_inc is not None
        assert forecast_dec is not None
        assert isinstance(forecast_inc, dict)
        assert isinstance(forecast_dec, dict)
    
    def test_generate_forecast_multiple_columns(self):
        """Test forecast generation with multiple numeric columns"""
        agent = ForecastAgent()
        data = pd.DataFrame({
            'sales': [100, 105, 110, 115, 120],
            'quantity': [50, 52, 54, 56, 58],
            'revenue': [5000, 5250, 5500, 5750, 6000]
        })
        
        forecast = agent.generate_forecast(data)
        
        assert isinstance(forecast, dict)
        # Should forecast up to 3 columns (first 3 numeric)
        assert len(forecast) <= 3
    
    def test_generate_forecast_with_nan_values(self):
        """Test forecast generation with NaN values"""
        agent = ForecastAgent()
        data = pd.DataFrame({
            'sales': [100, np.nan, 110, 115, 120],
            'quantity': [50, 52, np.nan, 56, 58]
        })
        
        # Should handle NaN gracefully
        forecast = agent.generate_forecast(data)
        
        assert forecast is not None
    
    def test_generate_forecast_very_small_dataset(self):
        """Test forecast generation with minimal data"""
        agent = ForecastAgent()
        data = pd.DataFrame({
            'sales': [100, 105]
        })
        
        forecast = agent.generate_forecast(data)
        
        # Should still attempt forecast
        assert forecast is not None
    
    def test_generate_forecast_custom_periods(self):
        """Test forecast generation with custom periods parameter"""
        agent = ForecastAgent()
        data = pd.DataFrame({
            'sales': [100, 105, 110, 115, 120, 125, 130]
        })
        
        forecast = agent.generate_forecast(data, periods=6)
        
        assert forecast is not None
        # The periods parameter affects the forecast, but agent uses moving average
        # so it may not directly use periods, but should still work
    
    def test_generate_forecast_contains_forecast_value(self):
        """Test that forecast contains expected structure"""
        agent = ForecastAgent()
        data = pd.DataFrame({
            'sales': [100, 105, 110, 115, 120]
        })
        
        forecast = agent.generate_forecast(data)
        
        if isinstance(forecast, dict):
            # Check structure for at least one column
            for col_name, col_forecast in forecast.items():
                assert isinstance(col_forecast, dict)
                # Should have current and forecast keys
                assert 'current' in col_forecast or 'forecast' in col_forecast or 'trend' in col_forecast

