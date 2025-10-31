"""
Unit tests for AnomalyAgent
"""
import pytest
from backend.agents.anomaly_agent import AnomalyAgent
import pandas as pd
import numpy as np


class TestAnomalyAgent:
    """Test suite for AnomalyAgent"""
    
    def test_anomaly_agent_initialization(self):
        """Test that AnomalyAgent initializes correctly"""
        agent = AnomalyAgent()
        assert agent is not None
    
    def test_detect_anomalies_with_valid_data(self):
        """Test anomaly detection with valid numeric data"""
        agent = AnomalyAgent()
        
        # Create sample data with some outliers
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 100)
        outlier_data = np.array([500, 600, -200])  # Clear outliers
        combined = np.concatenate([normal_data, outlier_data])
        
        data = pd.DataFrame({
            'sales': combined,
            'quantity': np.random.normal(50, 5, 103)
        })
        
        result = agent.detect_anomalies(data)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'total_records' in result
        assert 'anomalies_detected' in result
        assert 'anomaly_percentage' in result
        assert 'message' in result
        assert result['total_records'] == 103
    
    def test_detect_anomalies_empty_dataframe(self):
        """Test anomaly detection with empty DataFrame"""
        agent = AnomalyAgent()
        data = pd.DataFrame()
        
        result = agent.detect_anomalies(data)
        
        assert isinstance(result, str)
        assert "not enough data" in result.lower()
    
    def test_detect_anomalies_none_data(self):
        """Test anomaly detection with None data"""
        agent = AnomalyAgent()
        
        result = agent.detect_anomalies(None)
        
        assert isinstance(result, str)
        assert "not enough data" in result.lower()
    
    def test_detect_anomalies_too_small_dataset(self):
        """Test anomaly detection with insufficient data"""
        agent = AnomalyAgent()
        data = pd.DataFrame({
            'sales': [100, 105, 110]  # Only 3 rows
        })
        
        result = agent.detect_anomalies(data)
        
        assert isinstance(result, str)
        assert "not enough" in result.lower()
    
    def test_detect_anomalies_no_numeric_columns(self):
        """Test anomaly detection with no numeric columns"""
        agent = AnomalyAgent()
        data = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D', 'E'],
            'category': ['X', 'Y', 'Z', 'X', 'Y']
        })
        
        result = agent.detect_anomalies(data)
        
        assert isinstance(result, str)
        assert "no numeric" in result.lower() or "numeric" in result.lower()
    
    def test_detect_anomalies_with_nan_values(self):
        """Test anomaly detection with NaN values"""
        agent = AnomalyAgent()
        data = pd.DataFrame({
            'sales': [100, 105, np.nan, 110, 115, 500],  # NaN and outlier
            'quantity': [50, 52, 54, 56, 58, 600]  # Outlier
        })
        
        result = agent.detect_anomalies(data)
        
        # Should handle NaN by filling with mean
        assert result is not None
        if isinstance(result, dict):
            assert result['total_records'] == 6
    
    def test_detect_anomalies_result_structure(self):
        """Test that anomaly detection result has correct structure"""
        agent = AnomalyAgent()
        
        np.random.seed(42)
        data = pd.DataFrame({
            'sales': np.random.normal(100, 10, 50),
            'quantity': np.random.normal(50, 5, 50)
        })
        
        result = agent.detect_anomalies(data)
        
        assert isinstance(result, dict)
        assert 'total_records' in result
        assert 'anomalies_detected' in result
        assert 'anomaly_percentage' in result
        assert 'message' in result
        
        # Check types
        assert isinstance(result['total_records'], (int, np.integer))
        assert isinstance(result['anomalies_detected'], (int, np.integer))
        assert isinstance(result['anomaly_percentage'], (float, np.floating))
        assert isinstance(result['message'], str)
    
    def test_detect_anomalies_percentage_calculation(self):
        """Test that anomaly percentage is calculated correctly"""
        agent = AnomalyAgent()
        
        np.random.seed(42)
        data = pd.DataFrame({
            'sales': np.random.normal(100, 10, 100)
        })
        
        result = agent.detect_anomalies(data)
        
        if isinstance(result, dict):
            total = result['total_records']
            anomalies = result['anomalies_detected']
            percentage = result['anomaly_percentage']
            
            # Percentage should be approximately (anomalies / total) * 100
            expected_percentage = (anomalies / total) * 100
            assert abs(percentage - expected_percentage) < 0.1  # Allow small rounding differences
    
    def test_detect_anomalies_message_format(self):
        """Test that anomaly message is properly formatted"""
        agent = AnomalyAgent()
        
        np.random.seed(42)
        data = pd.DataFrame({
            'sales': np.random.normal(100, 10, 50)
        })
        
        result = agent.detect_anomalies(data)
        
        if isinstance(result, dict):
            message = result['message']
            assert isinstance(message, str)
            assert str(result['anomalies_detected']) in message
            assert 'detected' in message.lower() or 'anomal' in message.lower()
    
    def test_detect_anomalies_multiple_numeric_columns(self):
        """Test anomaly detection with multiple numeric columns"""
        agent = AnomalyAgent()
        
        np.random.seed(42)
        data = pd.DataFrame({
            'sales': np.random.normal(100, 10, 50),
            'quantity': np.random.normal(50, 5, 50),
            'revenue': np.random.normal(5000, 500, 50)
        })
        
        result = agent.detect_anomalies(data)
        
        # Should process all numeric columns
        assert result is not None
        if isinstance(result, dict):
            assert result['total_records'] == 50
    
    def test_detect_anomalies_isolates_outliers(self):
        """Test that obvious outliers are detected"""
        agent = AnomalyAgent()
        
        # Create data with clear outliers
        normal_values = np.random.normal(100, 10, 95)
        outliers = np.array([1000, 2000, -500, -1000, 5000])  # Clear outliers
        combined = np.concatenate([normal_values, outliers])
        
        data = pd.DataFrame({
            'sales': combined
        })
        
        result = agent.detect_anomalies(data)
        
        if isinstance(result, dict):
            # Should detect at least some anomalies
            assert result['anomalies_detected'] > 0
            assert result['anomaly_percentage'] > 0

