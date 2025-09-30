from backend.config import Config
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyAgent:
    def __init__(self):
        print("🤖 ANOMALY_AGENT: Initialized")
    
    def detect_anomalies(self, data):
        """Detect anomalies in the data using Isolation Forest"""
        print("🤖 ANOMALY_AGENT: Detecting anomalies in data")
        
        try:
            if data is None or data.empty or len(data) < 5:
                return "Not enough data for anomaly detection"
            
            # Select numeric columns for anomaly detection
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return "No numeric data available for anomaly detection"
            
            # Handle missing values
            numeric_data = numeric_data.fillna(numeric_data.mean())
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(numeric_data)
            
            # Count anomalies
            n_anomalies = (anomalies == -1).sum()
            anomaly_percentage = (n_anomalies / len(anomalies)) * 100
            
            result = {
                'total_records': len(data),
                'anomalies_detected': n_anomalies,
                'anomaly_percentage': round(anomaly_percentage, 2),
                'message': f"Detected {n_anomalies} anomalies ({anomaly_percentage:.1f}% of data)"
            }
            
            print(f"✅ ANOMALY_AGENT: {result['message']}")
            return result
            
        except Exception as e:
            print(f"❌ ANOMALY_AGENT Error: {e}")
            return f"Anomaly detection failed: {str(e)}"