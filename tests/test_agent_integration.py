"""
Integration tests for agents working together
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from backend.agents.planner import PlannerAgent
from backend.agents.sql_agent import SQLAgent
from backend.agents.analysis_agent import AnalysisAgent
from backend.agents.forecast_agent import ForecastAgent
from backend.agents.anomaly_agent import AnomalyAgent


class TestAgentIntegration:
    """Integration tests for agents working together"""
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.sql_agent.DatasetManager')
    @patch('backend.agents.sql_agent.db_manager')
    @patch('backend.agents.sql_agent.OpenAI')
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_sql_and_insight_agent_integration(self, mock_analysis_openai, mock_sql_openai, 
                                               mock_db, mock_dataset_manager, mock_vector_db):
        """Test SQL Agent and Analysis Agent working together"""
        # Setup Planner
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        planner = PlannerAgent()
        
        # Setup SQL Agent
        mock_dataset = Mock()
        mock_dataset.get_active_dataset.return_value = {'table_name': 'test_table'}
        mock_dataset_manager.return_value = mock_dataset
        
        mock_db.execute_query_dict.return_value = [
            {'column_name': 'id', 'data_type': 'integer'},
            {'column_name': 'name', 'data_type': 'text'},
            {'column_name': 'quantity', 'data_type': 'integer'}
        ]
        
        # Mock SQL generation
        mock_sql_client = Mock()
        mock_sql_response = Mock()
        mock_sql_response.choices = [Mock()]
        mock_sql_response.choices[0].message = Mock()
        mock_sql_response.choices[0].message.content = "SELECT * FROM test_table LIMIT 10"
        mock_sql_client.chat.completions.create.return_value = mock_sql_response
        mock_sql_openai.return_value = mock_sql_client
        
        # Mock SQL execution result
        mock_df = pd.DataFrame({
            'product_name': ['Laptop', 'Mouse'],
            'quantity': [10, 50],
            'price': [999.99, 29.99]
        })
        mock_db.execute_query = Mock(return_value=mock_df)
        
        # Mock Analysis Agent
        mock_analysis_client = Mock()
        mock_analysis_response = Mock()
        mock_analysis_response.choices = [Mock()]
        mock_analysis_response.choices[0].message = Mock()
        mock_analysis_response.choices[0].message.content = "Key insights: Laptops have high sales"
        mock_analysis_client.chat.completions.create.return_value = mock_analysis_response
        mock_analysis_openai.return_value = mock_analysis_client
        
        # Execute workflow
        sql_agent = SQLAgent()
        analysis_agent = AnalysisAgent()
        
        # Step 1: Generate SQL
        sql_query, error = sql_agent.generate_sql("What are the top products?")
        assert sql_query is not None
        assert error is None
        
        # Step 2: Execute SQL (simulated)
        data_results = mock_df  # Use mocked DataFrame
        
        # Step 3: Generate insights
        insights = analysis_agent.generate_insights("What are the top products?", data_results.to_dict('records'))
        assert insights is not None
        assert isinstance(insights, str)
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.sql_agent.DatasetManager')
    @patch('backend.agents.sql_agent.db_manager')
    @patch('backend.agents.sql_agent.OpenAI')
    def test_sql_and_forecast_agent_integration(self, mock_sql_openai, mock_db, 
                                               mock_dataset_manager, mock_vector_db):
        """Test SQL Agent and Forecast Agent working together"""
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        
        mock_dataset = Mock()
        mock_dataset.get_active_dataset.return_value = {'table_name': 'test_table'}
        mock_dataset_manager.return_value = mock_dataset
        
        mock_db.execute_query_dict.return_value = [
            {'column_name': 'sales', 'data_type': 'numeric'}
        ]
        
        # Mock SQL generation
        mock_sql_client = Mock()
        mock_sql_response = Mock()
        mock_sql_response.choices = [Mock()]
        mock_sql_response.choices[0].message = Mock()
        mock_sql_response.choices[0].message.content = "SELECT sales FROM test_table ORDER BY date"
        mock_sql_client.chat.completions.create.return_value = mock_sql_response
        mock_sql_openai.return_value = mock_sql_client
        
        # Mock time series data
        time_series_df = pd.DataFrame({
            'sales': [100, 105, 110, 115, 120, 125]
        })
        mock_db.execute_query = Mock(return_value=time_series_df)
        
        sql_agent = SQLAgent()
        forecast_agent = ForecastAgent()
        
        # Generate and execute SQL
        sql_query, _ = sql_agent.generate_sql("Show sales over time")
        data_results = time_series_df
        
        # Generate forecast
        forecast = forecast_agent.generate_forecast(data_results)
        
        assert forecast is not None
        assert isinstance(forecast, dict) or isinstance(forecast, str)
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.sql_agent.DatasetManager')
    @patch('backend.agents.sql_agent.db_manager')
    @patch('backend.agents.sql_agent.OpenAI')
    def test_sql_and_anomaly_agent_integration(self, mock_sql_openai, mock_db,
                                               mock_dataset_manager, mock_vector_db):
        """Test SQL Agent and Anomaly Agent working together"""
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        
        mock_dataset = Mock()
        mock_dataset.get_active_dataset.return_value = {'table_name': 'test_table'}
        mock_dataset_manager.return_value = mock_dataset
        
        mock_db.execute_query_dict.return_value = [
            {'column_name': 'sales', 'data_type': 'numeric'}
        ]
        
        # Mock SQL
        mock_sql_client = Mock()
        mock_sql_response = Mock()
        mock_sql_response.choices = [Mock()]
        mock_sql_response.choices[0].message = Mock()
        mock_sql_response.choices[0].message.content = "SELECT sales FROM test_table"
        mock_sql_client.chat.completions.create.return_value = mock_sql_response
        mock_sql_openai.return_value = mock_sql_client
        
        # Mock data with potential anomalies
        import numpy as np
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 90)
        outliers = np.array([500, 600, -200, 700, 800])
        combined = np.concatenate([normal_data, outliers])
        anomaly_df = pd.DataFrame({'sales': combined})
        mock_db.execute_query = Mock(return_value=anomaly_df)
        
        sql_agent = SQLAgent()
        anomaly_agent = AnomalyAgent()
        
        # Generate and execute SQL
        sql_query, _ = sql_agent.generate_sql("Show all sales")
        data_results = anomaly_df
        
        # Detect anomalies
        anomalies = anomaly_agent.detect_anomalies(data_results)
        
        assert anomalies is not None
        if isinstance(anomalies, dict):
            assert 'anomalies_detected' in anomalies
            assert anomalies['anomalies_detected'] > 0
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.planner.IntentDetector')
    @patch('backend.agents.sql_agent.DatasetManager')
    @patch('backend.agents.sql_agent.db_manager')
    @patch('backend.agents.sql_agent.OpenAI')
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_full_agent_workflow(self, mock_analysis_openai, mock_sql_openai,
                                 mock_db, mock_dataset_manager, mock_intent_detector,
                                 mock_vector_db):
        """Test complete workflow: Planner -> SQL -> Analysis"""
        # Setup mocks
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        
        mock_intent = Mock()
        mock_intent.detect_intent.return_value = {
            "is_data_query": True,
            "primary_intent": "data_analysis",
            "required_agents": ["SQL_AGENT", "INSIGHT_AGENT"],
            "reasoning": "Data query",
            "needs_clarification": False
        }
        mock_intent_detector.return_value = mock_intent
        
        mock_dataset = Mock()
        mock_dataset.get_active_dataset.return_value = {'table_name': 'test_table'}
        mock_dataset_manager.return_value = mock_dataset
        
        mock_db.execute_query_dict.return_value = [
            {'column_name': 'name', 'data_type': 'text'},
            {'column_name': 'quantity', 'data_type': 'integer'}
        ]
        
        # Mock SQL generation
        mock_sql_client = Mock()
        mock_sql_response = Mock()
        mock_sql_response.choices = [Mock()]
        mock_sql_response.choices[0].message = Mock()
        mock_sql_response.choices[0].message.content = "SELECT name, quantity FROM test_table"
        mock_sql_client.chat.completions.create.return_value = mock_sql_response
        mock_sql_openai.return_value = mock_sql_client
        
        mock_df = pd.DataFrame({
            'name': ['Product A', 'Product B'],
            'quantity': [10, 20]
        })
        mock_db.execute_query = Mock(return_value=mock_df)
        
        # Mock Analysis
        mock_analysis_client = Mock()
        mock_analysis_response = Mock()
        mock_analysis_response.choices = [Mock()]
        mock_analysis_response.choices[0].message = Mock()
        mock_analysis_response.choices[0].message.content = "Insights: Product B has higher quantity"
        mock_analysis_client.chat.completions.create.return_value = mock_analysis_response
        mock_analysis_openai.return_value = mock_analysis_client
        
        # Execute workflow
        planner = PlannerAgent()
        plan = planner.create_plan("What are the top products?")
        
        assert plan['is_data_query'] is True
        assert "SQL_AGENT" in plan['required_agents']
        assert "INSIGHT_AGENT" in plan['required_agents']
        
        # Execute agents
        sql_agent = SQLAgent()
        sql_query, _ = sql_agent.generate_sql("What are the top products?")
        data_results = mock_df
        
        analysis_agent = AnalysisAgent()
        insights = analysis_agent.generate_insights("What are the top products?", 
                                                    data_results.to_dict('records'))
        
        assert sql_query is not None
        assert insights is not None
    
    @patch('backend.agents.sql_agent.DatasetManager')
    @patch('backend.agents.sql_agent.db_manager')
    @patch('backend.agents.sql_agent.OpenAI')
    @patch('backend.agents.analysis_agent.OpenAI')
    @patch('backend.agents.forecast_agent.ForecastAgent')
    def test_multi_agent_chain(self, mock_forecast_class, mock_analysis_openai,
                               mock_sql_openai, mock_db, mock_dataset_manager):
        """Test chain of SQL -> Forecast -> Analysis agents"""
        mock_dataset = Mock()
        mock_dataset.get_active_dataset.return_value = {'table_name': 'test_table'}
        mock_dataset_manager.return_value = mock_dataset
        
        mock_db.execute_query_dict.return_value = [
            {'column_name': 'sales', 'data_type': 'numeric'}
        ]
        
        # Mock SQL
        mock_sql_client = Mock()
        mock_sql_response = Mock()
        mock_sql_response.choices = [Mock()]
        mock_sql_response.choices[0].message = Mock()
        mock_sql_response.choices[0].message.content = "SELECT sales FROM test_table"
        mock_sql_client.chat.completions.create.return_value = mock_sql_response
        mock_sql_openai.return_value = mock_sql_client
        
        # Mock data
        time_series_df = pd.DataFrame({
            'sales': [100, 105, 110, 115, 120, 125, 130]
        })
        mock_db.execute_query = Mock(return_value=time_series_df)
        
        # Mock Analysis
        mock_analysis_client = Mock()
        mock_analysis_response = Mock()
        mock_analysis_response.choices = [Mock()]
        mock_analysis_response.choices[0].message = Mock()
        mock_analysis_response.choices[0].message.content = "Sales are increasing"
        mock_analysis_client.chat.completions.create.return_value = mock_analysis_response
        mock_analysis_openai.return_value = mock_analysis_client
        
        # Execute chain
        sql_agent = SQLAgent()
        forecast_agent = ForecastAgent()
        analysis_agent = AnalysisAgent()
        
        sql_query, _ = sql_agent.generate_sql("Show sales")
        data_results = time_series_df
        
        forecast = forecast_agent.generate_forecast(data_results)
        insights = analysis_agent.generate_insights("Analyze sales", 
                                                    data_results.to_dict('records'))
        
        assert sql_query is not None
        assert forecast is not None
        assert insights is not None

