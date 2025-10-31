"""
Unit tests for PlannerAgent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.agents.planner import PlannerAgent


class TestPlannerAgent:
    """Test suite for PlannerAgent"""
    
    def test_planner_initialization(self):
        """Test that PlannerAgent initializes correctly"""
        planner = PlannerAgent()
        assert planner is not None
        assert hasattr(planner, 'intent_detector')
        assert hasattr(planner, 'agent_descriptions')
    
    def test_agent_descriptions_exist(self):
        """Test that all agent descriptions are defined"""
        planner = PlannerAgent()
        expected_agents = [
            "SQL_AGENT", "INSIGHT_AGENT", "FORECAST_AGENT",
            "ANOMALY_AGENT", "VISUALIZATION_AGENT", "VECTOR_AGENT"
        ]
        for agent in expected_agents:
            assert agent in planner.agent_descriptions
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.planner.IntentDetector')
    def test_create_plan_data_query(self, mock_intent_detector_class, mock_vector_db):
        """Test plan creation for data analysis query"""
        # Setup mocks
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        mock_intent_detector = Mock()
        mock_intent_detector.detect_intent.return_value = {
            "is_data_query": True,
            "primary_intent": "data_analysis",
            "required_agents": ["SQL_AGENT", "INSIGHT_AGENT"],
            "reasoning": "Query asks for data analysis",
            "needs_clarification": False,
            "clarification_question": ""
        }
        mock_intent_detector_class.return_value = mock_intent_detector
        
        planner = PlannerAgent()
        plan = planner.create_plan("What are the top products by sales?")
        
        assert plan['is_data_query'] is True
        assert plan['primary_intent'] == "data_analysis"
        assert "SQL_AGENT" in plan['required_agents']
        assert "INSIGHT_AGENT" in plan['required_agents']
        assert len(plan['execution_plan']) > 0
        assert plan['execution_plan'][0]['agent'] == "SQL_AGENT"
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.planner.IntentDetector')
    def test_create_plan_conversational_query(self, mock_intent_detector_class, mock_vector_db):
        """Test plan creation for conversational query"""
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        mock_intent_detector = Mock()
        mock_intent_detector.detect_intent.return_value = {
            "is_data_query": False,
            "primary_intent": "conversational",
            "required_agents": [],
            "reasoning": "Greeting query",
            "needs_clarification": False
        }
        mock_intent_detector_class.return_value = mock_intent_detector
        
        planner = PlannerAgent()
        plan = planner.create_plan("Hello, how are you?")
        
        assert plan['is_data_query'] is False
        assert plan['primary_intent'] == "conversational"
        assert len(plan['required_agents']) == 0
        assert len(plan['execution_plan']) == 0
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.planner.IntentDetector')
    def test_create_plan_semantic_search(self, mock_intent_detector_class, mock_vector_db):
        """Test plan creation for semantic search query"""
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        mock_intent_detector = Mock()
        mock_intent_detector.detect_intent.return_value = {
            "is_data_query": True,
            "primary_intent": "semantic_search",
            "required_agents": ["VECTOR_AGENT"],
            "reasoning": "Product search query",
            "needs_clarification": False
        }
        mock_intent_detector_class.return_value = mock_intent_detector
        
        planner = PlannerAgent()
        plan = planner.create_plan("Find products similar to laptop")
        
        assert plan['is_data_query'] is True
        assert plan['primary_intent'] == "semantic_search"
        assert "VECTOR_AGENT" in plan['required_agents']
        assert len(plan['execution_plan']) == 1
        assert plan['execution_plan'][0]['agent'] == "VECTOR_AGENT"
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.planner.IntentDetector')
    def test_create_plan_with_forecast(self, mock_intent_detector_class, mock_vector_db):
        """Test plan creation for forecast query"""
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        mock_intent_detector = Mock()
        mock_intent_detector.detect_intent.return_value = {
            "is_data_query": True,
            "primary_intent": "forecasting",
            "required_agents": ["SQL_AGENT", "FORECAST_AGENT", "INSIGHT_AGENT"],
            "reasoning": "Forecast request",
            "needs_clarification": False
        }
        mock_intent_detector_class.return_value = mock_intent_detector
        
        planner = PlannerAgent()
        plan = planner.create_plan("Predict sales for next month")
        
        assert "FORECAST_AGENT" in plan['required_agents']
        # Check that SQL_AGENT comes before FORECAST_AGENT
        sql_step = next((s for s in plan['execution_plan'] if s['agent'] == "SQL_AGENT"), None)
        forecast_step = next((s for s in plan['execution_plan'] if s['agent'] == "FORECAST_AGENT"), None)
        assert sql_step is not None
        assert forecast_step is not None
        assert sql_step['step'] < forecast_step['step']
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.planner.IntentDetector')
    def test_create_plan_with_anomaly_detection(self, mock_intent_detector_class, mock_vector_db):
        """Test plan creation for anomaly detection query"""
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        mock_intent_detector = Mock()
        mock_intent_detector.detect_intent.return_value = {
            "is_data_query": True,
            "primary_intent": "anomaly_detection",
            "required_agents": ["SQL_AGENT", "ANOMALY_AGENT", "INSIGHT_AGENT"],
            "reasoning": "Anomaly detection request",
            "needs_clarification": False
        }
        mock_intent_detector_class.return_value = mock_intent_detector
        
        planner = PlannerAgent()
        plan = planner.create_plan("Find unusual patterns in sales")
        
        assert "ANOMALY_AGENT" in plan['required_agents']
        anomaly_step = next((s for s in plan['execution_plan'] if s['agent'] == "ANOMALY_AGENT"), None)
        assert anomaly_step is not None
        assert "SQL_AGENT" in anomaly_step['dependencies']
    
    @patch('backend.agents.planner.vector_db')
    @patch('backend.agents.planner.IntentDetector')
    def test_create_plan_needs_clarification(self, mock_intent_detector_class, mock_vector_db):
        """Test plan creation when clarification is needed"""
        mock_vector_db.get_active_dataset.return_value = 'test_table'
        mock_intent_detector = Mock()
        mock_intent_detector.detect_intent.return_value = {
            "is_data_query": True,
            "primary_intent": "data_analysis",
            "required_agents": [],
            "reasoning": "Unclear query",
            "needs_clarification": True,
            "clarification_question": "Which products are you interested in?"
        }
        mock_intent_detector_class.return_value = mock_intent_detector
        
        planner = PlannerAgent()
        plan = planner.create_plan("Show me data")
        
        assert plan['needs_clarification'] is True
        assert len(plan['clarification_question']) > 0
    
    @patch('backend.agents.planner.vector_db')
    def test_create_plan_no_active_dataset(self, mock_vector_db):
        """Test plan creation when no active dataset exists"""
        mock_vector_db.get_active_dataset.return_value = None
        
        planner = PlannerAgent()
        # Should still create a plan, but with has_active_dataset=False
        plan = planner.create_plan("What are the top products?")
        
        assert plan['has_active_dataset'] is False
    
    def test_generate_execution_plan_vector_only(self):
        """Test execution plan generation for vector-only query"""
        planner = PlannerAgent()
        intent_result = {
            "required_agents": ["VECTOR_AGENT"]
        }
        
        plan = planner._generate_execution_plan(intent_result)
        
        assert len(plan) == 1
        assert plan[0]['agent'] == "VECTOR_AGENT"
        assert plan[0]['step'] == 1
    
    def test_generate_execution_plan_standard_flow(self):
        """Test execution plan generation for standard data query"""
        planner = PlannerAgent()
        intent_result = {
            "required_agents": ["SQL_AGENT", "INSIGHT_AGENT"]
        }
        
        plan = planner._generate_execution_plan(intent_result)
        
        assert len(plan) >= 2
        assert plan[0]['agent'] == "SQL_AGENT"
        assert plan[0]['step'] == 1
        assert any(s['agent'] == "INSIGHT_AGENT" for s in plan)
    
    def test_generate_execution_plan_forecast_auto_adds_sql(self):
        """Test that forecast agent automatically adds SQL_AGENT"""
        planner = PlannerAgent()
        intent_result = {
            "required_agents": ["FORECAST_AGENT"]
        }
        
        plan = planner._generate_execution_plan(intent_result)
        
        sql_agent = next((s for s in plan if s['agent'] == "SQL_AGENT"), None)
        assert sql_agent is not None
        assert sql_agent['step'] == 1

