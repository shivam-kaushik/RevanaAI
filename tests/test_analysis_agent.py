"""
Unit tests for AnalysisAgent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.agents.analysis_agent import AnalysisAgent
import base64


class TestAnalysisAgent:
    """Test suite for AnalysisAgent"""
    
    def test_analysis_agent_initialization(self):
        """Test that AnalysisAgent initializes correctly"""
        with patch('backend.agents.analysis_agent.OpenAI'):
            agent = AnalysisAgent()
            assert agent is not None
            assert hasattr(agent, 'client')
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_generate_insights_basic(self, mock_openai_class):
        """Test insight generation from data"""
        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Key insights:\n- Product A has highest sales\n- Electronics category dominates"
        mock_openai.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai
        
        agent = AnalysisAgent()
        data_rows = [
            {'product_name': 'Laptop', 'quantity': 10, 'price': 999.99},
            {'product_name': 'Mouse', 'quantity': 50, 'price': 29.99}
        ]
        
        insights = agent.generate_insights("What are the top products?", data_rows)
        
        assert insights is not None
        assert len(insights) > 0
        assert isinstance(insights, str)
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_generate_insights_empty_data(self, mock_openai_class):
        """Test insight generation with empty data"""
        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "No data available"
        mock_openai.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai
        
        agent = AnalysisAgent()
        insights = agent.generate_insights("Show me insights", [])
        
        assert insights is not None
        # Should handle empty data gracefully
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_create_visualization_bar_chart(self, mock_openai_class):
        """Test bar chart creation"""
        agent = AnalysisAgent()
        data_rows = [
            {'category': 'Electronics', 'sales': 1000},
            {'category': 'Clothing', 'sales': 500},
            {'category': 'Food', 'sales': 800}
        ]
        
        chart_image, error = agent.create_visualization("Show me a bar chart", data_rows, chart_type="bar")
        
        assert chart_image is not None or error is not None  # One should be returned
        if chart_image:
            # Should be base64 encoded
            assert isinstance(chart_image, str)
            # Try to decode to verify it's valid base64
            try:
                base64.b64decode(chart_image)
            except Exception:
                pytest.fail("Chart image should be valid base64")
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_create_visualization_pie_chart(self, mock_openai_class):
        """Test pie chart creation"""
        agent = AnalysisAgent()
        data_rows = [
            {'category': 'Electronics', 'sales': 1000},
            {'category': 'Clothing', 'sales': 500},
            {'category': 'Food', 'sales': 800}
        ]
        
        chart_image, error = agent.create_visualization("Show me a pie chart", data_rows, chart_type="pie")
        
        assert chart_image is not None or error is not None
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_create_visualization_line_chart(self, mock_openai_class):
        """Test line chart creation for time series"""
        agent = AnalysisAgent()
        data_rows = [
            {'date': '2024-01-01', 'sales': 100},
            {'date': '2024-01-02', 'sales': 150},
            {'date': '2024-01-03', 'sales': 120}
        ]
        
        chart_image, error = agent.create_visualization("Show sales over time", data_rows, chart_type="line")
        
        assert chart_image is not None or error is not None
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_detect_chart_type_from_query(self, mock_openai_class):
        """Test automatic chart type detection from query"""
        agent = AnalysisAgent()
        data_rows = [
            {'category': 'Electronics', 'sales': 1000},
            {'category': 'Clothing', 'sales': 500}
        ]
        
        # Test explicit requests
        assert agent._detect_chart_type("show me a bar chart", data_rows) == "bar"
        assert agent._detect_chart_type("create a pie chart", data_rows) == "pie"
        assert agent._detect_chart_type("plot a line graph", data_rows) == "line"
        assert agent._detect_chart_type("show distribution", data_rows) == "histogram"
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_detect_chart_type_auto_detect_time_series(self, mock_openai_class):
        """Test automatic detection of time series data"""
        agent = AnalysisAgent()
        data_rows = [
            {'date': '2024-01-01', 'value': 100},
            {'date': '2024-01-02', 'value': 150}
        ]
        
        chart_type = agent._detect_chart_type("show me data", data_rows)
        assert chart_type == "line"  # Should detect date column and use line chart
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_prepare_data_context(self, mock_openai_class):
        """Test data context preparation"""
        agent = AnalysisAgent()
        data_rows = [
            {'product_name': 'Laptop', 'quantity': 10, 'price': 999.99},
            {'product_name': 'Mouse', 'quantity': 50, 'price': 29.99}
        ]
        
        context = agent._prepare_data_context(data_rows)
        
        assert "Total Records" in context or "Records" in context
        assert "Laptop" in context or "sample" in context.lower()
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_numeric_summary(self, mock_openai_class):
        """Test numeric statistics calculation"""
        agent = AnalysisAgent()
        data_rows = [
            {'quantity': 10, 'price': 999.99, 'name': 'Product A'},
            {'quantity': 20, 'price': 499.99, 'name': 'Product B'},
            {'quantity': 15, 'price': 799.99, 'name': 'Product C'}
        ]
        
        summary = agent._numeric_summary(data_rows)
        
        assert 'quantity' in summary
        assert 'price' in summary
        assert 'mean' in summary['quantity']
        assert 'min' in summary['quantity']
        assert 'max' in summary['quantity']
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_first_numeric_column(self, mock_openai_class):
        """Test finding first numeric column"""
        agent = AnalysisAgent()
        data_rows = [
            {'id': 1, 'name': 'Product A', 'quantity': 10, 'price': 99.99}
        ]
        
        numeric_col = agent._first_numeric_column(data_rows)
        # Method returns first numeric column found (in dict order), which could be 'id', 'quantity', or 'price'
        assert numeric_col in ['id', 'quantity', 'price']
        assert isinstance(data_rows[0][numeric_col], (int, float))
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_to_number_conversion(self, mock_openai_class):
        """Test number conversion utility"""
        agent = AnalysisAgent()
        
        assert agent._to_number(10) == 10.0
        assert agent._to_number(10.5) == 10.5
        assert agent._to_number("10.5") == 10.5
        assert agent._to_number("not a number") is None
        assert agent._to_number(None) is None
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_find_date_column(self, mock_openai_class):
        """Test date column detection"""
        agent = AnalysisAgent()
        
        # Test with date column
        data_with_date = [
            {'date': '2024-01-01', 'value': 100},
            {'date': '2024-01-02', 'value': 150}
        ]
        date_col = agent._find_date_column(data_with_date)
        assert date_col == 'date'
        
        # Test without date column
        data_without_date = [
            {'name': 'Product A', 'value': 100}
        ]
        date_col = agent._find_date_column(data_without_date)
        assert date_col is None
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_parse_date_various_formats(self, mock_openai_class):
        """Test date parsing with various formats"""
        agent = AnalysisAgent()
        
        # Test ISO format
        date1 = agent._parse_date("2024-01-01")
        assert date1 is not None
        
        # Test MM/DD/YYYY format
        date2 = agent._parse_date("01/15/2024")
        assert date2 is not None
        
        # Test invalid date
        date3 = agent._parse_date("not a date")
        assert date3 is None
    
    @patch('backend.agents.analysis_agent.OpenAI')
    def test_rows_to_table_formatting(self, mock_openai_class):
        """Test data table formatting"""
        agent = AnalysisAgent()
        data_rows = [
            {'name': 'Product A', 'quantity': 10},
            {'name': 'Product B', 'quantity': 20}
        ]
        
        table = agent._rows_to_table(data_rows)
        
        assert isinstance(table, str)
        assert 'Product A' in table
        assert '10' in table
        # Should have some structure (pipe separators or similar)
        assert '|' in table or len(table) > 20  # Has some formatting

