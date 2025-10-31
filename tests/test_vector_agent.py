"""
Unit tests for VectorAgent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.agents.vector_agent import VectorAgent


class TestVectorAgent:
    """Test suite for VectorAgent"""
    
    def test_vector_agent_initialization(self):
        """Test that VectorAgent initializes correctly"""
        with patch('backend.agents.vector_agent.PostgresVectorStore'):
            with patch('backend.agents.vector_agent.OpenAI'):
                agent = VectorAgent()
                assert agent is not None
                assert hasattr(agent, 'client')
                assert hasattr(agent, 'vector_store')
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_handle_semantic_query_product_search(self, mock_openai_class, mock_store_class):
        """Test semantic query handling for product search"""
        # Setup mocks
        mock_store = Mock()
        mock_store.is_available.return_value = True
        mock_store.semantic_search_products.return_value = [
            {
                'product_name': 'Laptop Pro',
                'product_category': 'Electronics',
                'similarity': 0.95,
                'description': 'High-end laptop',
                'metadata': {'total_revenue': 10000.0}
            }
        ]
        mock_store_class.return_value = mock_store
        
        mock_openai = Mock()
        
        # Mock classification
        mock_classify_response = Mock()
        mock_classify_response.choices = [Mock()]
        mock_classify_response.choices[0].message = Mock()
        mock_classify_response.choices[0].message.content = "product_search"
        
        mock_openai.chat.completions.create.return_value = mock_classify_response
        mock_openai_class.return_value = mock_openai
        
        agent = VectorAgent()
        result = agent.handle_semantic_query("Find products similar to laptop")
        
        assert result is not None
        assert isinstance(result, str)
        assert 'Laptop Pro' in result or 'product' in result.lower()
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_handle_semantic_query_customer_search(self, mock_openai_class, mock_store_class):
        """Test semantic query handling for customer search"""
        mock_store = Mock()
        mock_store.is_available.return_value = True
        mock_store.semantic_search_customers.return_value = [
            {
                'customer_id': 'C001',
                'similarity': 0.88,
                'preferences': 'Electronics, Gadgets',
                'metadata': {'total_spent': 5000.0}
            }
        ]
        mock_store_class.return_value = mock_store
        
        mock_openai = Mock()
        mock_classify_response = Mock()
        mock_classify_response.choices = [Mock()]
        mock_classify_response.choices[0].message = Mock()
        mock_classify_response.choices[0].message.content = "customer_search"
        mock_openai.chat.completions.create.return_value = mock_classify_response
        mock_openai_class.return_value = mock_openai
        
        agent = VectorAgent()
        result = agent.handle_semantic_query("Find customers similar to John")
        
        assert result is not None
        assert isinstance(result, str)
        assert 'customer' in result.lower() or 'C001' in result
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_handle_semantic_query_vector_store_unavailable(self, mock_openai_class, mock_store_class):
        """Test handling when vector store is unavailable"""
        mock_store = Mock()
        mock_store.is_available.return_value = False
        mock_store_class.return_value = mock_store
        
        agent = VectorAgent()
        result = agent.handle_semantic_query("Find products")
        
        assert "unavailable" in result.lower() or "not available" in result.lower()
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_classify_query_type(self, mock_openai_class, mock_store_class):
        """Test query type classification"""
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "product_search"
        mock_openai.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai
        
        agent = VectorAgent()
        query_type = agent.classify_query_type("Find products")
        
        assert query_type == "product_search"
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_extract_category_filter(self, mock_openai_class, mock_store_class):
        """Test category filter extraction"""
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Electronics"
        mock_openai.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai
        
        agent = VectorAgent()
        category = agent.extract_category_filter("Find electronics products")
        
        assert category == "Electronics"
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_extract_category_filter_none(self, mock_openai_class, mock_store_class):
        """Test category filter extraction when no category found"""
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "None"
        mock_openai.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai
        
        agent = VectorAgent()
        category = agent.extract_category_filter("Find products")
        
        assert category is None
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_extract_city_filter(self, mock_openai_class, mock_store_class):
        """Test city filter extraction"""
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "New York"
        mock_openai.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai
        
        agent = VectorAgent()
        city = agent.extract_city_filter("Find products in New York")
        
        assert city == "New York"
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_format_product_results(self, mock_openai_class, mock_store_class):
        """Test product results formatting"""
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        agent = VectorAgent()
        results = [
            {
                'product_name': 'Laptop Pro',
                'product_category': 'Electronics',
                'similarity': 0.95,
                'description': 'High-end laptop',
                'metadata': {'total_revenue': 10000.0, 'purchase_count': 50}
            }
        ]
        
        formatted = agent.format_product_results(results, "Find laptop")
        
        assert isinstance(formatted, str)
        assert 'Laptop Pro' in formatted
        assert 'Electronics' in formatted or '95' in formatted
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_format_product_results_empty(self, mock_openai_class, mock_store_class):
        """Test product results formatting with empty results"""
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        agent = VectorAgent()
        formatted = agent.format_product_results([], "Find laptop")
        
        assert isinstance(formatted, str)
        assert "no" in formatted.lower() or "found" in formatted.lower()
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_format_customer_results(self, mock_openai_class, mock_store_class):
        """Test customer results formatting"""
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        agent = VectorAgent()
        results = [
            {
                'customer_id': 'C001',
                'similarity': 0.88,
                'preferences': 'Electronics, Gadgets',
                'metadata': {'total_spent': 5000.0, 'transaction_count': 25}
            }
        ]
        
        formatted = agent.format_customer_results(results, "Find customers")
        
        assert isinstance(formatted, str)
        assert 'C001' in formatted or 'customer' in formatted.lower()
    
    @patch('backend.agents.vector_agent.PostgresVectorStore')
    @patch('backend.agents.vector_agent.OpenAI')
    def test_handle_semantic_query_error_handling(self, mock_openai_class, mock_store_class):
        """Test error handling in semantic query"""
        mock_store = Mock()
        mock_store.is_available.return_value = True
        mock_store.semantic_search_products.side_effect = Exception("Database error")
        mock_store_class.return_value = mock_store
        
        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "product_search"
        mock_openai.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai
        
        agent = VectorAgent()
        result = agent.handle_semantic_query("Find products")
        
        # Should handle error gracefully
        assert isinstance(result, str)
        assert "error" in result.lower() or "sorry" in result.lower()

