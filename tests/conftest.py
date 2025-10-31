"""
Pytest configuration and shared fixtures for all tests
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    return pd.DataFrame({
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
        'quantity': [10, 50, 30, 15],
        'price': [999.99, 29.99, 79.99, 299.99],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics']
    })


@pytest.fixture
def sample_data_rows():
    """Sample data rows as list of dicts (for agents expecting dict format)"""
    return [
        {'product_name': 'Laptop', 'quantity': 10, 'price': 999.99, 'category': 'Electronics'},
        {'product_name': 'Mouse', 'quantity': 50, 'price': 29.99, 'category': 'Electronics'},
        {'product_name': 'Keyboard', 'quantity': 30, 'price': 79.99, 'category': 'Electronics'},
        {'product_name': 'Monitor', 'quantity': 15, 'price': 299.99, 'category': 'Electronics'}
    ]


@pytest.fixture
def sample_time_series_dataframe():
    """Sample time series DataFrame for forecasting/anomaly tests"""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'date': dates,
        'sales': [100 + i * 2 + (i % 5) * 10 for i in range(30)],
        'quantity': [50 + i for i in range(30)]
    })


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Mocked response"
    return mock_response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client"""
    with patch('openai.OpenAI') as mock_client_class:
        mock_client = Mock()
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = Mock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_dataset_manager():
    """Mock DatasetManager"""
    with patch('backend.utils.dataset_manager.DatasetManager') as mock_manager:
        mock_instance = Mock()
        mock_instance.get_active_dataset = Mock(return_value={
            'table_name': 'revana_test_table',
            'original_filename': 'test.csv',
            'row_count': 100,
            'column_count': 5
        })
        mock_instance.has_active_dataset = Mock(return_value=True)
        mock_manager.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_db_manager():
    """Mock DatabaseManager"""
    with patch('backend.utils.database.db_manager') as mock_db:
        mock_db.execute_query = Mock(return_value=pd.DataFrame({
            'column_name': ['id', 'name', 'quantity'],
            'data_type': ['integer', 'text', 'integer']
        }))
        mock_db.execute_query_dict = Mock(return_value=[
            {'column_name': 'id', 'data_type': 'integer'},
            {'column_name': 'name', 'data_type': 'text'},
            {'column_name': 'quantity', 'data_type': 'integer'}
        ])
        mock_db.get_active_tables = Mock(return_value=['revana_test_table'])
        yield mock_db


@pytest.fixture
def mock_vector_db():
    """Mock VectorDBManager"""
    with patch('backend.utils.vector_db.vector_db') as mock_vdb:
        mock_vdb.get_active_dataset = Mock(return_value='revana_test_table')
        mock_vdb.set_active_dataset = Mock()
        mock_vdb.get_schema_context = Mock(return_value="Active Dataset: revana_test_table\nSchema: id, name, quantity")
        yield mock_vdb


@pytest.fixture
def mock_vector_store():
    """Mock PostgresVectorStore"""
    with patch('backend.utils.vector_store.PostgresVectorStore') as mock_store_class:
        mock_store = Mock()
        mock_store.is_available = Mock(return_value=True)
        mock_store.semantic_search_products = Mock(return_value=[
            {
                'product_name': 'Laptop Pro',
                'product_category': 'Electronics',
                'similarity': 0.95,
                'description': 'High-end laptop',
                'metadata': {'total_revenue': 10000.0, 'purchase_count': 50}
            }
        ])
        mock_store.semantic_search_customers = Mock(return_value=[
            {
                'customer_id': 'C001',
                'similarity': 0.88,
                'preferences': 'Electronics, Gadgets',
                'metadata': {'total_spent': 5000.0, 'transaction_count': 25}
            }
        ])
        mock_store.get_vector_stats = Mock(return_value={'total_vectors': 1000, 'product_vectors': 500, 'customer_vectors': 500})
        mock_store_class.return_value = mock_store
        yield mock_store

