# Test Suite for Revana Agents

This directory contains comprehensive unit and integration tests for all agents in the Revana application.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and test configuration
├── test_planner_agent.py       # Tests for PlannerAgent
├── test_sql_agent.py           # Tests for SQLAgent
├── test_analysis_agent.py      # Tests for AnalysisAgent
├── test_forecast_agent.py      # Tests for ForecastAgent
├── test_anomaly_agent.py       # Tests for AnomalyAgent
├── test_vector_agent.py        # Tests for VectorAgent
└── test_agent_integration.py   # Integration tests for agents working together
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run tests with coverage:
```bash
pytest --cov=backend --cov-report=html
```

### Run specific test file:
```bash
pytest tests/test_sql_agent.py
```

### Run specific test class:
```bash
pytest tests/test_sql_agent.py::TestSQLAgent
```

### Run specific test:
```bash
pytest tests/test_sql_agent.py::TestSQLAgent::test_generate_sql_basic_query
```

### Run tests with verbose output:
```bash
pytest -v
```

### Run tests and show print statements:
```bash
pytest -s
```

## Test Coverage

### Unit Tests

Each agent has comprehensive unit tests covering:

- **PlannerAgent**: Intent detection, execution plan generation, agent routing
- **SQLAgent**: SQL generation, validation, date formatting, query execution
- **AnalysisAgent**: Insight generation, visualization creation, chart type detection
- **ForecastAgent**: Forecast generation, trend detection, data validation
- **AnomalyAgent**: Anomaly detection, statistics calculation, result formatting
- **VectorAgent**: Semantic search, query classification, result formatting

### Integration Tests

Tests for agents working together:

- SQL Agent + Analysis Agent: Data retrieval and insight generation
- SQL Agent + Forecast Agent: Time series data and forecasting
- SQL Agent + Anomaly Agent: Data retrieval and anomaly detection
- Full workflow: Planner → SQL → Analysis chain
- Multi-agent chains: Complex workflows with multiple agents

## Test Fixtures

Shared fixtures in `conftest.py`:

- `sample_dataframe`: Sample pandas DataFrame for testing
- `sample_data_rows`: Sample data as list of dicts
- `sample_time_series_dataframe`: Time series data for forecasting tests
- `mock_openai_client`: Mocked OpenAI API client
- `mock_dataset_manager`: Mocked DatasetManager
- `mock_db_manager`: Mocked DatabaseManager
- `mock_vector_db`: Mocked VectorDBManager
- `mock_vector_store`: Mocked PostgresVectorStore

## Writing New Tests

### Example Unit Test:
```python
def test_my_agent_feature(self):
    """Test description"""
    # Arrange
    agent = MyAgent()
    
    # Act
    result = agent.do_something()
    
    # Assert
    assert result is not None
    assert isinstance(result, str)
```

### Example Integration Test:
```python
@patch('module.dependency')
def test_agent_workflow(self, mock_dependency):
    """Test agent integration"""
    # Setup mocks
    mock_dependency.return_value = expected_value
    
    # Execute workflow
    agent1 = Agent1()
    agent2 = Agent2()
    
    result1 = agent1.step1()
    result2 = agent2.step2(result1)
    
    # Verify
    assert result2 is not None
```

## Mocking External Dependencies

Tests use `unittest.mock` to mock:
- OpenAI API calls
- Database connections
- File system operations
- External service calls

This ensures:
- Tests run fast (no API calls)
- Tests are reliable (no network dependencies)
- Tests are isolated (no side effects)

## Continuous Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest --cov=backend --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Tests fail with import errors:
- Ensure you're in the project root directory
- Check that `backend/` is in Python path
- Verify all dependencies are installed: `pip install -r requirements.txt`

### OpenAI API errors in tests:
- Tests mock OpenAI, so this shouldn't happen
- Check that mocks are properly configured
- Verify `@patch` decorators are correctly applied

### Database connection errors:
- Tests mock database, so this shouldn't happen
- Check that `mock_db_manager` fixture is used
- Verify database mocks are properly configured

