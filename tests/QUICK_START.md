# Quick Start Guide - Running Tests

## Step-by-Step Instructions

### 1. Install Test Dependencies

First, make sure you have the testing packages installed:

```bash
# If you haven't installed dependencies yet
pip install -r requirements.txt

# Or install just the test dependencies
pip install pytest pytest-cov pytest-mock
```

### 2. Verify Installation

Check that pytest is installed correctly:

```bash
pytest --version
# Should show: pytest 8.2.2 (or similar)
```

### 3. Run All Tests

From the project root directory, run:

```bash
pytest
```

This will:
- Discover all test files in the `tests/` directory
- Run all tests
- Show a summary at the end

**Expected output:**
```
========================= test session starts ==========================
tests/test_planner_agent.py::TestPlannerAgent::test_planner_initialization PASSED
tests/test_planner_agent.py::TestPlannerAgent::test_agent_descriptions_exist PASSED
...
========================= X passed in Y.YYs ==========================
```

### 4. Run Tests with More Details

For verbose output showing each test:

```bash
pytest -v
# or
pytest --verbose
```

### 5. Run Tests and See Print Statements

If your tests use `print()` statements for debugging:

```bash
pytest -s
# or
pytest --capture=no
```

### 6. Run a Specific Test File

Test only one agent:

```bash
# Test Planner Agent only
pytest tests/test_planner_agent.py

# Test SQL Agent only
pytest tests/test_sql_agent.py

# Test Analysis Agent only
pytest tests/test_analysis_agent.py
```

### 7. Run a Specific Test

Run a single test function:

```bash
# Run specific test
pytest tests/test_sql_agent.py::TestSQLAgent::test_generate_sql_basic_query

# Run all tests in a specific class
pytest tests/test_sql_agent.py::TestSQLAgent
```

### 8. Run Tests with Coverage Report

Generate a code coverage report:

```bash
# Terminal coverage report
pytest --cov=backend --cov-report=term

# HTML coverage report (opens in browser)
pytest --cov=backend --cov-report=html
# Then open: htmlcov/index.html
```

### 9. Run Only Integration Tests

Test agent interactions:

```bash
pytest tests/test_agent_integration.py
```

### 10. Run Tests Matching a Pattern

Run tests with names matching a pattern:

```bash
# All tests with "forecast" in the name
pytest -k forecast

# All tests with "integration" in the name
pytest -k integration

# All SQL-related tests
pytest -k sql
```

## Common Test Commands

### Show Test Execution Times

```bash
pytest --durations=10
# Shows the 10 slowest tests
```

### Stop on First Failure

```bash
pytest -x
# or
pytest --exitfirst
```

### Run Tests in Parallel (faster)

```bash
pip install pytest-xdist
pytest -n auto
# Runs tests using all available CPU cores
```

### Show Local Variables on Failure

```bash
pytest -l
# or
pytest --showlocals
```

## Expected Test Results

When all tests pass, you should see:

```
========================= test session starts ==========================
platform win32 -- Python 3.11.x, pytest-8.2.2
collected 84 items

tests/test_planner_agent.py ................ [ 19%]
tests/test_sql_agent.py ...........        [ 33%]
tests/test_analysis_agent.py ................ [ 52%]
tests/test_forecast_agent.py ............   [ 67%]
tests/test_anomaly_agent.py ............   [ 81%]
tests/test_vector_agent.py .............    [ 95%]
tests/test_agent_integration.py .....       [100%]

========================= 84 passed in 15.23s ==========================
```

## Troubleshooting

### Problem: Import Errors

**Error:** `ModuleNotFoundError: No module named 'backend'`

**Solution:**
```bash
# Make sure you're in the project root directory
cd /path/to/Revana

# Add the project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# On Windows:
set PYTHONPATH=%PYTHONPATH%;%CD%

# Then run tests again
pytest
```

### Problem: Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'pytest'`

**Solution:**
```bash
pip install -r requirements.txt
# or
pip install pytest pytest-cov pytest-mock
```

### Problem: Database Connection Errors

**Note:** Tests use mocks, so this shouldn't happen. If you see database errors:

1. Check that mocks are properly configured
2. Verify `@patch` decorators are applied correctly
3. Make sure you're using the test fixtures from `conftest.py`

### Problem: OpenAI API Errors

**Note:** Tests mock OpenAI, so no API calls should be made. If you see API errors:

- The mock might not be working correctly
- Check that `mock_openai_client` fixture is being used
- Verify `@patch('openai.OpenAI')` is applied

### Problem: Tests Hang or Take Too Long

**Solution:**
```bash
# Add timeout to tests
pytest --timeout=30  # Requires pytest-timeout plugin
```

## Running Tests in Different Environments

### Windows PowerShell

```powershell
# Navigate to project
cd C:\Users\Public\Documents\My_Projects\Revana

# Activate virtual environment (if using one)
.\venv\Scripts\Activate.ps1

# Run tests
pytest
```

### Windows CMD

```cmd
cd C:\Users\Public\Documents\My_Projects\Revana
venv\Scripts\activate
pytest
```

### Linux/Mac

```bash
cd /path/to/Revana
source venv/bin/activate  # if using virtual environment
pytest
```

## Continuous Integration

If you're setting up CI/CD, here's a sample GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=backend --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest -v` | Verbose output |
| `pytest -s` | Show print statements |
| `pytest -k keyword` | Run tests matching keyword |
| `pytest tests/test_file.py` | Run specific file |
| `pytest --cov=backend` | With coverage |
| `pytest -x` | Stop on first failure |
| `pytest --durations=10` | Show slowest tests |

## Next Steps

After running tests:

1. Check coverage report to see which areas need more tests
2. Fix any failing tests
3. Add more tests for edge cases you discover
4. Run tests before committing code changes

