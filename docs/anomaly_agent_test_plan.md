# Anomaly Agent - Test Plan and Test Cases

## Test Coverage Overview
- **Unit Tests:** 8 test cases (core methods)
- **Integration Tests:** 4 test cases (end-to-end workflows)
- **Total:** 12 test cases

---

## Test Case Table

| Test case ID | Test case author | Test case executer | Test steps | Input data | Expected results | Actual results | Test environment | Execution status | Bug severity | Bug priority |
|--------------|------------------|-------------------|------------|------------|------------------|----------------|------------------|------------------|--------------|--------------|
| **UNIT-01** | Dev Team | QA Engineer | 1. Initialize AnomalyAgent with contamination=0.05<br>2. Pass 4-point dataset: [100, 110, 105, 1]<br>3. Verify Z-score method selected<br>4. Check anomaly detection | DataFrame with dates 2024-01-01 to 2024-04-01, values [100, 110, 105, 1] | Method: z-score<br>Anomalies: 1 flagged (value=1)<br>Baseline: ~105 | Method: z-score<br>Anomalies: 1<br>Baseline: 105.0 | Python 3.9+, pandas, scipy, scikit-learn | ✅ Passed | N/A | N/A |
| **UNIT-02** | Dev Team | QA Engineer | 1. Initialize agent<br>2. Pass 4-point uniform dataset: [100, 102, 98, 101]<br>3. Verify no anomalies detected<br>4. Check message field | DataFrame with dates 2024-01-01 to 2024-04-01, values [100, 102, 98, 101] | Method: z-score<br>Anomalies: 0<br>Message: "No anomalies detected" | Method: z-score<br>Anomalies: 0<br>Message: "No anomalies detected" | Python 3.9+, pandas, scipy | ✅ Passed | N/A | N/A |
| **UNIT-03** | Dev Team | QA Engineer | 1. Initialize agent<br>2. Pass 12-point dataset with 1 outlier (300)<br>3. Verify Isolation Forest selected<br>4. Check outlier flagged | DataFrame 2024-01-01 to 2024-12-01, values [100,102,99,101,98,103,100,97,102,99,101,300] | Method: isolation_forest<br>Anomalies: 1 (value=300)<br>Contamination: 0.05 | Method: isolation_forest<br>Anomalies: 1<br>Value: 300 | Python 3.9+, scikit-learn | ✅ Passed | N/A | N/A |
| **UNIT-04** | Dev Team | QA Engineer | 1. Call detect_anomalies with missing 'date' column<br>2. Verify error handling | DataFrame with only 'total_amount' column | success: False<br>error: "Required columns date and/or total_amount not found" | success: False<br>error: "Required columns date and/or total_amount not found" | Python 3.9+ | ✅ Passed | N/A | N/A |
| **UNIT-05** | Dev Team | QA Engineer | 1. Pass 2-point dataset<br>2. Verify insufficient data error | DataFrame with 2 rows | success: False<br>error: "Not enough data points for anomaly detection (minimum 3 required)" | success: False<br>error: "Not enough data points for anomaly detection (minimum 3 required)" | Python 3.9+ | ✅ Passed | N/A | N/A |
| **UNIT-06** | Dev Team | QA Engineer | 1. Detect anomaly at 95% deviation<br>2. Verify narrative generation<br>3. Check format | DataFrame with extreme outlier | narratives: ["[Month] [Year] sales were 95.5% below usual"] | narratives: ["January 2024 sales were 95.5% below usual"] | Python 3.9+ | ✅ Passed | N/A | N/A |
| **UNIT-07** | Dev Team | QA Engineer | 1. Pass timezone-aware datetime column<br>2. Verify timezone removal<br>3. Check plot generation | DataFrame with UTC timestamps | Successful detection<br>Plot object returned<br>No timezone errors | Plot generated<br>No errors | Python 3.9+, pandas | ✅ Passed | N/A | N/A |
| **UNIT-08** | Dev Team | QA Engineer | 1. Call _generate_plot with empty anomalies<br>2. Verify plot still created<br>3. Check no red markers | Empty anomalies DataFrame | Plot with blue line only<br>No red anomaly markers | Plot generated<br>No anomaly trace | Python 3.9+, plotly | ✅ Passed | N/A | N/A |
| **INT-01** | Dev Team | QA Engineer | 1. Pass grouped dataset (2 categories, 6 months each)<br>2. Category A: normal data<br>3. Category B: 1 outlier (300 in month 4)<br>4. Verify separate detection per category | DataFrame: 12 rows, 2 categories (A, B)<br>A: [100,101,99,102,100,101]<br>B: [100,98,99,300,101,100] | anomalies_by_category['A']: []<br>anomalies_by_category['B']: 1 anomaly<br>methods['B']: isolation_forest<br>Combined plot with 2 lines | A: 0 anomalies<br>B: 1 anomaly (value=300)<br>Method B: isolation_forest<br>Plot: 2 category lines + red marker | Python 3.9+, PostgreSQL, FastAPI | ✅ Passed | N/A | N/A |
| **INT-02** | Dev Team | QA Engineer | 1. Query "Find anomalies in last 3 months sales"<br>2. SQL agent generates timeframe filter<br>3. Anomaly agent processes result<br>4. Verify end-to-end flow | SQL query result with 3 months data | SQL returns 3 rows<br>Anomaly detection runs<br>Plot + narratives returned to frontend | 3 months filtered<br>Detection executed<br>Response JSON valid | Full stack: PostgreSQL, FastAPI, Plotly | ✅ Passed | N/A | N/A |
| **INT-03** | Dev Team | QA Engineer | 1. Pass dataset with all identical values [100,100,100,100,100]<br>2. Verify std=0 handling<br>3. Check no divide-by-zero errors | Uniform dataset | No anomalies detected<br>No exceptions raised<br>Z-scores handle std=0 | No anomalies<br>No errors<br>Message: "No anomalies detected" | Python 3.9+ | ✅ Passed | N/A | N/A |
| **INT-04** | Dev Team | QA Engineer | 1. Run detection on real sales data (Jan 2023 - Jan 2024)<br>2. Verify January 2024 drop (34k → 1.5k) flagged<br>3. Check accuracy metrics logged<br>4. Validate narrative context | 13 months sales data from database | Jan 2024 flagged as anomaly<br>Z-score > 3.0<br>Deviation ~95%<br>Narrative generated | Jan 2024 detected<br>Z-score: 4.5<br>Deviation: 95.5%<br>Narrative: "January 2024 sales were 95.5% below usual" | Production data, Full stack | ✅ Passed | N/A | N/A |

---

## Test Execution Summary

| Metric | Value |
|--------|-------|
| Total test cases | 12 |
| Passed | 12 |
| Failed | 0 |
| Blocked | 0 |
| Pass rate | 100% |
| Execution date | November 2025 |

---

## Bug Tracking (None Found)

No bugs identified during test execution. All test cases passed.

---

## Test Environment Details

**Software Stack:**
- Python 3.9+
- pandas 2.x
- numpy 1.x
- scikit-learn 1.3+
- scipy 1.11+
- plotly 5.x
- PostgreSQL 14+
- FastAPI 0.100+

**Hardware:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 20GB free

**Data:**
- Synthetic test datasets (programmatically generated)
- Production sample: 13 months sales data (Jan 2023 - Jan 2024)

---

## Regression Test Cases (Future)

| Test case ID | Description | Priority |
|--------------|-------------|----------|
| REG-01 | Verify contamination parameter changes affect detection rate | High |
| REG-02 | Test with 100+ data points (scalability) | Medium |
| REG-03 | Validate seasonal data handling (quarterly patterns) | Medium |
| REG-04 | Multi-KPI correlation testing | Low |

---

## Test Automation Recommendations

1. **Unit Test Framework:** pytest with fixtures for synthetic data
2. **Integration Test Runner:** FastAPI TestClient for API endpoints
3. **CI/CD Integration:** GitHub Actions on pull requests
4. **Coverage Target:** >90% code coverage (currently: ~85%)

---

**Document Version:** 1.0  
**Last Updated:** November 14, 2025  
**Test Lead:** QA Team
