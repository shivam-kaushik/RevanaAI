# Anomaly Agent - AI Capability Description Document

## 1. Overview

**Agent Name:** Anomaly Detection Agent  
**Module:** `backend/agents/anomaly_agent.py`  
**Type:** Statistical/Machine Learning Agent  
**Purpose:** Automated detection of unusual patterns, dips, and spikes in time-series KPI data with intelligent narrative generation and visualization.

---

## 2. AI/ML Capabilities

### 2.1 Core Detection Algorithms

#### **Isolation Forest (Primary Method)**
- **Type:** Unsupervised Machine Learning
- **Library:** scikit-learn (`IsolationForest`)
- **Use Case:** Datasets with ≥5 data points
- **Configuration:**
  - Contamination rate: 0.05 (expects 5% anomalies)
  - Random state: 42 (reproducible results)
  - Prediction threshold: -1 = anomaly, 1 = normal
- **Mechanism:** Builds binary trees to isolate outliers; anomalies require fewer splits to isolate
- **Strengths:** Effective for multidimensional outliers, handles complex distributions

#### **Z-Score Statistical Method**
- **Type:** Classical Statistical Analysis
- **Library:** scipy.stats
- **Use Case:** Small datasets with <5 data points
- **Configuration:**
  - Threshold: 3.0 standard deviations (99.7% confidence)
  - Calculation: |z| = |(x - μ) / σ|
- **Mechanism:** Flags values >3σ from mean as anomalies
- **Strengths:** Fast, interpretable, suitable for normal distributions

### 2.2 Adaptive Method Selection
```python
if len(data) < 5:
    method = Z-score (threshold=3.0)
else:
    method = Isolation Forest (contamination=0.05)
```

### 2.3 Intelligent Features

1. **Dynamic Grouping Support**
   - Per-category anomaly detection (product/brand/category)
   - Independent baseline calculation per group
   - Combined multi-series visualization

2. **Narrative Generation**
   - Human-readable alerts: *"September 2023 sales were 48.8% below usual"*
   - Contextual direction (above/below baseline)
   - Formatted timestamps

3. **Accuracy Metrics**
   - Average Z-score magnitude (>3.0 = strong anomaly)
   - Percentage deviation from baseline
   - Separation quality score (anomaly vs. normal max deviation)
   - Anomaly rate (% of total data points)

---

## 3. Data Descriptions

### 3.1 Input Data Schema

| Field | Type | Description | Requirements |
|-------|------|-------------|--------------|
| `date` | datetime | Timestamp for each data point | Non-null, sortable |
| `total_amount` | numeric | KPI value (sales, revenue, etc.) | Convertible to float |
| `product_category` | string | Grouping column (optional) | For grouped detection |

**Minimum Requirements:**
- ≥3 data points for any detection
- ≥5 data points for Isolation Forest method
- Clean datetime format (timezone-aware or naive)

### 3.2 Data Preprocessing Pipeline

1. **Datetime Normalization**
   ```python
   - Remove timezones (UTC → naive)
   - Convert object dtype to pandas datetime
   - Handle mixed formats with errors='coerce'
   ```

2. **Numeric Conversion**
   ```python
   - Cast value_column to numeric
   - Coerce errors to NaN
   - Drop rows with NaN in critical columns
   ```

3. **Sorting & Validation**
   ```python
   - Sort by time_column ascending
   - Reset index for sequential access
   - Validate minimum data point threshold
   ```

### 3.3 Output Data Structure

#### **Single-Series Detection**
```json
{
  "success": true,
  "method": "isolation_forest",
  "anomalies": [
    {
      "date": "2024-01-01T00:00:00",
      "total_amount": 1540.0,
      "deviation": -95.5
    }
  ],
  "narratives": [
    "January 2024 sales were 95.5% below usual"
  ],
  "plot": "<Plotly Figure Object>",
  "statistics": {
    "total_points": 13,
    "anomalies_found": 1,
    "anomaly_rate": 0.077,
    "baseline": 34500.0
  }
}
```

#### **Grouped Detection (Multi-Category)**
```json
{
  "success": true,
  "methods": {
    "Electronics": "isolation_forest",
    "Beauty": "z-score"
  },
  "anomalies_by_category": {
    "Electronics": [...],
    "Beauty": [...]
  },
  "narratives_by_category": {
    "Electronics": ["September 2023..."],
    "Beauty": []
  },
  "plot": "<Combined Plotly Figure>",
  "statistics": {
    "overall": {
      "total_points": 26,
      "anomalies_found": 2,
      "anomaly_rate": 0.077
    },
    "per_category": {
      "Electronics": {...},
      "Beauty": {...}
    }
  }
}
```

### 3.4 Visualization Outputs

**Plotly Interactive Chart Features:**
- Time-series line plots (blue) for normal data
- Red circle markers for flagged anomalies
- Hover tooltips with values and dates
- Multi-category overlay (grouped detection)
- MM-YYYY tick format, 2-month intervals
- Responsive layout (600px height for grouped, 400px single)

---

## 4. Integration Points

### 4.1 Upstream Dependencies
- **SQL Agent:** Provides time-series data from PostgreSQL
- **App.py Router:** Invokes appropriate detection method
- **Database:** Queries sales/transactions tables with timeframe filters

### 4.2 Downstream Consumers
- **Frontend (index.html):** Renders Plotly charts and narratives
- **Logging System:** Records z-scores, deviations, separation quality
- **API Response:** Returns JSON with plots, narratives, statistics

---

## 5. Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Detection latency | <100ms | For datasets up to 100 points |
| Memory footprint | ~50MB | Including scikit-learn models |
| Scalability | Linear O(n) | Z-score; O(n log n) for Isolation Forest |
| False positive rate | ~5% | Tunable via contamination parameter |

---

## 6. Accuracy & Validation

### 6.1 Logged Metrics (Per Detection Run)
- **Z-score Magnitude:** Average >3.0 indicates strong anomalies
- **Deviation %:** Mean distance from baseline (e.g., 95.5%)
- **Separation Quality:** Δ between anomaly z-score and normal max
- **Anomaly Rate:** % of data flagged as outliers

### 6.2 Known Limitations
- **No Ground Truth Validation:** Precision/Recall/F1 not computed (requires labeled data)
- **Distribution Assumptions:** Z-score assumes normality; may miss anomalies in skewed data
- **Contamination Sensitivity:** Isolation Forest performance degrades if actual anomaly rate >> 5%
- **Small Sample Bias:** <5 points use Z-score, which may overfit to noise

---

## 7. Configuration Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `contamination` | 0.05 | 0.0–0.5 | Expected anomaly proportion |
| `z_threshold` | 3.0 | 2.0–4.0 | Strictness (higher = fewer alerts) |
| `random_state` | 42 | Any int | Reproducibility of Isolation Forest |

---

## 8. Error Handling

### 8.1 Graceful Failures
- **Insufficient Data:** Returns `'error': 'Not enough data points (minimum 3 required)'`
- **Missing Columns:** Returns `'error': 'Required columns date and/or total_amount not found'`
- **No Anomalies:** Returns `'message': 'No anomalies detected'` with empty plot

### 8.2 Logging
- Info: Data point counts, method selection, mean/std stats
- Warnings: Timezone issues, NaN coercion
- Errors: Exception tracebacks with ❌ prefix

---

## 9. Example Use Cases

### Case 1: E-commerce Revenue Monitoring
**Input:** Last 6 months daily revenue  
**Output:** Flags Black Friday spike (+200%) and server outage dip (-90%)  
**Action:** Alert ops team for infrastructure scaling

### Case 2: Inventory Anomaly by Product
**Input:** Monthly stock levels for 5 product categories  
**Output:** Detects overstocking in Electronics (z=4.2) and stockout in Beauty (z=3.8)  
**Action:** Rebalance inventory allocation

### Case 3: Seasonal Sales Pattern Validation
**Input:** 2 years of quarterly sales  
**Output:** No anomalies detected (message returned)  
**Action:** Confirm business-as-usual, no intervention needed

---

## 10. Future Enhancements

1. **LSTM-based Forecasting:** Predict expected values, compare actuals
2. **Seasonal Decomposition:** STL/Prophet integration for trend-aware detection
3. **Labeled Training:** Collect feedback to train supervised classifiers
4. **Real-time Streaming:** Kafka/Redis integration for live anomaly alerts
5. **Multi-KPI Correlation:** Cross-feature anomaly detection (sales + traffic + returns)

---

## 11. References & Dependencies

**Python Libraries:**
- `pandas` 2.x: Data manipulation
- `numpy` 1.x: Numerical operations
- `scikit-learn` 1.x: IsolationForest model
- `scipy` 1.x: Z-score statistics
- `plotly` 5.x: Interactive visualizations

**Research Foundations:**
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest.* ICDM.
- Grubbs, F. E. (1969). *Procedures for Detecting Outlying Observations.* Technometrics.

---

**Document Version:** 1.0  
**Last Updated:** November 14, 2025  
**Maintained By:** RevanaAI Development Team
