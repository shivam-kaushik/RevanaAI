# Revana Application Workflow - Agent Interaction Analysis

## Overview
Revana is an **Agentic Sales Assistant** that uses a multi-agent architecture to handle data analysis queries. The application intelligently routes user queries to specialized agents based on intent detection.

---

## 🏗️ Architecture Overview

### Core Components
1. **FastAPI Backend** (`backend/app.py`) - Main application server
2. **Planner Agent** - Orchestrates query execution
3. **Intent Detector** - Classifies user queries
4. **Specialized Agents** - Handle specific tasks
5. **Dataset Manager** - Manages active datasets
6. **Database Layer** - PostgreSQL for data storage
7. **Vector Store** - Semantic search capabilities

---

## 🔄 Complete Workflow: How Application Interacts with Agents

### Phase 1: Application Initialization

```
1. Server Startup (run.py → backend/app.py)
   ├── Initialize FastAPI app
   ├── Initialize all agents (singleton instances)
   │   ├── PlannerAgent()
   │   ├── SQLAgent()
   │   ├── AnalysisAgent()
   │   ├── ForecastAgent()
   │   ├── AnomalyAgent()
   │   └── VectorAgent()
   ├── Initialize utilities
   │   ├── FileProcessor()
   │   ├── DatasetManager()
   │   └── VectorDBManager()
   └── Startup event:
       ├── Test database connection
       ├── Ensure datasets table exists
       ├── Auto-register unregistered tables
       └── Set active dataset (latest if available)
```

**Key Files:**
- `run.py` (lines 15-94): Main entry point, starts uvicorn server
- `backend/app.py` (lines 694-785): Startup event handler

---

### Phase 2: Data Upload Workflow

When a user uploads a CSV file:

```
POST /upload
   │
   ├── FileProcessor.process_uploaded_file()
   │   ├── Save CSV to temp file
   │   ├── Read CSV into pandas DataFrame
   │   ├── Generate table name (revana_{filename}_{timestamp})
   │   ├── Create PostgreSQL table from DataFrame
   │   └── Insert data using execute_values (bulk insert)
   │
   ├── VectorDataProcessor.process_uploaded_data()
   │   └── Generate embeddings for products/customers
   │       └── Store in vector database (pgvector)
   │
   └── DatasetManager.register_dataset()
       ├── Insert/Update record in revana_datasets table
       ├── Set as active dataset (is_active = TRUE)
       └── Sync VectorDBManager with new active dataset
```

**Key Files:**
- `backend/app.py` (lines 168-219): Upload endpoint
- `backend/utils/file_processor.py`: CSV processing
- `backend/utils/dataset_manager.py`: Dataset registration

---

### Phase 3: Query Processing Workflow

#### Step 1: User Sends Query → Main Chat Endpoint

```
POST /chat
   │
   └── ChatRequest {
       message: str,
       conversation_id: str
   }
```

#### Step 2: Query Planning Phase

```
PlannerAgent.create_plan(user_query)
   │
   ├── IntentDetector.detect_intent(user_query, has_active_dataset)
   │   ├── Uses GPT-3.5-turbo to classify query
   │   ├── Returns JSON:
   │   │   {
   │   │     "is_data_query": bool,
   │   │     "primary_intent": str,
   │   │     "required_agents": [list],
   │   │     "reasoning": str,
   │   │     "needs_clarification": bool
   │   │   }
   │   └── Fallback: Keyword-based detection if LLM fails
   │
   └── PlannerAgent._generate_execution_plan(intent_result)
       └── Creates ordered execution steps:
           [
             {step: 1, agent: "SQL_AGENT", dependencies: []},
             {step: 2, agent: "INSIGHT_AGENT", dependencies: ["SQL_AGENT"]},
             ...
           ]
```

**Intent Classification Logic:**
- **Data Query**: Questions about data, numbers, trends, patterns
- **Semantic Search**: "find products", "similar customers", "best items"
- **Forecasting**: "predict", "forecast", "future trends"
- **Anomaly Detection**: "anomaly", "outlier", "unusual patterns"
- **Visualization**: "chart", "graph", "plot", "visualize"
- **Conversational**: Greetings, general questions

**Key Files:**
- `backend/app.py` (lines 446-541): Main chat endpoint
- `backend/agents/planner.py`: Planning logic
- `backend/utils/intent_detector.py`: Intent detection

---

### Phase 4: Agent Execution Workflow

The execution plan determines which agents run and in what order:

#### Execution Path A: Data Analysis Query (Standard Flow)

```
execute_agent_plan(plan, has_database_tables)
   │
   ├── STEP 1: SQL_AGENT
   │   ├── SQLAgent.generate_sql(user_query)
   │   │   ├── Get active dataset from DatasetManager
   │   │   ├── Retrieve table schema from PostgreSQL
   │   │   ├── Get schema context from VectorDBManager
   │   │   ├── Use GPT-3.5-turbo to generate SQL
   │   │   │   └── Prompt includes:
   │   │   │       - Table schema
   │   │   │       - Sample data
   │   │   │       - Date format rules (MM/DD/YYYY HH24:MI)
   │   │   ├── Post-process SQL:
   │   │   │   ├── Fix date format issues
   │   │   │   ├── Apply SQL patches (case-insensitive, COALESCE)
   │   │   │   └── Validate SQL (only SELECT, no dangerous operations)
   │   │   └── Return SQL query
   │   │
   │   └── Execute SQL via db_manager.execute_query()
   │       └── Returns: pandas DataFrame with results
   │
   ├── STEP 2: INSIGHT_AGENT (if data_results available)
   │   ├── AnalysisAgent.generate_insights(user_query, data_rows)
   │   │   ├── Prepare data context:
   │   │   │   ├── Format data as table
   │   │   │   ├── Calculate numeric statistics
   │   │   │   └── Include sample rows
   │   │   ├── Use GPT-3.5-turbo to generate insights
   │   │   │   └── Prompt includes:
   │   │   │       - User query
   │   │   │       - Data summary
   │   │   │       - Sample data
   │   │   │       - Numeric statistics
   │   │   └── Return: Natural language insights
   │   │
   │   └── (Optional) AnalysisAgent.create_visualization()
   │       ├── Auto-detect chart type (bar/pie/line/histogram)
   │       ├── Create matplotlib visualization
   │       └── Return: Base64-encoded image
   │
   ├── STEP 3: FORECAST_AGENT (if in plan)
   │   ├── ForecastAgent.generate_forecast(data_results)
   │   │   ├── Extract numeric columns from DataFrame
   │   │   ├── Apply moving average forecast
   │   │   └── Return: Forecast results with trends
   │   │
   └── STEP 4: ANOMALY_AGENT (if in plan)
       ├── AnomalyAgent.detect_anomalies(data_results)
       │   ├── Select numeric columns
       │   ├── Use Isolation Forest (scikit-learn)
       │   ├── Detect outliers (contamination=0.1)
       │   └── Return: Anomaly statistics and message
```

#### Execution Path B: Pure Vector Search Query

```
Special Case: VECTOR_AGENT only (semantic search)
   │
   └── VectorAgent.handle_semantic_query(user_query)
       ├── Classify query type using GPT:
       │   ├── "product_search"
       │   ├── "customer_search"
       │   └── "hybrid_search"
       │
       ├── Extract filters (if any):
       │   ├── Category filter (e.g., "Electronics")
       │   └── City filter (e.g., "New York")
       │
       ├── Perform semantic search:
       │   ├── product_search → vector_store.semantic_search_products()
       │   ├── customer_search → vector_store.semantic_search_customers()
       │   └── hybrid_search → vector_store.hybrid_search()
       │
       └── Format results:
           ├── Product results: name, category, similarity, metadata
           └── Customer results: ID, preferences, purchase history
```

#### Execution Path C: Conversational Query

```
If NOT a data query:
   │
   └── get_chatgpt_response(user_query, has_dataset)
       ├── Use OpenAI GPT-3.5-turbo directly
       ├── System prompt: "Helpful AI assistant for data analysis platform"
       └── Return: Conversational response
```

**Key Files:**
- `backend/app.py` (lines 543-611): Agent execution logic
- `backend/agents/sql_agent.py`: SQL generation
- `backend/agents/analysis_agent.py`: Insights & visualization
- `backend/agents/forecast_agent.py`: Forecasting
- `backend/agents/anomaly_agent.py`: Anomaly detection
- `backend/agents/vector_agent.py`: Semantic search

---

### Phase 5: Response Assembly

```
build_final_response(insights, forecasts, anomalies, data_results, user_query)
   │
   ├── Combine all agent outputs:
   │   ├── Insights section (if available)
   │   ├── Forecasts section (if available)
   │   ├── Anomaly detection section (if available)
   │   └── Data summary with sample rows
   │
   └── Return: Formatted markdown response

ChatResponse {
   response: str,                    # Combined text response
   data: dict,                        # Raw data (SQL results, forecasts, anomalies)
   agents_used: list,                 # List of agents that executed
   execution_plan: list,              # Original execution plan
   needs_clarification: bool,
   has_dataset: bool
}
```

**Key Files:**
- `backend/app.py` (lines 652-683): Response building

---

## 🤖 Agent Details

### 1. Planner Agent
**Purpose:** Orchestrates query execution  
**Location:** `backend/agents/planner.py`

**Responsibilities:**
- Detects user intent via IntentDetector
- Creates execution plan with ordered agent steps
- Determines dependencies between agents
- Routes queries appropriately

**Dependencies:**
- IntentDetector
- VectorDBManager (for active dataset check)

---

### 2. SQL Agent
**Purpose:** Natural language to SQL conversion  
**Location:** `backend/agents/sql_agent.py`

**Responsibilities:**
- Generates SQL queries from natural language
- Handles complex date formatting (MM/DD/YYYY HH24:MI)
- Applies SQL patches for robustness
- Validates SQL (security: only SELECT queries)
- Executes queries and returns DataFrames

**Dependencies:**
- OpenAI GPT-3.5-turbo
- DatasetManager (for active dataset)
- DatabaseManager (for schema and execution)

**Key Features:**
- Automatic date parsing and conversion
- Case-insensitive string matching
- COALESCE for NULL handling
- Column name normalization

---

### 3. Analysis Agent (Insight Agent)
**Purpose:** Generate insights and visualizations  
**Location:** `backend/agents/analysis_agent.py`

**Responsibilities:**
- Generates natural language insights from data
- Creates visualizations (bar, pie, line, histogram)
- Auto-detects appropriate chart types
- Calculates statistical summaries

**Dependencies:**
- OpenAI GPT-3.5-turbo (for insights)
- Matplotlib (for visualizations)

**Visualization Types:**
- **Bar Chart**: Categorical vs numerical data
- **Pie Chart**: Distribution (≤8 categories)
- **Line Chart**: Time series data
- **Histogram**: Numerical distribution

---

### 4. Forecast Agent
**Purpose:** Generate predictions  
**Location:** `backend/agents/forecast_agent.py`

**Responsibilities:**
- Simple moving average forecasts
- Trend detection (increasing/decreasing)
- Multi-column forecasting (up to 3 columns)

**Algorithm:**
- Rolling window moving average
- Current value vs forecast comparison
- Trend direction calculation

---

### 5. Anomaly Agent
**Purpose:** Detect unusual patterns  
**Location:** `backend/agents/anomaly_agent.py`

**Responsibilities:**
- Isolation Forest anomaly detection
- Anomaly percentage calculation
- Statistical reporting

**Algorithm:**
- Isolation Forest (scikit-learn)
- Contamination rate: 10%
- Handles missing values with mean imputation

---

### 6. Vector Agent
**Purpose:** Semantic search for products/customers  
**Location:** `backend/agents/vector_agent.py`

**Responsibilities:**
- Semantic product search (using embeddings)
- Semantic customer search
- Hybrid search with filters
- Query classification (product vs customer)
- Filter extraction (category, city)

**Dependencies:**
- OpenAI (for embeddings and query classification)
- PostgresVectorStore (pgvector)

**Search Types:**
- Product search: "find similar products", "best electronics"
- Customer search: "similar customers", "who buys X"
- Hybrid: Combination with category/location filters

---

## 🔧 Supporting Utilities

### DatasetManager (Singleton)
**Purpose:** Manages active datasets  
**Location:** `backend/utils/dataset_manager.py`

**Key Functions:**
- `register_dataset()`: Register new dataset
- `set_active_dataset()`: Switch active dataset
- `get_active_dataset()`: Get current active dataset (with caching)
- `has_active_dataset()`: Check if dataset exists

**Data Structure:**
- Stores metadata in `revana_datasets` PostgreSQL table
- Tracks: table_name, filename, row_count, column_count, is_active

---

### VectorDBManager
**Purpose:** Manages vector database state  
**Location:** `backend/utils/vector_db.py`

**Key Functions:**
- `set_active_dataset()`: Set active dataset for vector operations
- `get_schema_context()`: Get schema information for SQL generation
- Schema caching for performance

---

### IntentDetector
**Purpose:** Classify user queries  
**Location:** `backend/utils/intent_detector.py`

**Methods:**
- `detect_intent()`: Main classification using GPT-3.5-turbo
- `_fallback_intent_detection()`: Keyword-based fallback

**Classification Output:**
```json
{
  "is_data_query": true/false,
  "primary_intent": "data_analysis|forecasting|anomaly_detection|visualization|semantic_search|conversational",
  "required_agents": ["SQL_AGENT", "INSIGHT_AGENT", ...],
  "reasoning": "explanation",
  "needs_clarification": false,
  "clarification_question": ""
}
```

---

## 📊 Data Flow Diagram

```
User Query
    │
    ▼
[FastAPI /chat endpoint]
    │
    ▼
[PlannerAgent.create_plan()]
    │
    ├──► [IntentDetector.detect_intent()]
    │       └──► GPT-3.5-turbo classification
    │
    └──► Generate execution plan
            │
            ├──► Data Query?
            │       │
            │       ├──► [SQL_AGENT]
            │       │       └──► Generate & Execute SQL
            │       │
            │       ├──► [INSIGHT_AGENT]
            │       │       └──► Generate insights + visualization
            │       │
            │       ├──► [FORECAST_AGENT] (if needed)
            │       │       └──► Moving average forecast
            │       │
            │       └──► [ANOMALY_AGENT] (if needed)
            │               └──► Isolation Forest detection
            │
            ├──► Semantic Search?
            │       │
            │       └──► [VECTOR_AGENT]
            │               └──► Semantic search via pgvector
            │
            └──► Conversational?
                    │
                    └──► [ChatGPT Direct]
                            └──► Direct GPT-3.5-turbo response
    │
    ▼
[build_final_response()]
    │
    ▼
ChatResponse (JSON)
    │
    ▼
Frontend Display
```

---

## 🔄 Agent Execution Order (Typical Scenarios)

### Scenario 1: "What are the top 10 products by sales?"
```
1. PlannerAgent → Intent: data_analysis
2. Execution Plan:
   - Step 1: SQL_AGENT → Generate SQL, Execute query
   - Step 2: INSIGHT_AGENT → Generate insights from results
3. Response: Insights + data summary
```

### Scenario 2: "Show me a chart of sales over time"
```
1. PlannerAgent → Intent: visualization + data_analysis
2. Execution Plan:
   - Step 1: SQL_AGENT → Generate SQL, Execute query
   - Step 2: INSIGHT_AGENT → Generate insights + create line chart
3. Response: Insights + base64 chart image
```

### Scenario 3: "Predict sales for next 6 months"
```
1. PlannerAgent → Intent: forecasting + data_analysis
2. Execution Plan:
   - Step 1: SQL_AGENT → Generate SQL, Execute query
   - Step 2: FORECAST_AGENT → Generate forecasts
   - Step 3: INSIGHT_AGENT → Generate insights
3. Response: Forecasts + insights
```

### Scenario 4: "Find products similar to laptop"
```
1. PlannerAgent → Intent: semantic_search
2. Execution Plan:
   - Step 1: VECTOR_AGENT → Semantic product search
3. Response: Product results with similarity scores
```

### Scenario 5: "Hello, how are you?"
```
1. PlannerAgent → Intent: conversational
2. Execution Plan: [] (empty)
3. Response: Direct ChatGPT response
```

---

## 🔐 Security & Validation

### SQL Agent Security
- **Read-only queries**: Only SELECT statements allowed
- **Dangerous keywords blocked**: DROP, DELETE, UPDATE, INSERT, ALTER, CREATE
- **Input validation**: SQL syntax validation before execution

### Data Validation
- **Dataset existence checks**: Verify active dataset before queries
- **Error handling**: Graceful fallbacks if agents fail
- **Type validation**: Pydantic models for request/response

---

## 🚀 Performance Optimizations

1. **Singleton Pattern**: DatasetManager uses singleton to avoid multiple instances
2. **Schema Caching**: VectorDBManager caches table schemas
3. **Bulk Inserts**: FileProcessor uses execute_values for fast CSV imports
4. **Lazy Loading**: Schema loaded on-demand, cached for reuse
5. **Connection Pooling**: PostgreSQL connection pooling via psycopg2

---

## 📝 Key Takeaways

1. **Multi-Agent Architecture**: Specialized agents handle specific tasks
2. **Intelligent Routing**: Intent detection determines which agents to use
3. **Orchestrated Execution**: PlannerAgent coordinates agent execution order
4. **Dependency Management**: Agents execute based on data dependencies
5. **Flexible Query Types**: Supports data analysis, semantic search, and conversational queries
6. **Error Resilience**: Fallback mechanisms at multiple levels
7. **State Management**: DatasetManager maintains active dataset state across requests

---

## 🔍 Debugging Tips

- Check execution plan in response: `execution_plan` field shows which agents ran
- View agent routing: PlannerAgent prints routing info to console
- Monitor active dataset: Check `has_dataset` field in response
- SQL debugging: Check `sql_query` field in analyze endpoint response
- Vector search: Use `/vector-stats` endpoint to check vector database status

