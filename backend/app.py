import sys
import os
import logging
import base64
from openai import OpenAI

# Fix Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# FastAPI deps
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

# --- NEW: imports needed for ForecastAgent wiring ---
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from backend.agents.forecast_agent import ForecastAgent, ForecastAgentConfig
# ---------------------------------------------------

# Import our agents and utilities
from backend.agents.planner import PlannerAgent
from backend.agents.sql_agent import SQLAgent
from backend.agents.analysis_agent import AnalysisAgent
from backend.agents.anomaly_agent import AnomalyAgent
from backend.agents.vector_agent import VectorAgent
from backend.agents.feedback_agent import FeedbackAgent  # <-- NEW

from backend.utils.file_processor import FileProcessor
from backend.utils.vector_db import vector_db
from backend.utils.dataset_manager import DatasetManager
from backend.config import Config
from backend.utils.database import db_manager  # used in several places

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Sales Assistant", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- helper to build schema text ----------------
def build_schema_text(engine, active_table: str) -> str:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='public'
            ORDER BY table_name, ordinal_position
        """)).fetchall()

    tables = {}
    for t, c, dt in rows:
        tables.setdefault(t, []).append(f"- {c} ({dt})")

    lines = []
    lines.append(f"ACTIVE_TABLE: {active_table}")
    for t, cols in tables.items():
        lines.append(f"Table {t}:\n" + "\n".join(cols))

    lines.append(f"""
    Example monthly aggregation using ACTIVE_TABLE:
    SELECT
        date_trunc('month', to_timestamp(date, 'MM-DD-YYYY HH24:MI:SS'))::date AS ds,
        COALESCE(SUM(line_total), 0) AS y
    FROM {active_table}
    WHERE to_timestamp(date, 'MM-DD-YYYY HH24:MI:SS')::date <= CURRENT_DATE
    GROUP BY 1
    ORDER BY 1;
    """)
    return "\n\n".join(lines)
# ------------------------------------------------------------------

# Initialize components
planner = PlannerAgent()
sql_agent = SQLAgent()
analysis_agent = AnalysisAgent()
anomaly_agent = AnomalyAgent()
file_processor = FileProcessor()
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
vector_agent = VectorAgent()
dataset_manager = DatasetManager()
feedback_agent = FeedbackAgent()  # <-- NEW

# ---------------- ForecastAgent wiring ----------------
DATABASE_URL = os.getenv("DATABASE_URL", getattr(Config, "DATABASE_URL", None))
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set (env or Config)")

_engine = create_engine(DATABASE_URL, future=True)
active = dataset_manager.get_active_dataset(force_refresh=True)
active_table = active["table_name"] if active else "revana_online_retail_clean_..."
SCHEMA_TEXT = build_schema_text(_engine, active_table)

FC_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=Config.OPENAI_API_KEY,
)

FORECAST_STATIC_DIR = os.path.join(parent_dir, "frontend", "static", "forecast")
print("DEBUG: FORECAST STATIC DIR", FORECAST_STATIC_DIR)
os.makedirs(FORECAST_STATIC_DIR, exist_ok=True)

def _png_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


fc_cfg = ForecastAgentConfig(
    database_url=DATABASE_URL,
    schema_text=SCHEMA_TEXT,
    output_dir=FORECAST_STATIC_DIR,
    default_horizon=6,
)
forecast_agent = ForecastAgent(cfg=fc_cfg, llm=FC_LLM)
# ----------------------------------------------------------

# In-memory storage
conversation_history = {}

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    data: dict = {}
    agents_used: list = []
    execution_plan: list = []
    needs_clarification: bool = False
    clarification_question: str = ""
    has_dataset: bool = False
    charts: dict | None = None
    interaction_id: int | None = None   # <-- NEW

class FeedbackRequest(BaseModel):       # <-- NEW
    interaction_id: int
    rating: int
    comment: str | None = None


@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        frontend_path = os.path.join(parent_dir, 'frontend', 'templates', 'index.html')
        if os.path.exists(frontend_path):
            return FileResponse(frontend_path)
    except:
        pass

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agentic Sales Assistant</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        <h1>Agentic Sales Assistant</h1>
        <p>Please use the main application interface.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    try:
        if not dataset_manager.has_active_dataset():
            dataset_manager.auto_set_latest_dataset()

        active_dataset = dataset_manager.get_active_dataset()
        available_datasets = dataset_manager.get_available_datasets()

        return {
            "status": "healthy",
            "message": "Server is running",
            "active_dataset": active_dataset,
            "available_datasets_count": len(available_datasets),
            "database": "PostgreSQL"
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "healthy",
            "message": "Server is running (dataset info unavailable)",
            "active_dataset": None,
            "available_datasets_count": 0,
            "database": "Unknown"
        }

@app.post("/semantic-search")
async def semantic_search(request: ChatRequest):
    try:
        logger.info(f"Semantic search request: {request.message}")
        results = vector_agent.handle_semantic_query(request.message)
        return {"success": True, "response": results, "query_type": "semantic_search"}
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return {"success": False, "error": f"Semantic search failed: {str(e)}"}

@app.get("/vector-stats")
async def get_vector_stats():
    try:
        stats = vector_agent.vector_store.get_vector_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Vector stats error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file upload: {file.filename}")
        result = await file_processor.process_uploaded_file(file, file.filename)

        if isinstance(result, dict) and result.get("success") is True:
            success = dataset_manager.register_dataset(
                table_name=result["table_name"],
                filename=file.filename,
                row_count=result["row_count"],
                column_count=result["column_count"],
                description="Retail sales dataset",
                is_active=True
            )

            vector_db.set_active_dataset(result["table_name"])

            return {
                "success": True,
                "message": (
                    f"File uploaded successfully! Created table '{result['table_name']}' "
                    f"with {result['row_count']} rows. This dataset is now active."
                ),
                "dataset_info": {
                    "table_name": result["table_name"],
                    "row_count": result["row_count"],
                    "column_count": result["column_count"],
                    "database": "PostgreSQL"
                },
                "table_name": result["table_name"],
                "database": "PostgreSQL"
            }
        else:
            error_message = "Unknown error"
            if isinstance(result, dict):
                error_message = result.get('error', 'Upload failed')
            elif isinstance(result, str):
                error_message = result
            return {"success": False, "message": f"Upload failed: {error_message}"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"success": False, "message": f"Upload failed: {str(e)}"}

@app.get("/tables")
async def list_tables():
    try:
        tables = file_processor.list_tables()
        return {"success": True, "tables": tables, "database": "PostgreSQL"}
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return {"success": False, "error": str(e)}

@app.get("/datasets")
async def list_datasets():
    try:
        datasets = dataset_manager.get_available_datasets()
        active_dataset = dataset_manager.get_active_dataset()
        return {
            "success": True,
            "datasets": datasets,
            "active_dataset": active_dataset,
            "total_datasets": len(datasets)
        }
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return {"success": False, "error": str(e)}

@app.post("/datasets/active")
async def set_active_dataset(request: dict):
    try:
        table_name = request.get('table_name')
        if not table_name:
            return {"success": False, "error": "Table name is required"}

        logger.info(f"🔄 Setting active dataset to: {table_name}")
        success = dataset_manager.set_active_dataset(table_name)

        if success:
            active_dataset = dataset_manager.get_active_dataset(force_refresh=True)
            logger.info(f"✅ Dataset switch successful: {active_dataset['table_name']}")
            return {
                "success": True,
                "message": f"Dataset '{table_name}' is now active",
                "active_dataset": active_dataset
            }
        else:
            return {"success": False, "error": f"Could not set '{table_name}' as active dataset"}
    except Exception as e:
        logger.error(f"Error setting active dataset: {e}")
        import traceback; traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/datasets/active")
async def get_active_dataset():
    try:
        active_dataset = dataset_manager.get_active_dataset(force_refresh=True)
        if active_dataset:
            return {"success": True, "active_dataset": active_dataset}
        else:
            return {"success": False, "error": "No active dataset"}
    except Exception as e:
        logger.error(f"Error getting active dataset: {e}")
        return {"success": False, "error": str(e)}

@app.get("/table/{table_name}")
async def get_table_info(table_name: str):
    try:
        info = file_processor.get_table_info(table_name)
        return {"success": True, "table_info": info}
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        return {"success": False, "error": str(e)}

@app.post("/analyze")
async def analyze_data(request: ChatRequest):
    """
    Analyze data with SQL + insight agent.
    Now also logs interaction + returns interaction_id for feedback.
    """
    try:
        logger.info(f"Data analysis request: {request.message}")

        if not dataset_manager.has_active_dataset():
            return {
                "success": False,
                "error": "No active dataset found. Please upload a CSV file first or select an existing dataset."
            }

        active_dataset = dataset_manager.get_active_dataset()

        sql_query, sql_error = sql_agent.generate_sql(request.message)
        if not sql_query:
            return {"success": False, "error": f"Could not generate SQL query: {sql_error}"}

        try:
            data_rows = db_manager.execute_query_dict(sql_query)
            if not data_rows:
                return {"success": False, "error": "No data found for your query"}
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {"success": False, "error": f"Database query error: {str(e)}"}

        insights = analysis_agent.generate_insights(request.message, data_rows)

        charts = {}
        user_query_lower = request.message.lower()
        visualization_keywords = [
            'chart', 'graph', 'plot', 'visualize', 'show me', 'display',
            'bar', 'pie', 'line', 'histogram'
        ]
        if any(keyword in user_query_lower for keyword in visualization_keywords):
            chart_image, chart_error = analysis_agent.create_visualization(request.message, data_rows)
            if chart_image:
                charts['main'] = chart_image

        response_text = f"📊 **Analysis Results**\n\n"
        response_text += f"**Dataset:** {active_dataset['original_filename']}\n"
        response_text += f"**SQL Query Used:**\n```sql\n{sql_query}\n```\n\n"
        response_text += f"**Data Insights:**\n{insights}\n\n"
        response_text += f"**Records Found:** {len(data_rows)}\n"

        if data_rows:
            sample_data = _format_data_table(data_rows[:5])
            response_text += f"**Sample Data:**\n```\n{sample_data}\n```"

        # ------ NEW: log interaction for feedback ------
        try:
            interaction_id = feedback_agent.log_interaction(
                session_id=request.conversation_id,
                user_query=request.message,
                agent_name="SQL+INSIGHT",
                dataset_table=active_dataset.get("table_name") if active_dataset else None,
                response_summary=response_text[:500],
                chart_reference="has_chart" if charts else None,
            )
        except Exception as log_err:
            logger.error(f"Feedback logging failed (analyze): {log_err}")
            interaction_id = None
        # ------------------------------------------------

        return {
            "success": True,
            "response": response_text,
            "data": data_rows,
            "charts": charts,
            "sql_query": sql_query,
            "interaction_id": interaction_id,   # <-- NEW
        }
    except Exception as e:
        logger.error(f"Data analysis error: {e}")
        return {"success": False, "error": f"Analysis failed: {str(e)}"}

@app.post("/query")
async def execute_query(request: dict):
    try:
        query = request.get('query', '')
        if not query:
            return {"success": False, "error": "Query is required"}

        result = db_manager.execute_query(query)
        return {
            "success": True,
            "data": result.to_dict('records'),
            "row_count": len(result),
            "columns": list(result.columns) if not result.empty else []
        }
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with planner flow + feedback logging.
    """
    try:
        logger.info(f"Processing query: {request.message}")

        has_active_dataset = dataset_manager.has_active_dataset()
        active_dataset_info = dataset_manager.get_active_dataset()
        try:
            if active_dataset_info and active_dataset_info.get('table_name'):
                vector_db.set_active_dataset(active_dataset_info['table_name'])
        except Exception as sync_err:
            logger.warning(f"Could not sync vector_db active dataset: {sync_err}")

        plan = planner.create_plan(request.message)

        if not plan.get('has_active_dataset', False) and has_active_dataset:
            logger.info("🔄 Overriding planner: We have active dataset but planner doesn't know!")
            plan['has_active_dataset'] = True
            if plan.get('is_data_query', False):
                plan['required_agents'] = plan.get('required_agents', []) or ["SQL_AGENT", "INSIGHT_AGENT"]
                if not plan.get('execution_plan'):
                    plan['execution_plan'] = [
                        {"agent": "SQL_AGENT", "step": 1, "description": "Query database"},
                        {"agent": "INSIGHT_AGENT", "step": 2, "description": "Generate insights"}
                    ]

        if plan.get('needs_clarification', False):
            return ChatResponse(
                response="I need more information to help you.",
                needs_clarification=True,
                clarification_question=plan.get('clarification_question', 'Could you provide more details?'),
                agents_used=["PLANNER"],
                execution_plan=plan.get('execution_plan', []),
                has_dataset=has_active_dataset,
                interaction_id=None,
            )

        is_data_query = plan.get('is_data_query', False) and has_active_dataset

        charts = None
        interaction_id = None

        if is_data_query:
            active_dataset_fresh = dataset_manager.get_active_dataset(force_refresh=True)
            logger.info(f"📊 Executing with active dataset: {active_dataset_fresh['table_name'] if active_dataset_fresh else 'None'}")

            results = await execute_agent_plan(plan, has_active_dataset)
            charts = results.get("charts")
            response_data = results.get('data', {}) or {}
            agents_used = plan.get('required_agents', []) or ["SQL_AGENT", "INSIGHT_AGENT"]
            final_text = results['final_response']

            # --------- NEW: log interaction ----------
            try:
                interaction_id = feedback_agent.log_interaction(
                    session_id=request.conversation_id,
                    user_query=request.message,
                    agent_name="+".join(agents_used),
                    dataset_table=active_dataset_fresh.get("table_name") if active_dataset_fresh else None,
                    response_summary=final_text[:500],
                    chart_reference="has_chart" if charts else None,
                )
            except Exception as log_err:
                logger.error(f"Feedback logging failed (chat data): {log_err}")
                interaction_id = None
            # -----------------------------------------
        else:
            chat_response = await get_chatgpt_response(request.message, has_active_dataset)
            results = {'final_response': chat_response}
            response_data = {}
            agents_used = ["CHATGPT"]
            final_text = chat_response

            # log conversational interaction as well
            try:
                interaction_id = feedback_agent.log_interaction(
                    session_id=request.conversation_id,
                    user_query=request.message,
                    agent_name="CHATGPT",
                    dataset_table=active_dataset_info.get("table_name") if active_dataset_info else None,
                    response_summary=final_text[:500],
                    chart_reference=None,
                )
            except Exception as log_err:
                logger.error(f"Feedback logging failed (chat conv): {log_err}")
                interaction_id = None

        return ChatResponse(
            response=results['final_response'],
            data=response_data,
            agents_used=agents_used,
            execution_plan=plan.get('execution_plan', []),
            needs_clarification=False,
            has_dataset=has_active_dataset,
            charts=charts,
            interaction_id=interaction_id,
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        try:
            has_active_dataset = dataset_manager.has_active_dataset()
            fallback_response = await get_chatgpt_response(request.message, has_active_dataset)
            return ChatResponse(
                response=fallback_response,
                data={},
                agents_used=["CHATGPT"],
                execution_plan=[],
                needs_clarification=False,
                has_dataset=has_active_dataset,
                interaction_id=None,
            )
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return ChatResponse(
                response="Hello! I'm here to help. You can upload a CSV file for data analysis or ask me questions about sales analytics.",
                data={},
                agents_used=[],
                execution_plan=[],
                needs_clarification=False,
                has_dataset=dataset_manager.has_active_dataset(),
                interaction_id=None,
            )

async def execute_agent_plan(plan, has_database_tables):
    # ... (UNCHANGED BODY EXCEPT END OF FUNCTION) ...
    user_query = plan['user_query']
    data_results = None
    insights = ""
    forecasts = ""
    anomalies = ""

    active_dataset = dataset_manager.get_active_dataset(force_refresh=True)
    if active_dataset and active_dataset.get('table_name'):
        try:
            vector_db.set_active_dataset(active_dataset['table_name'])
            logger.info(f"🔄 Agent plan executing with dataset: {active_dataset['table_name']}")
        except Exception as sync_err:
            logger.warning(f"Agent plan sync warning: {sync_err}")
    table_name = active_dataset['table_name'] if active_dataset else None

    if (len(plan.get('execution_plan', [])) == 1 and
        plan['execution_plan'][0].get('agent') == 'VECTOR_AGENT'):
        logger.info("🔍 Pure vector search query detected")
        vector_response = vector_agent.handle_semantic_query(user_query)
        return {'final_response': vector_response, 'data': None}

    for step in plan.get('execution_plan', []):
        agent_name = step.get('agent', '')

        if agent_name == "SQL_AGENT" and has_database_tables and table_name:
            sql_query, error = sql_agent.generate_sql(user_query)
            if sql_query:
                try:
                    data_results = db_manager.execute_query(sql_query)
                    logger.info(f"✅ SQL query executed successfully: {len(data_results)} rows returned")
                except Exception as e:
                    logger.error(f"❌ SQL execution error: {e}")
                    return {'final_response': f"❌ Database query error: {str(e)}"}
            elif error:
                return {'final_response': f"❌ SQL Error: {error}"}

        elif agent_name == "INSIGHT_AGENT" and data_results is not None:
            insights = analysis_agent.generate_insights(user_query, data_results)

        elif agent_name == "FORECAST_AGENT":
            try:
                active_now = dataset_manager.get_active_dataset(force_refresh=True)
                table_now = active_now["table_name"] if active_now else None
                if not table_now:
                    return {'final_response': "❌ No active dataset is set."}

                new_schema_text = build_schema_text(_engine, table_now)
                forecast_agent.refresh_schema(new_schema_text)
                forecasts = forecast_agent.run(user_query)
            except Exception as e:
                logger.error(f"Forecast agent error: {e}")
                return {'final_response': f"❌ Forecast error: {str(e)}"}

        elif agent_name == "ANOMALY_AGENT" and data_results is not None:
            anomalies = anomaly_agent.detect_anomalies(data_results)

    final_response = build_final_response(insights, forecasts, anomalies, data_results, user_query)

    charts = {}
    if isinstance(forecasts, dict):
        plots = forecasts.get("plots", {}) or {}

        combined_b64 = plots.get("combined_base64")
        if combined_b64:
            charts["forecast_combined"] = combined_b64
        else:
            combined = plots.get("combined_png")
            if combined:
                fs_path = combined
                if fs_path.startswith("static/"):
                    fs_path = os.path.join(parent_dir, "frontend", fs_path)
                elif fs_path.startswith("/static/"):
                    fs_path = os.path.join(parent_dir, "frontend", fs_path[1:])
                charts["forecast_combined"] = _png_to_base64(fs_path)

    if data_results is not None:
        data_payload = {
            "sql_data": data_results.to_dict("records"),
            "forecasts": forecasts,
            "anomalies": anomalies,
        }
    else:
        data_payload = {"forecasts": forecasts} if forecasts else None
        print("Debug: data payload:", data_payload)

    return {
        "final_response": final_response,
        "data": data_payload,
        "charts": charts if charts else None,
    }

async def get_chatgpt_response(user_query, has_dataset):
    try:
        system_message = "You are a helpful AI assistant for a data analysis platform. "
        if has_dataset:
            system_message += "Users can upload CSV files for analysis. "
        system_message += "Be friendly, helpful, and concise."

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"ChatGPT error: {e}")
        return "I'm here to help! You can upload a CSV file for data analysis or ask me questions."

def _format_data_table(rows):
    if not rows:
        return "<empty>"
    cols = list(rows[0].keys())
    col_widths = {c: max(len(str(c)), max((len(str(r.get(c, ''))) for r in rows), default=0)) for c in cols}
    header = " | ".join(str(c).ljust(col_widths[c]) for c in cols)
    sep = "-+-".join('-' * col_widths[c] for c in cols)
    lines = [header, sep]
    for r in rows:
        lines.append(" | ".join(str(r.get(c, '')).ljust(col_widths[c]) for c in cols))
    return "\n".join(lines)

def build_final_response(insights, forecasts, anomalies, data_results, user_query):
    response_parts = []

    if insights:
        response_parts.append(f"📊 **Insights:**\n{insights}")

    if isinstance(forecasts, dict):
        md = forecasts.get("markdown")
        if md:
            response_parts.append(md)
        else:
            response_parts.append(
                "🔮 **Forecasts:**\n" + "\n".join([f"- {k}: {v}" for k, v in forecasts.items()])
            )
    elif forecasts:
        response_parts.append(f"🔮 **Forecasts:**\n{forecasts}")

    if anomalies and isinstance(anomalies, dict):
        response_parts.append(
            f"🚨 **Anomaly Detection:**\n{anomalies.get('message', 'No significant anomalies detected')}"
        )
    elif anomalies:
        response_parts.append(f"🚨 **Anomaly Detection:**\n{anomalies}")

    if data_results is not None and not data_results.empty:
        response_parts.append(f"📈 **Data Summary:** Retrieved {len(data_results)} records")
        if len(data_results) <= 10:
            sample_data = data_results.to_string(index=False)
            response_parts.append(f"**Data:**\n```\n{sample_data}\n```")
        else:
            sample_data = data_results.head(5).to_string(index=False)
            response_parts.append(f"**Sample Data (first 5 rows):**\n```\n{sample_data}\n```")

    if not response_parts:
        response_parts.append(f"I've analyzed your query: '{user_query}'")

    return "\n\n".join(response_parts)

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    static_path = os.path.join(parent_dir, 'frontend', 'static', file_path)
    print("INFO: static path: ", static_path)
    if os.path.exists(static_path):
        return FileResponse(static_path)
    raise HTTPException(status_code=404, detail="File not found")

# ---------- NEW: feedback endpoint ----------
@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    try:
        rating = max(1, min(5, req.rating))  # clamp 1–5
        feedback_agent.store_feedback(req.interaction_id, rating, req.comment)
        return {"success": True}
    except Exception as e:
        logger.error(f"Feedback submit error: {e}")
        return {"success": False, "error": str(e)}
# --------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Agentic Sales Assistant v2.0...")

    try:
        if db_manager.test_connection():
            logger.info("✅ Database connection successful")
        else:
            logger.warning("⚠ Database connection failed")
    except Exception as e:
        logger.warning(f"⚠ Database: {e}")

    try:
        logger.info("📊 Initializing dataset manager...")
        table_exists = dataset_manager.ensure_datasets_table()

        if table_exists:
            datasets = dataset_manager.get_available_datasets()
            logger.info(f"📊 Found {len(datasets)} registered datasets")

            all_tables = file_processor.list_tables()
            revana_tables = [t for t in all_tables if t.startswith('revana_') and t != 'revana_datasets']

            if revana_tables:
                logger.info(f"📋 Found {len(revana_tables)} revana data tables in database")
                registered_count = 0
                for table_name in revana_tables:
                    existing_dataset = dataset_manager.get_dataset_by_name(table_name)
                    if not existing_dataset:
                        try:
                            table_info = file_processor.get_table_info(table_name)
                            if table_info:
                                original_filename = table_name.replace('revana_', '').replace('_', ' ') + '.csv'
                                success = dataset_manager.register_dataset(
                                    table_name=table_name,
                                    filename=original_filename,
                                    row_count=table_info.get('row_count', 0),
                                    column_count=table_info.get('column_count', 0),
                                    description="Auto-registered dataset",
                                    is_active=False
                                )
                                if success:
                                    registered_count += 1
                                    logger.info(f"✅ Registered: {table_name}")
                        except Exception as e:
                            logger.warning(f"Could not register {table_name}: {e}")
                logger.info(f"🔄 Registered {registered_count} new datasets")

            datasets = dataset_manager.get_available_datasets()
            logger.info(f"📊 Total datasets available: {len(datasets)}")

            active_dataset = dataset_manager.get_active_dataset()
            if active_dataset:
                logger.info(f"🎯 Active dataset: {active_dataset['table_name']}")
            elif datasets:
                logger.info("🔄 Setting first dataset as active...")
                active_dataset = dataset_manager.auto_set_latest_dataset()
                if active_dataset:
                    logger.info(f"🎯 Auto-set active dataset: {active_dataset['table_name']}")
                else:
                    logger.info("📭 Could not auto-set active dataset")
            else:
                logger.info("📭 No datasets available")
        else:
            logger.warning("📭 revana_datasets table does not exist - datasets will be registered on first upload")

    except Exception as e:
        logger.error(f"❌ Dataset manager initialization failed: {e}")

    logger.info("✅ All systems ready!")
    logger.info("📡 Server ready at http://localhost:8000")
    logger.info("💾 Database: PostgreSQL")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="info")
