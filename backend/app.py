import sys
import os
import logging
from openai import OpenAI

# Fix Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import dependencies
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

# Import our agents and utilities
from backend.agents.planner import PlannerAgent
from backend.agents.sql_agent import SQLAgent
from backend.agents.insight_agent import InsightAgent
from backend.agents.forecast_agent import ForecastAgent
from backend.agents.anomaly_agent import AnomalyAgent
from backend.utils.file_processor import FileProcessor
from backend.utils.vector_db import vector_db
from backend.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Sales Assistant", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
planner = PlannerAgent()
sql_agent = SQLAgent()
insight_agent = InsightAgent()
forecast_agent = ForecastAgent()
anomaly_agent = AnomalyAgent()
file_processor = FileProcessor()
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

# In-memory storage
conversation_history = {}

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    data: dict = None
    agents_used: list = []
    execution_plan: list = []
    needs_clarification: bool = False
    clarification_question: str = ""
    has_dataset: bool = False

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend"""
    try:
        frontend_path = os.path.join(parent_dir, 'frontend', 'templates', 'index.html')
        if os.path.exists(frontend_path):
            return FileResponse(frontend_path)
    except:
        pass
    
    return HTMLResponse(content="<html><body><h1>Agentic Sales Assistant</h1><p>Frontend not found</p></body></html>")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    active_dataset = vector_db.get_active_dataset()
    tables = file_processor.list_tables() if hasattr(file_processor, 'list_tables') else []
    return {
        "status": "healthy", 
        "message": "Server is running",
        "active_dataset": active_dataset,
        "database_tables": tables
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle CSV file upload and automatically create database table"""
    try:
        logger.info(f"Processing file upload: {file.filename}")
        
        result = await file_processor.process_uploaded_file(file, file.filename)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"File uploaded successfully! Created table '{result['table_name']}' with {result['row_count']} rows.",
                "dataset_info": result,
                "table_name": result["table_name"],
                "database": "revana_database.db"
            }
        else:
            return {
                "success": False,
                "message": f"Upload failed: {result.get('error', 'Unknown error')}"
            }
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {
            "success": False,
            "message": f"Upload failed: {str(e)}"
        }

@app.get("/tables")
async def list_tables():
    """List all tables in Revana database"""
    try:
        tables = file_processor.list_tables()
        return {
            "success": True,
            "tables": tables,
            "database": "revana_database.db"
        }
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/table/{table_name}")
async def get_table_info(table_name: str):
    """Get information about a specific table"""
    try:
        info = file_processor.get_table_info(table_name)
        return {
            "success": True,
            "table_info": info
        }
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/query")
async def execute_query(query: str):
    """Execute a custom SQL query"""
    try:
        from backend.utils.database import db_manager
        result = db_manager.execute_query(query)
        return {
            "success": True,
            "data": result.to_dict('records'),
            "row_count": len(result),
            "columns": list(result.columns) if not result.empty else []
        }
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with intelligent routing"""
    try:
        logger.info(f"Processing query: {request.message}")
        
        # Check if we have database tables
        has_database_tables = len(file_processor.list_tables()) > 0
        
        # Step 1: Plan the query execution - FIXED: Only pass user_query
        plan = planner.create_plan(request.message)
        
        # Check if plan needs clarification
        if plan.get('needs_clarification', False):
            return ChatResponse(
                response="I need more information to help you.",
                needs_clarification=True,
                clarification_question=plan.get('clarification_question', 'Could you provide more details?'),
                agents_used=["PLANNER"],
                execution_plan=plan.get('execution_plan', []),
                has_dataset=has_database_tables
            )
        
        # Step 2: Determine if this is a data query
        is_data_query = plan.get('is_data_query', False) and has_database_tables
        
        if is_data_query:
            # Use agents for data analysis
            results = await execute_agent_plan(plan, has_database_tables)
            response_data = results.get('data', {})
        else:
            # Use ChatGPT for conversational queries
            chat_response = await get_chatgpt_response(request.message, has_database_tables)
            results = {'final_response': chat_response}
            response_data = None
        
        # Step 3: Prepare response
        return ChatResponse(
            response=results['final_response'],
            data=response_data if response_data is not None else {},
            agents_used=plan.get('required_agents', []) if is_data_query else ["CHATGPT"],
            execution_plan=plan.get('execution_plan', []),
            needs_clarification=False,
            has_dataset=has_database_tables
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            response="Sorry, I encountered an error processing your request.",
            agents_used=[],
            execution_plan=[],
            needs_clarification=False,
            has_dataset=len(file_processor.list_tables()) > 0
        )

async def execute_agent_plan(plan, has_database_tables):
    """Execute the agent plan for data analysis"""
    user_query = plan['user_query']
    data_results = None
    insights = ""
    forecasts = ""
    anomalies = ""
    
    # Execute each agent in the plan
    for step in plan.get('execution_plan', []):
        agent_name = step.get('agent', '')
        
        if agent_name == "SQL_AGENT" and has_database_tables:
            # Generate and execute SQL query
            sql_query, error = sql_agent.generate_sql(user_query)
            if sql_query:
                try:
                    from backend.utils.database import db_manager
                    data_results = db_manager.execute_query(sql_query)
                    logger.info(f"✅ SQL query executed successfully: {len(data_results)} rows returned")
                except Exception as e:
                    logger.error(f"❌ SQL execution error: {e}")
                    return {'final_response': f"❌ Database query error: {str(e)}"}
            elif error:
                return {'final_response': f"❌ SQL Error: {error}"}
        
        elif agent_name == "INSIGHT_AGENT" and data_results is not None:
            # Generate insights from data
            insights = insight_agent.generate_insights(user_query, data_results)
        
        elif agent_name == "FORECAST_AGENT" and data_results is not None:
            # Generate forecasts
            forecasts = forecast_agent.generate_forecast(data_results)
        
        elif agent_name == "ANOMALY_AGENT" and data_results is not None:
            # Detect anomalies
            anomalies = anomaly_agent.detect_anomalies(data_results)
    
    # Combine results into final response
    final_response = build_final_response(insights, forecasts, anomalies, data_results, user_query)
    
    return {
        'final_response': final_response,
        'data': {
            'sql_data': data_results.to_dict('records') if data_results is not None else None,
            'forecasts': forecasts,
            'anomalies': anomalies
        } if data_results is not None else None
    }

async def get_chatgpt_response(user_query, has_dataset):
    """Get response from ChatGPT for conversational queries"""
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

def build_final_response(insights, forecasts, anomalies, data_results, user_query):
    """Build the final response from all agent outputs"""
    response_parts = []
    
    if insights:
        response_parts.append(f"📊 **Insights:**\n{insights}")
    
    if forecasts and isinstance(forecasts, dict):
        forecast_text = "\n".join([f"- {k}: {v}" for k, v in forecasts.items()])
        response_parts.append(f"🔮 **Forecasts:**\n{forecast_text}")
    elif forecasts:
        response_parts.append(f"🔮 **Forecasts:**\n{forecasts}")
    
    if anomalies and isinstance(anomalies, dict):
        response_parts.append(f"🚨 **Anomaly Detection:**\n{anomalies.get('message', 'No significant anomalies detected')}")
    elif anomalies:
        response_parts.append(f"🚨 **Anomaly Detection:**\n{anomalies}")
    
    if data_results is not None and not data_results.empty:
        response_parts.append(f"📈 **Data Summary:** Retrieved {len(data_results)} records")
        # Add sample data preview for small results
        if len(data_results) <= 10:
            sample_data = data_results.to_string(index=False)
            response_parts.append(f"**Data:**\n```\n{sample_data}\n```")
        else:
            sample_data = data_results.head(5).to_string(index=False)
            response_parts.append(f"**Sample Data (first 5 rows):**\n```\n{sample_data}\n```")
    
    if not response_parts:
        response_parts.append(f"I've analyzed your query: '{user_query}'")
    
    return "\n\n".join(response_parts)

# Serve static files
@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files"""
    static_path = os.path.join(parent_dir, 'frontend', 'static', file_path)
    if os.path.exists(static_path):
        return FileResponse(static_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting Agentic Sales Assistant v2.0...")
    logger.info("📁 Features: Dynamic CSV uploads + Intelligent agent routing + Automatic database tables")
    
    # Test components
    try:
        from backend.utils.database import db_manager
        if db_manager.test_connection():
            logger.info("✅ Database connection successful")
        else:
            logger.warning("⚠ Database connection failed")
    except Exception as e:
        logger.warning(f"⚠ Database: {e}")
    
    # Check for existing tables
    try:
        tables = file_processor.list_tables()
        if tables:
            logger.info(f"📊 Found {len(tables)} existing tables: {tables}")
        else:
            logger.info("📊 No existing tables found - ready for uploads")
    except Exception as e:
        logger.warning(f"⚠ Table check: {e}")
    
    active_dataset = vector_db.get_active_dataset()
    if active_dataset:
        logger.info(f"📊 Active dataset: {active_dataset}")
    else:
        logger.info("📊 No active dataset - waiting for file upload")
    
    logger.info("✅ All systems ready!")
    logger.info("📡 Server ready at http://localhost:8000")
    logger.info("💾 Database: revana_database.db")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="info")