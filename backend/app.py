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
from backend.agents.data_analyzer import DataAnalyzer
from backend.agents.vector_agent import VectorAgent
from backend.utils.dataset_manager import DatasetManager

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
data_analyzer = DataAnalyzer()
vector_agent = VectorAgent()
dataset_manager = DatasetManager()

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

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend"""
    try:
        frontend_path = os.path.join(parent_dir, 'frontend', 'templates', 'index.html')
        if os.path.exists(frontend_path):
            return FileResponse(frontend_path)
    except:
        pass
    
    # Return the HTML directly if file not found
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
    """Health check endpoint"""
    try:
        # Auto-set latest dataset if none is active
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

# Add new endpoint for semantic search
@app.post("/semantic-search")
async def semantic_search(request: ChatRequest):
    """Handle semantic search queries using pgvector"""
    try:
        logger.info(f"Semantic search request: {request.message}")
        
        # Use vector agent for semantic search
        results = vector_agent.handle_semantic_query(request.message)
        
        return {
            "success": True,
            "response": results,
            "query_type": "semantic_search"
        }
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return {
            "success": False,
            "error": f"Semantic search failed: {str(e)}"
        }

# Add vector stats endpoint
@app.get("/vector-stats")
async def get_vector_stats():
    """Get statistics about vector data"""
    try:
        stats = vector_agent.vector_store.get_vector_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Vector stats error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle CSV file upload and automatically create database table"""
    try:
        logger.info(f"Processing file upload: {file.filename}")
        
        result = await file_processor.process_uploaded_file(file, file.filename)
        
        if isinstance(result, dict) and result.get("success") is True:
            # Register the dataset in dataset manager
            success = dataset_manager.register_dataset(
                table_name=result["table_name"],
                filename=file.filename,
                row_count=result["row_count"],
                column_count=result["column_count"],
                description="Retail sales dataset",
                is_active=True  # Set the newly uploaded dataset as active
            )

            vector_db.set_active_dataset(result["table_name"])
            
            # Return the response in the format the frontend expects
            return {
                "success": True,
                "message": f"File uploaded successfully! Created table '{result['table_name']}' with {result['row_count']} rows. This dataset is now active.",
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
            
            return {
                "success": False,
                "message": f"Upload failed: {error_message}"
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
            "database": "PostgreSQL"
        }
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/datasets")
async def list_datasets():
    """Get all available datasets"""
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
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/datasets/active")
async def set_active_dataset(request: dict):
    """Set a specific dataset as active"""
    try:
        table_name = request.get('table_name')
        if not table_name:
            return {
                "success": False,
                "error": "Table name is required"
            }
        
        success = dataset_manager.set_active_dataset(table_name)
        
        if success:
            active_dataset = dataset_manager.get_active_dataset()
            return {
                "success": True,
                "message": f"Dataset '{table_name}' is now active",
                "active_dataset": active_dataset
            }
        else:
            return {
                "success": False,
                "error": f"Could not set '{table_name}' as active dataset"
            }
            
    except Exception as e:
        logger.error(f"Error setting active dataset: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/datasets/active")
async def get_active_dataset():
    """Get the currently active dataset"""
    try:
        active_dataset = dataset_manager.get_active_dataset()
        
        if active_dataset:
            return {
                "success": True,
                "active_dataset": active_dataset
            }
        else:
            return {
                "success": False,
                "error": "No active dataset"
            }
            
    except Exception as e:
        logger.error(f"Error getting active dataset: {e}")
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

@app.post("/analyze")
async def analyze_data(request: ChatRequest):
    """Analyze data using GPT-powered natural language queries"""
    try:
        logger.info(f"Data analysis request: {request.message}")
        
        # Check if we have an active dataset
        if not dataset_manager.has_active_dataset():
            return {
                "success": False,
                "error": "No active dataset found. Please upload a CSV file first or select an existing dataset."
            }
        
        # Get active dataset info
        active_dataset = dataset_manager.get_active_dataset()
        
        # Ensure data_analyzer has db_manager
        from backend.utils.database import db_manager
        if not hasattr(data_analyzer, 'db_manager') or data_analyzer.db_manager is None:
            data_analyzer.set_db_manager(db_manager)
        
        # Use data analyzer for comprehensive analysis
        analysis_result = data_analyzer.analyze_query(request.message)
        
        if "error" in analysis_result:
            return {
                "success": False,
                "error": analysis_result["error"]
            }
        
        # Format the response
        response_text = f"📊 **Analysis Results**\n\n"
        response_text += f"**Dataset:** {active_dataset['original_filename']}\n"
        response_text += f"**SQL Query Used:**\n```sql\n{analysis_result['sql_query']}\n```\n\n"
        response_text += f"**Data Insights:**\n{analysis_result['insights']}\n\n"
        response_text += f"**Records Found:** {analysis_result['row_count']}\n"
        
        # Add sample data
        if analysis_result['data']:
            sample_data = pd.DataFrame(analysis_result['data']).head(5).to_string(index=False)
            response_text += f"**Sample Data:**\n```\n{sample_data}\n```\n"
        
        return {
            "success": True,
            "response": response_text,
            "data": analysis_result['data'],
            "charts": analysis_result.get('charts', {}),
            "sql_query": analysis_result['sql_query']
        }
        
    except Exception as e:
        logger.error(f"Data analysis error: {e}")
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        }
    
@app.post("/query")
async def execute_query(request: dict):
    """Execute a custom SQL query"""
    try:
        query = request.get('query', '')
        if not query:
            return {
                "success": False,
                "error": "Query is required"
            }
            
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
        
        # Use dataset manager to check for active dataset
        has_active_dataset = dataset_manager.has_active_dataset()
        active_dataset_info = dataset_manager.get_active_dataset()
        
        # Step 1: Plan the query execution
        plan = planner.create_plan(request.message)
        
        # OVERRIDE: If planner says no dataset but we actually have one, fix it
        if not plan.get('has_active_dataset', False) and has_active_dataset:
            logger.info("🔄 Overriding planner: We have active dataset but planner doesn't know!")
            plan['has_active_dataset'] = True
            # Also ensure data queries are enabled if it's a data query
            if plan.get('is_data_query', False):
                plan['required_agents'] = plan.get('required_agents', []) or ["SQL_AGENT", "INSIGHT_AGENT"]
                if not plan.get('execution_plan'):
                    plan['execution_plan'] = [
                        {"agent": "SQL_AGENT", "step": 1, "description": "Query database"},
                        {"agent": "INSIGHT_AGENT", "step": 2, "description": "Generate insights"}
                    ]
        
        # Check if plan needs clarification
        if plan.get('needs_clarification', False):
            return ChatResponse(
                response="I need more information to help you.",
                needs_clarification=True,
                clarification_question=plan.get('clarification_question', 'Could you provide more details?'),
                agents_used=["PLANNER"],
                execution_plan=plan.get('execution_plan', []),
                has_dataset=has_active_dataset
            )
        
        # Step 2: Determine if this is a data query
        is_data_query = plan.get('is_data_query', False) and has_active_dataset
        
        if is_data_query:
            # Use agents for data analysis
            results = await execute_agent_plan(plan, has_active_dataset)
            response_data = results.get('data', {}) or {}
            agents_used = plan.get('required_agents', []) or ["SQL_AGENT", "INSIGHT_AGENT"]
        else:
            # Use ChatGPT for conversational queries
            chat_response = await get_chatgpt_response(request.message, has_active_dataset)
            results = {'final_response': chat_response}
            response_data = {}
            agents_used = ["CHATGPT"]
        
        # Step 3: Prepare response
        return ChatResponse(
            response=results['final_response'],
            data=response_data,
            agents_used=agents_used,
            execution_plan=plan.get('execution_plan', []),
            needs_clarification=False,
            has_dataset=has_active_dataset
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        # Fallback to ChatGPT for any errors
        try:
            has_active_dataset = dataset_manager.has_active_dataset()
            fallback_response = await get_chatgpt_response(request.message, has_active_dataset)
            return ChatResponse(
                response=fallback_response,
                data={},
                agents_used=["CHATGPT"],
                execution_plan=[],
                needs_clarification=False,
                has_dataset=has_active_dataset
            )
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return ChatResponse(
                response="Hello! I'm here to help. You can upload a CSV file for data analysis or ask me questions about sales analytics.",
                data={},
                agents_used=[],
                execution_plan=[],
                needs_clarification=False,
                has_dataset=dataset_manager.has_active_dataset()
            )

async def execute_agent_plan(plan, has_database_tables):
    """Execute the agent plan for data analysis"""
    user_query = plan['user_query']
    data_results = None
    insights = ""
    forecasts = ""
    anomalies = ""
    
    # Get active dataset from vector_db for consistency
    table_name = vector_db.get_active_dataset()
    active_dataset = dataset_manager.get_active_dataset() if table_name else None
    
    # Execute each agent in the plan
    for step in plan.get('execution_plan', []):
        agent_name = step.get('agent', '')
        
        if agent_name == "SQL_AGENT" and has_database_tables and table_name:
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
    
    # Test database connection first
    try:
        from backend.utils.database import db_manager
        if db_manager.test_connection():
            logger.info("✅ Database connection successful")
            data_analyzer.set_db_manager(db_manager)
        else:
            logger.warning("⚠ Database connection failed")
    except Exception as e:
        logger.warning(f"⚠ Database: {e}")
    
    # Initialize dataset manager
    try:
        logger.info("📊 Initializing dataset manager...")
        
        # Check if datasets table exists
        table_exists = dataset_manager.ensure_datasets_table()
        
        if table_exists:
            # Get current datasets
            datasets = dataset_manager.get_available_datasets()
            logger.info(f"📊 Found {len(datasets)} registered datasets")
            
            # Check for existing data tables that aren't registered
            all_tables = file_processor.list_tables()
            revana_tables = [t for t in all_tables if t.startswith('revana_') and t != 'revana_datasets']
            
            if revana_tables:
                logger.info(f"📋 Found {len(revana_tables)} revana data tables in database")
                
                # Register any unregistered tables
                registered_count = 0
                for table_name in revana_tables:
                    # Check if already registered
                    existing_dataset = dataset_manager.get_dataset_by_name(table_name)
                    if not existing_dataset:
                        try:
                            # Get table info
                            table_info = file_processor.get_table_info(table_name)
                            if table_info:
                                # Extract filename
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
            
            # Refresh datasets list after registration
            datasets = dataset_manager.get_available_datasets()
            logger.info(f"📊 Total datasets available: {len(datasets)}")
            
            # Set active dataset if none exists
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