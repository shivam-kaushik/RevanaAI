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
from backend.agents.analysis_agent import AnalysisAgent
from backend.agents.forecast_agent import ForecastAgent
from backend.agents.anomaly_agent import AnomalyAgent
from backend.utils.file_processor import FileProcessor
from backend.utils.vector_db import vector_db
from backend.config import Config
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
analysis_agent = AnalysisAgent()
forecast_agent = ForecastAgent()
anomaly_agent = AnomalyAgent()
file_processor = FileProcessor()
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
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
        
        logger.info(f"🔄 Setting active dataset to: {table_name}")
        success = dataset_manager.set_active_dataset(table_name)
        
        if success:
            # FORCE REFRESH to ensure cache is cleared
            active_dataset = dataset_manager.get_active_dataset(force_refresh=True)
            logger.info(f"✅ Dataset switch successful: {active_dataset['table_name']}")
            
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
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/datasets/active")
async def get_active_dataset():
    """Get the currently active dataset"""
    try:
        # FORCE REFRESH to get latest from database
        active_dataset = dataset_manager.get_active_dataset(force_refresh=True)
        
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
    """Analyze data using the new agent architecture"""
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
        
        # Check if this is an anomaly detection query
        anomaly_keywords = ["anomal", "outlier", "unusual", "drop", "spike", "irregular", "abnormal", "unexpected"]
        is_anomaly_query = any(keyword in request.message.lower() for keyword in anomaly_keywords)
        
        # Step 1: Generate SQL query
        sql_query, sql_error = sql_agent.generate_sql(request.message)
        if not sql_query:
            return {
                "success": False,
                "error": f"Could not generate SQL query: {sql_error}"
            }
        
        # Step 2: Execute SQL query
        from backend.utils.database import db_manager
        try:
            if is_anomaly_query:
                # For anomaly queries, get DataFrame directly
                data_results = db_manager.execute_query(sql_query)
                if data_results.empty:
                    return {
                        "success": False,
                        "error": "No data found for your query"
                    }
                
                # DEBUG: Log what we got from SQL
                logger.info(f"📊 SQL returned {len(data_results)} rows")
                logger.info(f"📊 Columns: {data_results.columns.tolist()}")
                logger.info(f"📊 First few rows:\n{data_results.head()}")
                if 'date' in data_results.columns:
                    logger.info(f"📊 Date range: {data_results['date'].min()} to {data_results['date'].max()}")
                    logger.info(f"📊 Unique dates: {data_results['date'].unique()}")
            else:
                # For regular queries, get dict format
                data_rows = db_manager.execute_query_dict(sql_query)
                if not data_rows:
                    return {
                        "success": False,
                        "error": "No data found for your query"
                    }
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {
                "success": False,
                "error": f"Database query error: {str(e)}"
            }
        
        # Step 3: Handle anomaly detection or generate insights
        if is_anomaly_query:
            try:
                logger.info("🚨 Running anomaly detection...")
                # Decide if this is a grouped anomaly scenario by scanning for common group columns
                group_candidates = ['product', 'product_name', 'product_title', 'product_id', 'sku', 'item', 'item_name', 'brand', 'brand_name', 'product_category', 'category', 'category_name']
                group_col = next((c for c in group_candidates if c in data_results.columns), None)
                if group_col:
                    logger.info(f"🧩 Group column detected ('{group_col}'); running grouped anomaly detection")
                    anomaly_result = anomaly_agent.detect_anomalies_by_category(
                        data_results,
                        time_column='date',
                        value_column='total_amount',
                        category_column=group_col
                    )
                else:
                    anomaly_result = anomaly_agent.detect_anomalies(
                        data_results,
                        time_column='date',
                        value_column='total_amount'
                    )
                
                if anomaly_result.get('success'):
                    # Get the plot HTML (single or grouped)
                    plot_html = anomaly_result['plot'].to_html(full_html=False, include_plotlyjs='cdn')
                    summary = anomaly_agent.summarize_anomalies(anomaly_result)
                    
                    logger.info(f"✅ Plot HTML generated, length: {len(plot_html)}")
                    logger.info(f"✅ Plot HTML preview: {plot_html[:200]}...")
                    
                    # Format response (text only, plot will be rendered separately)
                    if group_col:
                        response_text = f"🚨 **Category Anomaly Detection Results**\n\n"
                    else:
                        response_text = f"🚨 **Anomaly Detection Results**\n\n"
                    response_text += f"**Dataset:** {active_dataset['original_filename']}\n\n"
                    response_text += f"**SQL Query Used:**\n```sql\n{sql_query}\n```\n\n"
                    response_text += f"**Analysis Summary:**\n{summary}\n\n"
                    
                    logger.info(f"✅ Returning response with plot_html field")
                    
                    return {
                        "success": True,
                        "response": response_text,
                        "plot_html": plot_html,  # Send plot separately
                        "data": anomaly_result.get('anomalies', anomaly_result.get('anomalies_by_category', [])),
                        "grouped": bool(group_col),
                        "group_column": group_col,
                        "statistics": anomaly_result.get('statistics', {}),
                        "sql_query": sql_query
                    }
                else:
                    return {
                        "success": False,
                        "error": anomaly_result.get('error', 'Anomaly detection failed')
                    }
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "error": f"Anomaly detection failed: {str(e)}"
                }
        
        # Regular analysis flow
        insights = analysis_agent.generate_insights(request.message, data_rows)
        
        # Step 4: Check if visualization is requested
        charts = {}
        user_query_lower = request.message.lower()
        visualization_keywords = ['chart', 'graph', 'plot', 'visualize', 'show me', 'display', 'bar', 'pie', 'line', 'histogram']
        
        if any(keyword in user_query_lower for keyword in visualization_keywords):
            chart_image, chart_error = analysis_agent.create_visualization(request.message, data_rows)
            if chart_image:
                charts['main'] = chart_image
            elif chart_error:
                logger.warning(f"Visualization failed: {chart_error}")
        
        # Format the response
        response_text = f"📊 **Analysis Results**\n\n"
        response_text += f"**Dataset:** {active_dataset['original_filename']}\n"
        response_text += f"**SQL Query Used:**\n```sql\n{sql_query}\n```\n\n"
        response_text += f"**Data Insights:**\n{insights}\n\n"
        response_text += f"**Records Found:** {len(data_rows)}\n"
        
        # Add sample data
        if data_rows:
            sample_data = _format_data_table(data_rows[:5])
            response_text += f"**Sample Data:**\n```\n{sample_data}\n```\n"
        
        return {
            "success": True,
            "response": response_text,
            "data": data_rows,
            "charts": charts,
            "sql_query": sql_query
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
        # Always sync in-memory vector_db with DB-backed active dataset for this process
        try:
            if active_dataset_info and active_dataset_info.get('table_name'):
                vector_db.set_active_dataset(active_dataset_info['table_name'])
        except Exception as sync_err:
            logger.warning(f"Could not sync vector_db active dataset: {sync_err}")
        
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
            # FORCE REFRESH active dataset before executing agent plan
            active_dataset_fresh = dataset_manager.get_active_dataset(force_refresh=True)
            logger.info(f"📊 Executing with active dataset: {active_dataset_fresh['table_name'] if active_dataset_fresh else 'None'}")
            
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
    
    # FORCE REFRESH: Ensure we have the latest active dataset
    active_dataset = dataset_manager.get_active_dataset(force_refresh=True)
    if active_dataset and active_dataset.get('table_name'):
        try:
            vector_db.set_active_dataset(active_dataset['table_name'])
            logger.info(f"🔄 Agent plan executing with dataset: {active_dataset['table_name']}")
        except Exception as sync_err:
            logger.warning(f"Agent plan sync warning: {sync_err}")
    table_name = active_dataset['table_name'] if active_dataset else None
    
    # Check if this is a pure vector search query
    if (len(plan.get('execution_plan', [])) == 1 and 
        plan['execution_plan'][0].get('agent') == 'VECTOR_AGENT'):
        logger.info("🔍 Pure vector search query detected")
        vector_response = vector_agent.handle_semantic_query(user_query)
        return {
            'final_response': vector_response,
            'data': None
        }
    
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
            insights = analysis_agent.generate_insights(user_query, data_results)
        
        elif agent_name == "FORECAST_AGENT" and data_results is not None:
            # Generate forecasts
            forecasts = forecast_agent.generate_forecast(data_results)
        
        elif agent_name == "ANOMALY_AGENT" and data_results is not None:
            # Detect anomalies with visualization
            try:
                group_candidates = ['product', 'product_name', 'product_title', 'product_id', 'sku', 'item', 'item_name', 'brand', 'brand_name', 'product_category', 'category', 'category_name']
                group_col = next((c for c in group_candidates if c in data_results.columns), None)
                if group_col:
                    anomaly_result = anomaly_agent.detect_anomalies_by_category(
                        data_results,
                        time_column='date',
                        value_column='total_amount',
                        category_column=group_col
                    )
                else:
                    anomaly_result = anomaly_agent.detect_anomalies(
                        data_results,
                        time_column='date',
                        value_column='total_amount'
                    )
                if anomaly_result.get('success'):
                    # Get the plot HTML
                    plot_html = anomaly_result['plot'].to_html(full_html=False, include_plotlyjs='cdn')
                    anomalies = {
                        'summary': anomaly_agent.summarize_anomalies(anomaly_result),
                        'plot_html': plot_html,
                        'anomalies': anomaly_result.get('anomalies', anomaly_result.get('anomalies_by_category', [])),
                        'statistics': anomaly_result.get('statistics', {})
                    }
                else:
                    anomalies = {'message': anomaly_result.get('error', 'Anomaly detection failed')}
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                anomalies = {'message': f'Error detecting anomalies: {str(e)}'}
    
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

def _format_data_table(rows):
    """Format list of dicts as a simple table string"""
    if not rows:
        return "<empty>"
    
    # Build a simple aligned table string from list of dicts
    cols = list(rows[0].keys())
    col_widths = {c: max(len(str(c)), max((len(str(r.get(c, ''))) for r in rows), default=0)) for c in cols}
    header = " | ".join(str(c).ljust(col_widths[c]) for c in cols)
    sep = "-+-".join('-' * col_widths[c] for c in cols)
    lines = [header, sep]
    for r in rows:
        lines.append(" | ".join(str(r.get(c, '')).ljust(col_widths[c]) for c in cols))
    return "\n".join(lines)

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
    
    if anomalies:
        if isinstance(anomalies, dict):
            if 'summary' in anomalies:
                # New anomaly format with plot
                response_parts.append(f"🚨 **Anomaly Detection:**\n{anomalies['summary']}")
                if anomalies.get('plot_html'):
                    response_parts.append(f"**Visualization:**\n{anomalies['plot_html']}")
                if anomalies.get('anomalies'):
                    anomaly_count = len(anomalies['anomalies'])
                    response_parts.append(f"\n**Found {anomaly_count} anomalous periods**")
            elif 'message' in anomalies:
                response_parts.append(f"🚨 **Anomaly Detection:**\n{anomalies['message']}")
        else:
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