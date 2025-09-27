from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
from backend.config import Config
from backend.agents.planner import PlannerAgent
from backend.agents.sql_agent import SQLAgent

app = FastAPI(title="Agentic Sales Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
planner = PlannerAgent()
sql_agent = SQLAgent()

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    data: dict = None
    actions: list = []
    needs_clarification: bool = False
    clarification_question: str = ""

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        user_query = request.message
        
        # Step 1: Plan actions
        plan = planner.plan_actions(user_query)
        
        if plan.get('needs_clarification', False):
            return ChatResponse(
                response="I need more information to help you.",
                needs_clarification=True,
                clarification_question=plan.get('clarification_question', '')
            )
        
        # Step 2: Execute SQL query if needed
        data_result = None
        if 'SQL_QUERY' in plan.get('actions', []):
            sql_query = sql_agent.generate_sql(user_query)
            if sql_query:
                data_result = sql_agent.execute_query(sql_query)
                
                # Convert DataFrame to JSON serializable format
                if data_result is not None and not data_result.empty:
                    data_result = {
                        'columns': data_result.columns.tolist(),
                        'data': data_result.to_dict('records'),
                        'sql_query': sql_query
                    }
        
        # Step 3: Generate response
        response_text = generate_response(user_query, plan, data_result)
        
        return ChatResponse(
            response=response_text,
            data=data_result,
            actions=plan.get('actions', [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

def generate_response(user_query, plan, data_result):
    """Generate natural language response based on results"""
    if data_result and not data_result.get('data'):
        return "I couldn't find any data matching your query. Please try rephrasing your question."
    
    # Simple response generation (will be enhanced later)
    actions = plan.get('actions', [])
    
    if 'SQL_QUERY' in actions:
        if data_result and data_result.get('data'):
            num_records = len(data_result['data'])
            return f"I found {num_records} records matching your query. Here's the data analysis."
        else:
            return "I've processed your data query, but no specific results were returned."
    
    return "I've processed your request. Here are the results."

import os

@app.get("/")
async def serve_frontend():
    file_path = os.path.join(os.path.dirname(__file__), "../frontend/templates/index.html")
    return FileResponse(os.path.abspath(file_path))

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("Starting Agentic Sales Assistant...")
    
    # Test database connection
    try:
        from backend.utils.database import db_manager
        schema = db_manager.get_table_schema()
        print("Database connection successful")
        print(f"Table schema: {len(schema)} columns")
    except Exception as e:
        print(f"Database connection failed: {e}")

if __name__ == "__main__" or __name__ == "backend.app":
    import uvicorn
    print(f"Starting server on host={Config.HOST}, port={Config.PORT}, debug={Config.DEBUG}")
    try:
        uvicorn.run(app, host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
    except Exception as e:
        print(f"Error starting server: {e}")