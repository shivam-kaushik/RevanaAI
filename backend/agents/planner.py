from langchain.schema import HumanMessage, SystemMessage
from backend.utils.intent_detector import IntentDetector
from backend.utils.vector_db import vector_db
from backend.config import Config
import json

class PlannerAgent:
    def __init__(self):
        self.intent_detector = IntentDetector()
        self.agent_descriptions = {
            "SQL_AGENT": "Handles database queries and data retrieval",
            "INSIGHT_AGENT": "Generates insights and narratives from data",
            "FORECAST_AGENT": "Creates predictions and future trend forecasts", 
            "ANOMALY_AGENT": "Detects unusual patterns and outliers in data",
            "VISUALIZATION_AGENT": "Creates charts and visualizations",
            "VECTOR_AGENT": "Performs semantic search for products and customers"
        }
    
    def create_plan(self, user_query):
        """Create an execution plan based on user query"""
        print(f"🔍 Planning for query: {user_query}")
        
        # Check if we have an active dataset
        has_active_dataset = vector_db.get_active_dataset() is not None
        
        # Step 1: Detect intent
        intent_result = self.intent_detector.detect_intent(user_query, has_active_dataset)
        
        # Step 2: Create execution plan
        plan = {
            "user_query": user_query,
            "is_data_query": intent_result["is_data_query"],
            "primary_intent": intent_result["primary_intent"],
            "required_agents": intent_result["required_agents"],
            "execution_plan": self._generate_execution_plan(intent_result) if intent_result["is_data_query"] else [],
            "reasoning": intent_result["reasoning"],
            "needs_clarification": intent_result.get("needs_clarification", False),
            "clarification_question": intent_result.get("clarification_question", ""),
            "has_active_dataset": has_active_dataset
        }
        
        # Print routing information
        self._print_routing_info(plan)
        
        return plan
    
    def _generate_execution_plan(self, intent_result):
        """Generate execution steps for data queries"""
        execution_steps = []
        agents = intent_result["required_agents"]

        # Special case: VECTOR_AGENT only (semantic search)
        if "VECTOR_AGENT" in agents and len(agents) == 1:
            execution_steps.append({
                "step": 1,
                "agent": "VECTOR_AGENT",
                "description": "Perform semantic search using vector embeddings",
                "dependencies": []
            })
            return execution_steps

        #if "FORECAST_AGENT" in agents and "SQL_AGENT" not in agents:
            #agents = ["SQL_AGENT"] + agents
        
        # ------- New forecast approaches ---------
        q = intent_result.get("user_query", "").lower()

        # NEW: if user explicitly asks to predict/forecast for N periods, run forecast end-to-end
        if any(k in q for k in ["predict", "forecast", "projection", "next ", "future"]):
            return [{
                "step": 1,
                "agent": "FORECAST_AGENT",
                "description": "End-to-end forecast (NL→SQL→fetch→Prophet→viz→summary)",
                "dependencies": []
            }]
        # Special case: VECTOR_AGENT only (semantic search)
        if "FORECAST_AGENT" in agents:
            execution_steps.append({
                "step": 1,
                "agent": "FORECAST_AGENT",
                "description": "End-to-end forecast (NL→SQL→fetch→Prophet→viz→summary)",
                "dependencies": []
            })
            return execution_steps
        # ------------------------------------------
        # Always start with SQL agent for data queries (except vector-only)
        if "SQL_AGENT" in agents:
            execution_steps.append({
                "step": 1,
                "agent": "SQL_AGENT",
                "description": "Query database for relevant data",
                "dependencies": []
            })
        
        # Add other agents in logical order
        step_number = 2
        
        if "ANOMALY_AGENT" in agents:
            execution_steps.append({
                "step": step_number,
                "agent": "ANOMALY_AGENT",
                "description": "Analyze data for unusual patterns",
                "dependencies": ["SQL_AGENT"]
            })
            step_number += 1
        
        if "FORECAST_AGENT" in agents:
            execution_steps.append({
                "step": step_number,
                "agent": "FORECAST_AGENT",
                "description": "Generate future predictions",
                "dependencies": ["SQL_AGENT"]
            })
            step_number += 1
        
        if "VISUALIZATION_AGENT" in agents:
            execution_steps.append({
                "step": step_number,
                "agent": "VISUALIZATION_AGENT",
                "description": "Create data visualizations",
                "dependencies": ["SQL_AGENT"]
            })
            step_number += 1
        
        # Insight agent usually comes last
        if "INSIGHT_AGENT" in agents:
            execution_steps.append({
                "step": step_number,
                "agent": "INSIGHT_AGENT",
                "description": "Generate insights and narrative",
                "dependencies": [agent for agent in agents if agent != "INSIGHT_AGENT"]
            })
        
        return execution_steps
    
    def _print_routing_info(self, plan):
        """Print detailed routing information"""
        print("\n" + "="*60)
        print("🚀 QUERY PLANNER - AGENT ROUTING")
        print("="*60)
        print(f"📝 User Query: {plan['user_query']}")
        print(f"🎯 Data Query: {plan['is_data_query']}")
        print(f"📊 Active Dataset: {plan['has_active_dataset']}")
        
        if plan['is_data_query']:
            print(f"🤖 Required Agents: {', '.join(plan['required_agents'])}")
            print(f"📋 Execution Steps:")
            for step in plan['execution_plan']:
                agent_desc = self.agent_descriptions.get(step['agent'], step['agent'])
                print(f"   {step['step']}. {step['agent']} - {agent_desc}")
        else:
            print(f"💬 Routing to: ChatGPT (Conversational)")
        
        print(f"💭 Reasoning: {plan['reasoning']}")
        if plan['needs_clarification']:
            print(f"❓ Clarification Needed: {plan['clarification_question']}")
        print("="*60 + "\n")