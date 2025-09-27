from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from backend.config import Config
import re

class PlannerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.system_prompt = """
        You are an AI sales assistant planner. Analyze the user's query and determine which actions are needed.
        
        Available actions:
        - SQL_QUERY: Need to query the database for data
        - FORECAST: User wants predictions about future trends
        - ANOMALY_DETECTION: User wants to identify unusual patterns
        - VISUALIZATION: User wants charts/graphs
        - EXPLANATION: User wants explanations or insights
        
        Respond with a JSON format:
        {
            "actions": ["SQL_QUERY", "FORECAST", ...],
            "reasoning": "Brief explanation of why these actions are needed",
            "query_type": "sales_trends|product_analysis|customer_analysis|...",
            "time_frame": "last_month|last_quarter|last_year|custom...",
            "needs_clarification": true/false,
            "clarification_question": "What specifically would you like to know?"
        }
        """
    
    def plan_actions(self, user_query):
        """Plan which actions are needed for the user query"""
        try:
            message = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = self.llm(message)
            
            # Extract JSON from response
            import json
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback if JSON parsing fails
                return {
                    "actions": ["SQL_QUERY"],
                    "reasoning": "Default action for data query",
                    "query_type": "general",
                    "time_frame": "unspecified",
                    "needs_clarification": False,
                    "clarification_question": ""
                }
                
        except Exception as e:
            print(f"Error in planner: {e}")
            return {
                "actions": ["SQL_QUERY"],
                "reasoning": "Error occurred, defaulting to SQL query",
                "query_type": "general",
                "time_frame": "unspecified",
                "needs_clarification": False,
                "clarification_question": ""
            }