from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from backend.config import Config
import json
import re

class IntentDetector:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.system_prompt = """
        You are an intent detection system for a data analysis assistant.
        Analyze the user's query and determine if it requires data analysis or is just conversational.
        
        Data Analysis Intent (requires agents):
        - Questions about data, numbers, trends, patterns
        - Requests for analysis, insights, summaries
        - Queries about specific metrics or KPIs
        - Requests for predictions, forecasts
        - Looking for anomalies or unusual patterns (show/provide/find/detect/identify anomalies/outliers)
        - Asking for charts, graphs, visualizations
        - Searching for products, items, or best products
        - Looking for customers, users, or similar customers
        - Questions about specific products or product categories
        - Questions about specific customers or customer segments
        - Queries about product recommendations or similarities
        - Queries about customer behavior or preferences
        
        Conversational Intent (use ChatGPT directly):
        - Greetings, small talk
        - Questions about how the system works
        - General knowledge questions not related to data
        - Requests for help or instructions
        
        Return JSON response:
        {
            "is_data_query": true/false,
            "primary_intent": "conversational/data_analysis/forecasting/anomaly_detection/visualization/semantic_search",
            "required_agents": ["list", "of", "agents", "if", "data", "query"],
            "reasoning": "explanation of classification",
            "needs_clarification": true/false,
            "clarification_question": "if clarification needed"
        }
        
        Available agents: SQL_AGENT, INSIGHT_AGENT, FORECAST_AGENT, ANOMALY_AGENT, VISUALIZATION_AGENT, VECTOR_AGENT
        
        IMPORTANT RULES: 
        - SQL_AGENT is REQUIRED for all data analysis queries (except pure VECTOR_AGENT searches)
        - ANOMALY_AGENT requires SQL_AGENT to fetch data first: ["SQL_AGENT", "ANOMALY_AGENT"]
        - INSIGHT_AGENT typically works with data from SQL_AGENT: ["SQL_AGENT", "INSIGHT_AGENT"]
        - FORECAST_AGENT can work independently (has built-in SQL capability)
        - Queries asking for "best products", "find products", "show me products", etc. should use VECTOR_AGENT only
        - Queries asking for "similar customers", "find customers", etc. should use VECTOR_AGENT only
        - ANY query containing words like "anomal", "outlier", "unusual", "spike", "drop", "detect", "provide", "show", "find" combined with data analysis context should return: ["SQL_AGENT", "ANOMALY_AGENT", "INSIGHT_AGENT"]
        - "Provide me anomalies", "Show me anomalies", "Find anomalies", "Detect anomalies" are ALL anomaly detection queries
        """
    
    def detect_intent(self, user_query, has_active_dataset=True):
        """Detect intent and required agents"""
        try:
            context = "Active dataset is available." if has_active_dataset else "No dataset uploaded yet."
            
            message = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Context: {context}\n\nUser Query: {user_query}")
            ]
            
            response = self.llm(message)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return self._fallback_intent_detection(user_query, has_active_dataset)
                
        except Exception as e:
            print(f"Error in intent detection: {e}")
            return self._fallback_intent_detection(user_query, has_active_dataset)
    
    def _fallback_intent_detection(self, user_query, has_active_dataset):
        """Fallback intent detection using keyword matching"""
        query_lower = user_query.lower()
        
        # Check if it's conversational
        conversational_keywords = ['hello', 'hi', 'hey', 'how are you', 'what can you do', 'help', 'thank you']
        if any(keyword in query_lower for keyword in conversational_keywords) or len(query_lower.split()) < 3:
            return {
                "is_data_query": False,
                "primary_intent": "conversational",
                "required_agents": [],
                "reasoning": "Detected as conversational/greeting",
                "needs_clarification": False,
                "clarification_question": ""
            }
        
        # Check for semantic search queries (products/customers)
        semantic_keywords = ['product', 'products', 'item', 'items', 'customer', 'customers', 
                            'similar', 'best', 'top', 'find', 'search', 'show me', 'looking for',
                            'dairy', 'bakery', 'electronics', 'clothing', 'beauty', 
                            'who buy', 'who bought', 'like', 'recommend']
        
        if any(keyword in query_lower for keyword in semantic_keywords):
            return {
                "is_data_query": True,
                "primary_intent": "semantic_search",
                "required_agents": ["VECTOR_AGENT"],
                "reasoning": "Detected as semantic search query for products or customers",
                "needs_clarification": False,
                "clarification_question": ""
            }
        
        # If no active dataset, treat as conversational
        if not has_active_dataset:
            return {
                "is_data_query": False,
                "primary_intent": "conversational",
                "required_agents": [],
                "reasoning": "No dataset available for data analysis",
                "needs_clarification": False,
                "clarification_question": ""
            }
        
        # Data analysis queries
        agents = ["SQL_AGENT"]  # Most data queries need SQL
        
        # Check for specific intents
        forecast_keywords = ['forecast', 'predict', 'next', 'future', 'will', 'going to']
        if any(keyword in query_lower for keyword in forecast_keywords):
            agents.append("FORECAST_AGENT")
        
        # More comprehensive anomaly detection keywords
        anomaly_keywords = ['anomal', 'outlier', 'unusual', 'strange', 'spike', 'drop', 'unexpected', 
                           'irregularit', 'aberration', 'deviation', 'abnormal']
        if any(keyword in query_lower for keyword in anomaly_keywords):
            agents.append("ANOMALY_AGENT")
        
        viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'show me', 'display']
        if any(keyword in query_lower for keyword in viz_keywords):
            agents.append("VISUALIZATION_AGENT")
        
        # Always include insight agent for data analysis
        agents.append("INSIGHT_AGENT")
        
        return {
            "is_data_query": True,
            "primary_intent": "data_analysis",
            "required_agents": list(set(agents)),
            "reasoning": "Detected as data analysis query based on keywords",
            "needs_clarification": False,
            "clarification_question": ""
        }