from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from backend.config import Config

class InsightAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def generate_insights(self, user_query, data_results, context=""):
        """Generate insights and narratives from data"""
        print(f"🤖 INSIGHT_AGENT: Generating insights for query: {user_query}")
        
        # Prepare data context
        data_context = ""
        if data_results and not data_results.empty:
            data_summary = data_results.describe()
            data_context = f"""
            Data Summary:
            {data_summary.to_string()}
            
            First few rows:
            {data_results.head().to_string()}
            """
        
        system_prompt = """
        You are a data analyst providing insights about retail sales data.
        Create clear, concise, and actionable insights based on the data provided.
        
        Guidelines:
        - Focus on key trends and patterns
        - Provide business-relevant insights
        - Be specific and data-driven
        - Use simple, clear language
        - Highlight important findings
        - Suggest potential actions when relevant
        """
        
        try:
            message = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                User Question: {user_query}
                
                {data_context}
                
                Additional Context: {context}
                
                Please provide insightful analysis:
                """)
            ]
            
            response = self.llm(message)
            insights = response.content.strip()
            
            print(f"✅ INSIGHT_AGENT: Insights generated successfully")
            return insights
            
        except Exception as e:
            print(f"❌ INSIGHT_AGENT Error: {e}")
            return "I analyzed the data but couldn't generate specific insights at this time."