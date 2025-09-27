from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from backend.utils.vector_db import vector_db
from backend.utils.database import db_manager
from backend.config import Config
import re

class SQLAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.schema_info = self._get_schema_info()
    
    def _get_schema_info(self):
        """Get database schema information"""
        schema_df = db_manager.get_table_schema()
        schema_info = "Database Schema:\n"
        for _, row in schema_df.iterrows():
            schema_info += f"{row['column_name']} ({row['data_type']}) - {'NULL' if row['is_nullable'] == 'YES' else 'NOT NULL'}\n"
        return schema_info
    
    def generate_sql(self, user_query, context=""):
        """Generate SQL query from natural language"""
        
        # Get similar examples from vector DB
        similar_examples = vector_db.search_similar(user_query)
        examples_text = "\n".join([f"Example: {ex}" for ex in similar_examples])
        
        system_prompt = f"""
        You are an expert SQL query generator. Convert natural language questions to PostgreSQL queries.
        
        {self.schema_info}
        
        Important rules:
        - Only generate SELECT queries (read-only)
        - Use proper aggregation (SUM, COUNT, AVG) when needed
        - Include appropriate GROUP BY clauses for breakdowns
        - Use WHERE clauses for filtering
        - Always use the actual column names from the schema
        - Return only the SQL query, no explanations
        
        Examples:
        {examples_text}
        
        Context: {context}
        """
        
        try:
            message = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {user_query}\n\nSQL Query:")
            ]
            
            response = self.llm(message)
            sql_query = self._extract_sql_query(response.content)
            
            return self._validate_sql(sql_query)
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return None
    
    def _extract_sql_query(self, text):
        """Extract SQL query from LLM response"""
        # Look for SQL between ```sql ... ``` or just the query
        sql_match = re.search(r'```sql\n(.*?)\n```', text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # If no code blocks, try to find SELECT statement
        select_match = re.search(r'(SELECT.*?)(?=;|$)', text, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip() + ';'
        
        return text.strip()
    
    def _validate_sql(self, sql_query):
        """Basic SQL validation"""
        if not sql_query:
            return None
        
        # Check if it's a SELECT query
        if not sql_query.strip().upper().startswith('SELECT'):
            print("Error: Only SELECT queries are allowed")
            return None
        
        # Check for dangerous keywords
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
            print("Error: Query contains dangerous operations")
            return None
        
        return sql_query
    
    def execute_query(self, sql_query):
        """Execute SQL query and return results"""
        if not sql_query:
            return None
        
        try:
            results = db_manager.execute_query(sql_query)
            return results
        except Exception as e:
            print(f"Error executing query: {e}")
            return None