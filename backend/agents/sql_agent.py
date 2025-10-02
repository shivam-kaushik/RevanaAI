from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from backend.utils.database import db_manager
from backend.utils.vector_db import vector_db
from backend.config import Config
import re

class SQLAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY
        )
    

    def generate_sql(self, user_query):
        """Generate SQL query from natural language"""
        print(f"🤖 SQL_AGENT: Generating SQL for: {user_query}")
        
        # Get schema context from vector DB
        schema_context = vector_db.get_schema_context(user_query)
        active_dataset = vector_db.get_active_dataset()
        
        if not active_dataset:
            return None, "No active dataset. Please upload a CSV file first."
        
        system_prompt = f"""
        You are an expert SQL query generator. Convert natural language questions to PostgreSQL queries.
        
        Current Active Table: {active_dataset}
        
        Schema Information:
        {schema_context}
        
        Important rules:
        - Only generate SELECT queries (read-only)
        - Use the actual table name: {active_dataset}
        - Use proper aggregation (SUM, COUNT, AVG) when needed
        - Include appropriate GROUP BY clauses for breakdowns
        - Use WHERE clauses for filtering
        - For string comparisons, use LOWER(column) = 'value' for case-insensitivity.
        - For date comparisons, cast text columns to date using date::DATE.
        - For aggregates like SUM, use COALESCE(SUM(...), 0) to avoid NULL results.
        - Return only the SQL query, no explanations
        
        If the query asks for columns that don't exist in the schema, return an error message explaining what columns are available.
        """
        
        try:
            message = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {user_query}\n\nSQL Query:")
            ]
            
            response = self.llm(message)
            sql_query = self._extract_sql_query(response.content)
            
            # Patch SQL for robustness
            if sql_query:
                # Patch common column name mistakes
                sql_query = re.sub(r'\bcategory\b', 'product_category', sql_query)

                # Case-insensitive product_category
                sql_query = re.sub(
                    r"WHERE\s+product_category\s*=\s*'([^']+)'",
                    lambda m: f"WHERE LOWER(product_category) = '{m.group(1).lower()}'",
                    sql_query,
                    flags=re.IGNORECASE
                )
                # Case-insensitive gender
                sql_query = re.sub(
                    r"WHERE\s+gender\s*=\s*'([^']+)'",
                    lambda m: f"WHERE LOWER(gender) = '{m.group(1).lower()}'",
                    sql_query,
                    flags=re.IGNORECASE
                )
                # Date casting for comparisons
                sql_query = re.sub(r"(\s+date\s*)([<>=]+)", r" date::DATE \2", sql_query, flags=re.IGNORECASE)
                # COALESCE for SUM
                sql_query = re.sub(r"SUM\(([^)]+)\)", r"COALESCE(SUM(\1), 0)", sql_query)
            
            # Validate SQL
            validated_sql = self._validate_sql(sql_query)
            if validated_sql:
                return validated_sql, None
            else:
                return None, "Generated SQL query is invalid."
            
        except Exception as e:
            print(f"❌ SQL_AGENT Error: {e}")
            return None, f"Error generating SQL: {str(e)}"


    
    def _extract_sql_query(self, text):
        """Extract SQL query from LLM response"""
        # Check if it's an error message about missing columns
        if "does not contain" in text or "available columns" in text.lower():
            return None
        
        sql_match = re.search(r'```sql\n(.*?)\n```', text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        select_match = re.search(r'(SELECT.*?)(?=;|$)', text, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip() + ';'
        
        return text.strip()
    
    def _validate_sql(self, sql_query):
        """Basic SQL validation"""
        if not sql_query:
            return None
        
        if not sql_query.strip().upper().startswith('SELECT'):
            print("❌ SQL_AGENT: Only SELECT queries allowed")
            return None
        
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
            print("❌ SQL_AGENT: Query contains dangerous operations")
            return None
        
        return sql_query
    
    def execute_query(self, sql_query):
        """Execute SQL query and return results"""
        if not sql_query:
            return None
        
        try:
            print(f"🤖 SQL_AGENT: Executing query: {sql_query}")
            results = db_manager.execute_query(sql_query)
            print(f"✅ SQL_AGENT: Retrieved {len(results) if results is not None else 0} rows")
            return results
        except Exception as e:
            print(f"❌ SQL_AGENT: Query execution failed: {e}")
            return None