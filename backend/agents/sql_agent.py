import logging
import re
from openai import OpenAI
from backend.config import Config
from backend.utils.dataset_manager import DatasetManager
from backend.utils.database import db_manager
from backend.utils.vector_db import vector_db

logger = logging.getLogger(__name__)

class SQLAgent:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.dataset_manager = DatasetManager()
    
    def generate_sql(self, user_query, active_table=None):
        """Generate SQL query from natural language"""
        try:
            logger.info(f"🔍 SQL_AGENT: Generating SQL for: {user_query}")
            
            # Get active dataset from DB if not provided
            if not active_table:
                active_info = self.dataset_manager.get_active_dataset()
                active_table = active_info['table_name'] if isinstance(active_info, dict) and active_info.get('table_name') else None
            
            if not active_table:
                return None, "No active dataset. Please upload a CSV file first."
            
            # Get schema information
            schema_info = self._get_table_schema(active_table)
            
            # Get additional schema context from vector_db
            schema_context = vector_db.get_schema_context(user_query)
            
            system_prompt = f"""
            You are an expert SQL query generator. Convert natural language questions to PostgreSQL queries.
            
            Current Active Table: {active_table}
            
            Database Schema:
            {schema_info}
            
            Additional Schema Context:
            {schema_context}
            
            Rules:
            - Only generate SELECT queries (read-only)
            - Use the exact table name: {active_table}
            - Use proper aggregation (SUM, COUNT, AVG) when needed
            - Include appropriate GROUP BY clauses for breakdowns
            - Use WHERE clauses for filtering
            - For string comparisons, use LOWER(column) = 'value' for case-insensitivity
            - For date operations, use to_timestamp(date_column, 'MM/DD/YYYY HH24:MI')::DATE for proper date parsing
            - For aggregates like SUM, use COALESCE(SUM(...), 0) to avoid NULL results
            - Return only the SQL query, no explanations
            
            IMPORTANT DATE HANDLING:
            - If you see date columns in the schema, assume they are stored as text in 'MM/DD/YYYY HH24:MI' format
            - ALWAYS use to_timestamp(date_column, 'MM/DD/YYYY HH24:MI')::DATE for ALL date operations
            - For date filtering in WHERE clauses, use: to_timestamp(date_column, 'MM/DD/YYYY HH24:MI')::DATE BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
            - In SELECT clause, use: to_timestamp(date_column, 'MM/DD/YYYY HH24:MI')::DATE AS invoice_date
            - In GROUP BY and ORDER BY, use the alias: invoice_date
            - NEVER use direct date column comparisons like "WHERE invoicedate BETWEEN"
            - Example: SELECT to_timestamp(invoicedate, 'MM/DD/YYYY HH24:MI')::DATE AS invoice_date, SUM(quantity) FROM table WHERE to_timestamp(invoicedate, 'MM/DD/YYYY HH24:MI')::DATE BETWEEN '2011-01-01' AND '2011-12-31' GROUP BY invoice_date ORDER BY invoice_date
            
            If the query asks for columns that don't exist in the schema, return an error message explaining what columns are available.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate SQL query for: {user_query}"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean the SQL query (remove markdown code blocks if any)
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # Post-process to fix common issues
            sql_query = self._fix_date_format_issues(sql_query)
            sql_query = self._apply_sql_patches(sql_query)
            
            # Validate SQL
            validated_sql = self._validate_sql(sql_query)
            if validated_sql:
                logger.info(f"✅ SQL_AGENT: Generated SQL: {validated_sql}")
                return validated_sql, None
            else:
                return None, "Generated SQL query is invalid."
                
        except Exception as e:
            logger.error(f"❌ SQL_AGENT Error: {e}")
            return None, f"Error generating SQL: {str(e)}"
    
    def _get_table_schema(self, table_name):
        """Get table schema information"""
        try:
            from backend.utils.database import db_manager
            query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            columns_info = db_manager.execute_query_dict(query)
            schema_description = f"Table: {table_name}\nColumns:\n"
            for row in columns_info:
                schema_description += f"- {row['column_name']} ({row['data_type']})\n"
            
            # Get sample data
            sample_query = f"SELECT * FROM {table_name} LIMIT 3"
            sample_rows = db_manager.execute_query_dict(sample_query)
            if sample_rows:
                schema_description += f"\nSample data:\n{self._rows_to_table(sample_rows)}"
            
            return schema_description
            
        except Exception as e:
            logger.error(f"Schema retrieval error: {e}")
            return f"Table: {table_name} (schema unavailable)"
    
    def _validate_sql(self, sql_query):
        """Basic SQL validation"""
        if not sql_query:
            return None
        
        if not sql_query.strip().upper().startswith('SELECT'):
            logger.warning("❌ SQL_AGENT: Only SELECT queries allowed")
            return None
        
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
            logger.warning("❌ SQL_AGENT: Query contains dangerous operations")
            return None
        
        return sql_query
    
    def _fix_date_format_issues(self, sql_query):
        """Fix common date format issues in generated SQL"""
        import re
        
        # First, fix direct date casting issues in SELECT
        # Replace invoicedate::DATE with to_timestamp(invoicedate, 'MM/DD/YYYY HH24:MI')::DATE AS invoice_date
        sql_query = re.sub(
            r'SELECT\s+(\w+date)::DATE',
            r"SELECT to_timestamp(\1, 'MM/DD/YYYY HH24:MI')::DATE AS invoice_date",
            sql_query,
            flags=re.IGNORECASE
        )
        
        # Fix other direct date casting issues (not in SELECT)
        sql_query = re.sub(
            r'(?<!SELECT\s)(\w+date)::DATE',
            r"to_timestamp(\1, 'MM/DD/YYYY HH24:MI')::DATE",
            sql_query,
            flags=re.IGNORECASE
        )
        
        # Fix WHERE clauses with direct date column comparisons
        sql_query = re.sub(
            r'WHERE\s+(\w+date)\s+BETWEEN',
            r"WHERE to_timestamp(\1, 'MM/DD/YYYY HH24:MI')::DATE BETWEEN",
            sql_query,
            flags=re.IGNORECASE
        )
        
        # Fix WHERE clauses with date comparisons
        sql_query = re.sub(
            r'WHERE\s+to_timestamp\((\w+date),\s*\'MM/DD/YYYY HH24:MI\'\)::DATE\s*([<>=]+)',
            r"WHERE to_timestamp(\1, 'MM/DD/YYYY HH24:MI')::DATE \2",
            sql_query,
            flags=re.IGNORECASE
        )
        
        # Fix BETWEEN clauses
        sql_query = re.sub(
            r'to_timestamp\((\w+date),\s*\'MM/DD/YYYY HH24:MI\'\)::DATE\s+BETWEEN',
            r"to_timestamp(\1, 'MM/DD/YYYY HH24:MI')::DATE BETWEEN",
            sql_query,
            flags=re.IGNORECASE
        )
        
        # Fix GROUP BY clauses - use the alias
        sql_query = re.sub(
            r'GROUP BY\s+to_timestamp\((\w+date),\s*\'MM/DD/YYYY HH24:MI\'\)::DATE',
            r"GROUP BY invoice_date",
            sql_query,
            flags=re.IGNORECASE
        )
        
        # Fix ORDER BY clauses - use the alias
        sql_query = re.sub(
            r'ORDER BY\s+to_timestamp\((\w+date),\s*\'MM/DD/YYYY HH24:MI\'\)::DATE',
            r"ORDER BY invoice_date",
            sql_query,
            flags=re.IGNORECASE
        )
        
        # Clean up any duplicate aliases
        sql_query = re.sub(
            r'AS invoice_date AS \w+',
            r'AS invoice_date',
            sql_query,
            flags=re.IGNORECASE
        )
        
        return sql_query
    
    def _apply_sql_patches(self, sql_query):
        """Apply common SQL patches for robustness"""
        if not sql_query:
            return sql_query
        
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
        
        # COALESCE for SUM (avoid double COALESCE)
        sql_query = re.sub(r"COALESCE\(COALESCE\(SUM\(([^)]+)\), 0\), 0\)", r"COALESCE(SUM(\1), 0)", sql_query)
        sql_query = re.sub(r"SUM\(([^)]+)\)", r"COALESCE(SUM(\1), 0)", sql_query)
        
        return sql_query
    
    def execute_query(self, sql_query):
        """Execute SQL query and return results"""
        if not sql_query:
            return None
        
        try:
            logger.info(f"🔍 SQL_AGENT: Executing query: {sql_query}")
            results = db_manager.execute_query_dict(sql_query)
            logger.info(f"✅ SQL_AGENT: Retrieved {len(results) if results else 0} rows")
            return results
        except Exception as e:
            logger.error(f"❌ SQL_AGENT: Query execution failed: {e}")
            return None
    
    def _rows_to_table(self, rows):
        """Format list of dicts as a simple table string"""
        if not rows:
            return "<empty>"
        
        cols = list(rows[0].keys())
        col_widths = {c: max(len(str(c)), max((len(str(r.get(c, ''))) for r in rows), default=0)) for c in cols}
        header = " | ".join(str(c).ljust(col_widths[c]) for c in cols)
        sep = "-+-".join('-' * col_widths[c] for c in cols)
        lines = [header, sep]
        for r in rows:
            lines.append(" | ".join(str(r.get(c, '')).ljust(col_widths[c]) for c in cols))
        return "\n".join(lines)
