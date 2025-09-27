import psycopg2
import pandas as pd
from backend.config import Config

class DatabaseManager:
    def __init__(self):
        self.connection_string = Config.DATABASE_URL
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.connection_string)
    
    def execute_query(self, query, params=None):
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def get_table_schema(self, table_name="retail_transactions"):
        """Get schema information for a table"""
        query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
        """
        return self.execute_query(query, (table_name,))
    
    def get_sample_data(self, table_name="retail_transactions", limit=5):
        """Get sample data from table"""
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        return self.execute_query(query)

# Singleton instance
db_manager = DatabaseManager()