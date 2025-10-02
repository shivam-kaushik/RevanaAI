import psycopg2
import logging
import pandas as pd
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import urllib.parse

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.db_config = {
            'dbname': 'Revana',
            'user': 'postgres',
            'password': 'Password1!',
            'host': 'localhost',
            'port': '5432'
        }
        # Create SQLAlchemy engine
        self.engine = self._create_engine()
    
    def _create_engine(self):
        """Create SQLAlchemy engine"""
        password_encoded = urllib.parse.quote_plus(self.db_config['password'])
        connection_string = f"postgresql://{self.db_config['user']}:{password_encoded}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        return create_engine(connection_string)
    
    def get_connection(self):
        """Get PostgreSQL database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """Execute SQL query and return results as DataFrame using SQLAlchemy"""
        try:
            if params:
                result = pd.read_sql_query(text(query), self.engine, params=params)
            else:
                result = pd.read_sql_query(text(query), self.engine)
            return result
        except Exception as e:
            # If it's an UPDATE/INSERT query that doesn't return rows, return empty DataFrame
            if "doesn't return rows" in str(e) or "no results to fetch" in str(e):
                return pd.DataFrame()
            logger.error(f"Query execution failed: {e}")
            raise e
    
    def execute_query_dict(self, query, params=None):
        """Execute SQL query and return results as dictionary"""
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                # Get column names
                columns = result.keys()
                # Fetch all rows and convert to list of dictionaries
                rows = result.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise e
        
    def execute_non_query(self, query, params=None):
        """Execute SQL query that does not return rows (e.g., UPDATE, INSERT, DELETE)"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(query), params or {})
                conn.commit()  # Commit changes
            return True
        except Exception as e:
            logger.error(f"Non-query execution failed: {e}")
            return False
    
    def get_active_tables(self):
        """Get all Revana tables"""
        try:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'revana_%'
                ORDER BY table_name DESC
            """
            result = self.execute_query(query)
            return result['table_name'].tolist() if not result.empty else []
        except Exception as e:
            logger.error(f"Failed to get active tables: {e}")
            return []
    
    def get_latest_table(self):
        """Get the most recently created Revana table"""
        tables = self.get_active_tables()
        return tables[0] if tables else None

    def get_table_columns(self, table_name):
        """Get column names for a specific table"""
        try:
            query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = %s
                ORDER BY ordinal_position
            """
            result = self.execute_query(query, (table_name,))
            return result['column_name'].tolist() if not result.empty else []
        except Exception as e:
            logger.error(f"Failed to get table columns: {e}")
            return []

# Global instance - REMOVE THE IMPORT FROM SELF
db_manager = DatabaseManager()