import psycopg2
import logging
import pandas as pd
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.db_config = {
            'dbname': 'Revana',
            'user': 'postgres',  # Change if different
            'password': 'Password1!',  # Change to your actual password
            'host': 'localhost',
            'port': '5432'
        }
    
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
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """Execute SQL query and return results as DataFrame"""
        conn = self.get_connection()
        try:
            if params:
                result = pd.read_sql_query(query, conn, params=params)
            else:
                result = pd.read_sql_query(query, conn)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise e
        finally:
            conn.close()
    
    def get_active_tables(self):
        """Get all Revana tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'revana_%'
                ORDER BY table_name DESC
            """)
            tables = [row[0] for row in cursor.fetchall()]
            return tables
        finally:
            cursor.close()
            conn.close()
    
    def get_latest_table(self):
        """Get the most recently created Revana table"""
        tables = self.get_active_tables()
        return tables[0] if tables else None

# Global instance
db_manager = DatabaseManager()