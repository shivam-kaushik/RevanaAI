import pandas as pd
import os
import logging
from datetime import datetime
from fastapi import UploadFile
import tempfile
import psycopg2
from psycopg2.extras import execute_values
from backend.utils.vector_data_processor import VectorDataProcessor

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        self.db_config = {
            'dbname': 'Revana',
            'user': 'postgres',
            'password': 'Password1!',
            'host': 'localhost',
            'port': '5432'
        }
        self.vector_processor = VectorDataProcessor()
        self.ensure_database()
    
    def get_db_connection(self):
        """Get PostgreSQL database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            raise
    
    def ensure_database(self):
        """Ensure the database exists and has proper structure"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create a table to track uploaded datasets if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS revana_datasets (
                    id SERIAL PRIMARY KEY,
                    table_name VARCHAR(255) UNIQUE NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✅ PostgreSQL database initialized")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    async def process_uploaded_file(self, file: UploadFile, filename: str):
        temp_file_path = None
        try:
            logger.info(f"📥 Processing uploaded file: {filename}")
            
            # Save uploaded file to temporary location
            temp_file_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Read CSV file
            df = pd.read_csv(temp_file_path)
            logger.info(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Generate table name
            table_name = self.generate_table_name(filename)
            
            # Create table in database
            result = self.create_table_from_dataframe(df, table_name, filename)
            
            # DEBUG: Show what we're processing
            print(f"🔍 FILE PROCESSOR - Starting vector processing")
            print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
            
            # Process data for vector embeddings
            print("🔄 Starting vector processing...")
            self.vector_processor.process_uploaded_data(df, table_name)
            print("✅ Vector processing completed")
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            logger.info(f"✅ Successfully created table '{table_name}' with {len(df)} records")
            
            # FIX: Return proper success response
            return {
                "success": True,
                "table_name": table_name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "database": "PostgreSQL"
            }
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            logger.error(f"❌ File processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_table_name(self, filename):
        """Generate PostgreSQL-compatible table name from filename"""
        base_name = os.path.splitext(filename)[0]
        # Clean the name for PostgreSQL
        clean_name = "".join(c if c.isalnum() else "_" for c in base_name)
        clean_name = "_".join(filter(None, clean_name.split("_")))
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_name = f"revana_{clean_name}_{timestamp}"
        
        return table_name.lower()
    
    def create_table_from_dataframe(self, df, table_name, original_filename):
        """Create PostgreSQL table from DataFrame"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check if table exists
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = %s
            """, (table_name,))
            
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                logger.info(f"🔄 Table {table_name} exists, replacing...")
                cursor.execute(f'DROP TABLE "{table_name}"')
            
            # Create column definitions with proper PostgreSQL types
            columns = []
            for col in df.columns:
                col_clean = self.clean_column_name(col)
                sql_type = self.map_pandas_to_postgres_type(df[col].dtype)
                columns.append(f'"{col_clean}" {sql_type}')

            # Create table SQL
            columns_str = ",\n    ".join(columns)
            create_sql = f'CREATE TABLE "{table_name}" (\n    {columns_str}\n)'
            cursor.execute(create_sql)
            
            # Insert data using execute_values for better performance
            columns_list = [self.clean_column_name(col) for col in df.columns]
            insert_sql = f'''
                INSERT INTO "{table_name}" ({", ".join([f'"{col}"' for col in columns_list])}) 
                VALUES %s
            '''
            
            # Convert DataFrame to list of tuples
            data_tuples = [tuple(x) for x in df.to_numpy()]
            execute_values(cursor, insert_sql, data_tuples)
            
            # Record the dataset in our tracking table
            cursor.execute("""
                INSERT INTO revana_datasets (table_name, original_filename, row_count, column_count)
                VALUES (%s, %s, %s, %s)
            """, (table_name, original_filename, len(df), len(df.columns)))
            
            # Commit transaction
            conn.commit()
            
            logger.info(f"✅ PostgreSQL table '{table_name}' created successfully with {len(df)} rows")
            
            return {
                "table_name": table_name,
                "rows_inserted": len(df),
                "columns": list(df.columns)
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Table creation failed: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def clean_column_name(self, col_name):
        """Clean column name for PostgreSQL compatibility"""
        clean = "".join(c if c.isalnum() else "_" for c in str(col_name))
        clean = "_".join(filter(None, clean.split("_")))
        return clean.lower()
    
    def map_pandas_to_postgres_type(self, pandas_dtype):
        """Map pandas data types to PostgreSQL types"""
        dtype_str = str(pandas_dtype)
        
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'DOUBLE PRECISION'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        elif 'datetime' in dtype_str:
            return 'TIMESTAMP'
        else:
            return 'TEXT'
    
    def list_tables(self):
        """List all Revana tables in PostgreSQL"""
        conn = self.get_db_connection()
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
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def get_table_info(self, table_name):
        """Get information about a specific table"""
        try:
            from backend.utils.database import db_manager
            
            # Get row count
            row_count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            row_result = db_manager.execute_query(row_count_query)
            row_count = row_result.iloc[0]['count'] if not row_result.empty else 0
            
            # Get column count and names
            column_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND table_schema = 'public'
            """
            column_result = db_manager.execute_query(column_query, (table_name,))
            column_count = len(column_result) if not column_result.empty else 0
            columns = column_result['column_name'].tolist() if not column_result.empty else []
            
            return {
                'table_name': table_name,
                'row_count': row_count,
                'column_count': column_count,
                'columns': columns
            }
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return None