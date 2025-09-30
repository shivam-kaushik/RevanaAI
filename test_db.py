import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import Config
from backend.utils.database import db_manager

def test_database():
    print("🔍 Testing PostgreSQL connection...")
    print(f"Database URL: {Config.DATABASE_URL}")
    
    if db_manager.test_connection():
        print("✅ PostgreSQL connection successful!")
        
        # Test creating a sample table
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS test_table (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100),
                        value INTEGER
                    )
                """)
                conn.commit()
                print("✅ Test table created successfully")
        except Exception as e:
            print(f"⚠ Test table creation: {e}")
    else:
        print("❌ PostgreSQL connection failed!")
        print("💡 Make sure:")
        print("1. PostgreSQL is running")
        print("2. Database 'revana' exists")
        print("3. Username/password in .env file are correct")

if __name__ == "__main__":
    test_database()