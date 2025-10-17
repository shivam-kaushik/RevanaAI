import psycopg2
import sys

def diagnose_pgvector():
    print("🔍 Starting pgvector Diagnosis...")
    
    # Test multiple connection scenarios
    test_cases = [
        {'dbname': 'Revana', 'description': 'Exact case match'},
        {'dbname': 'revana', 'description': 'Lowercase'},
        {'dbname': 'REVANA', 'description': 'Uppercase'},
        {'dbname': 'postgres', 'description': 'Default postgres database'}
    ]
    
    for test in test_cases:
        print(f"\n🧪 Testing: {test['description']} (dbname: {test['dbname']})")
        print("-" * 50)
        
        try:
            conn = psycopg2.connect(
                dbname=test['dbname'],
                user='postgres', 
                password='Password1!',  # Use your actual password
                host='localhost',
                port='5432'
            )
            cursor = conn.cursor()
            
            # Test basic connection
            cursor.execute("SELECT current_database(), current_user, version()")
            db_info = cursor.fetchone()
            print(f"✅ Connected to: {db_info[0]}")
            print(f"📊 User: {db_info[1]}")
            
            # Check extensions
            cursor.execute("SELECT extname FROM pg_extension ORDER BY extname")
            extensions = [row[0] for row in cursor.fetchall()]
            print(f"🔧 Extensions: {extensions}")
            
            # Check if vector exists
            vector_exists = 'vector' in extensions
            print(f"📦 Vector extension: {'✅ FOUND' if vector_exists else '❌ NOT FOUND'}")
            
            if vector_exists:
                # Test vector operations
                try:
                    cursor.execute("SELECT '[1,2,3]'::vector <=> '[4,5,6]'::vector as distance")
                    distance = cursor.fetchone()[0]
                    print(f"🎯 Vector operations: ✅ WORKING (distance: {distance})")
                except Exception as e:
                    print(f"🎯 Vector operations: ❌ FAILED - {e}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    diagnose_pgvector()