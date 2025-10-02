# check_embeddings.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from utils.vector_store import PostgresVectorStore
import psycopg2

def check_embeddings():
    print("📊 Checking Vector Embeddings in Database")
    print("=" * 50)
    
    vector_store = PostgresVectorStore()
    
    if not vector_store.vector_available:
        print("❌ Vector store not available")
        return
    
    # Check what's in the product_embeddings table
    conn = vector_store.get_connection()
    cursor = conn.cursor()
    
    try:
        # Count embeddings
        cursor.execute("SELECT COUNT(*) FROM product_embeddings")
        product_count = cursor.fetchone()[0]
        print(f"📦 Product embeddings: {product_count}")
        
        cursor.execute("SELECT COUNT(*) FROM customer_embeddings")
        customer_count = cursor.fetchone()[0]
        print(f"👤 Customer embeddings: {customer_count}")
        
        # Show sample product embeddings
        if product_count > 0:
            print(f"\n🔍 Sample Product Embeddings:")
            cursor.execute("""
                SELECT product_name, product_category, description 
                FROM product_embeddings 
                LIMIT 5
            """)
            for row in cursor.fetchall():
                print(f"  • {row[0]} ({row[1]})")
                print(f"    Desc: {row[2][:100]}...")
                print()
        
        # Show sample customer embeddings
        if customer_count > 0:
            print(f"🔍 Sample Customer Embeddings:")
            cursor.execute("""
                SELECT customer_id, preferences, purchase_history
                FROM customer_embeddings 
                LIMIT 3
            """)
            for row in cursor.fetchall():
                print(f"  • Customer {row[0]}")
                print(f"    Prefs: {row[1]}")
                print(f"    History: {row[2][:100]}...")
                print()
                
    except Exception as e:
        print(f"❌ Error checking embeddings: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    check_embeddings()