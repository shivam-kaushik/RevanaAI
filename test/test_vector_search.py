import sys
import os

# Add the backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from utils.vector_store import PostgresVectorStore

def test_vector_search():
    print("🚀 Testing pgvector Semantic Search")
    print("=" * 50)
    
    # Initialize vector store
    vector_store = PostgresVectorStore()
    
    if not vector_store.vector_available:
        print("❌ pgvector is not available")
        return
    
    print("✅ pgvector is available!")
    
    # Test semantic searches
    test_queries = [
        "luxury skincare products",
        "electronics for professionals", 
        "trendy fashion items",
        "home office equipment",
        "gaming accessories"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        print("-" * 30)
        
        results = vector_store.semantic_search_products(query, limit=3)
        
        if results:
            for i, product in enumerate(results, 1):
                print(f"{i}. {product['product_name']}")
                print(f"   Category: {product['product_category']}")
                print(f"   Similarity: {product['similarity']:.2%}")
                print(f"   Description: {product.get('description', 'N/A')}")
                print()
        else:
            print("   No results found")
    
    print("\n📊 Vector Store Stats:")
    stats = vector_store.get_vector_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    test_vector_search()