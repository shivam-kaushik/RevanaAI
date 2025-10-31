"""
Interactive Vector Chatbot Test

Run this script to test different query types in your vector database.
"""

from backend.agents.vector_agent import VectorAgent
from backend.utils.vector_store import PostgresVectorStore
from backend.config import Config

def show_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def test_vector_system():
    """Interactive test of vector system"""
    
    # Initialize
    agent = VectorAgent()
    vs = PostgresVectorStore()
    
    # Show stats
    show_header("📊 VECTOR DATABASE STATUS")
    stats = vs.get_vector_stats()
    print(f"✅ Vector Extension Available: {stats.get('vector_available', False)}")
    print(f"📦 Total Products: {stats.get('products_count', 0)}")
    print(f"👥 Total Customers: {stats.get('customers_count', 0)}")
    
    # Test queries
    show_header("🧪 EXAMPLE PRODUCT SEARCHES")
    
    product_queries = [
        ("Show me bakery products", "Searches for bakery items"),
        ("Find dairy products", "Searches for dairy items"),
        ("Products similar to Bread", "Finds products similar to Bread"),
    ]
    
    for query, description in product_queries:
        print(f"\n💬 Query: \"{query}\"")
        print(f"📝 Description: {description}")
        print("-" * 70)
        
        # Show classification
        query_type = agent.classify_query_type(query)
        print(f"🎯 Classified as: {query_type}\n")
        
        # Get results
        result = agent.handle_semantic_query(query)
        print(result)
        print()
    
    show_header("🧪 EXAMPLE CUSTOMER SEARCHES")
    
    customer_queries = [
        ("Find customers similar to CUST010", "Finds similar buying patterns to Customer_10"),
        ("Show me customers who buy bread", "Searches for customers by product preference"),
        ("High-value customers", "Finds customers by spending behavior"),
    ]
    
    for query, description in customer_queries:
        print(f"\n💬 Query: \"{query}\"")
        print(f"📝 Description: {description}")
        print("-" * 70)
        
        # Show classification
        query_type = agent.classify_query_type(query)
        print(f"🎯 Classified as: {query_type}\n")
        
        # Get results
        result = agent.handle_semantic_query(query)
        print(result)
        print()
    
    # Interactive mode
    show_header("💬 INTERACTIVE MODE")
    print("Try your own queries! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("🤔 Your query: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        print()
        print("-" * 70)
        
        # Classify
        query_type = agent.classify_query_type(user_input)
        print(f"🎯 Query classified as: {query_type}")
        print()
        
        # Get results
        result = agent.handle_semantic_query(user_input)
        print(result)
        print()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  🚀 VECTOR CHATBOT INTERACTIVE TEST")
    print("="*70)
    print("\nThis tool demonstrates how the vector database handles different")
    print("types of queries and routes them to appropriate embeddings.\n")
    
    try:
        test_vector_system()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ Error: {e}\n")
