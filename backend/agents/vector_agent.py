import logging
from openai import OpenAI
from backend.config import Config
from backend.utils.vector_store import PostgresVectorStore

logger = logging.getLogger(__name__)

class VectorAgent:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.vector_store = PostgresVectorStore()
    
    def handle_semantic_query(self, user_query: str):
        """Handle natural language semantic queries"""
        try:
            print(f"\n🎯 SEMANTIC QUERY DETECTED: '{user_query}'")
            
            # Check if vector store is available
            if not self.vector_store.is_available():
                return "🔍 Semantic search is currently unavailable."
            
            print("✅ Vector store is available")
            
            # Use GPT to determine query type
            query_type = self.classify_query_type(user_query)
            print(f"📋 Query classified as: {query_type}")
            
            if query_type == "product_search":
                print("🔍 Performing product semantic search...")
                results = self.vector_store.semantic_search_products(user_query, limit=5)
                print(f"📦 Found {len(results)} similar products")
                return self.format_product_results(results, user_query)
            
            elif query_type == "customer_search":
                print("👥 Performing customer semantic search...")
                results = self.vector_store.semantic_search_customers(user_query, limit=5)
                print(f"👤 Found {len(results)} similar customers")
                return self.format_customer_results(results, user_query)
            
            elif query_type == "hybrid_search":
                print("🔀 Performing hybrid search...")
                category = self.extract_category_filter(user_query)
                print(f"🎯 Category filter: {category}")
                results = self.vector_store.hybrid_search(user_query, category, limit=5)
                print(f"📊 Found {len(results)} hybrid results")
                return self.format_hybrid_results(results, user_query)
            
            else:
                return "I can help you search for products or customers using semantic search."
                
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return "Sorry, I encountered an error during the semantic search."
    
    def classify_query_type(self, query: str) -> str:
        """Use GPT to classify the type of semantic query"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    Classify the user query into one of these categories:
                    - "product_search": Searching for products, items, things to buy, similar products
                    - "customer_search": Searching for customers, users, people, similar customers
                    - "hybrid_search": Combination with category filters
                    - "unknown": Can't determine
                    
                    Return only the category name.
                    """},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            return response.choices[0].message.content.strip().lower()
            
        except Exception as e:
            logger.error(f"Query classification error: {e}")
            return "unknown"
    
    def extract_category_filter(self, query: str) -> str:
        """Extract product category from query"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    Extract the product category from the query. Common categories: 
                    Electronics, Clothing, Beauty, etc.
                    Return only the category name or "None" if no category found.
                    """},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            category = response.choices[0].message.content.strip()
            return None if category.lower() == "none" else category
            
        except Exception as e:
            logger.error(f"Category extraction error: {e}")
            return None
    
    def format_product_results(self, results: list, original_query: str) -> str:
        """Format product search results"""
        if not results:
            return f"🔍 No similar products found for '{original_query}'"
        
        response = f"🔍 **Semantic Search Results for '{original_query}':**\n\n"
        
        for i, product in enumerate(results, 1):
            response += f"**{i}. {product['product_name']}**\n"
            response += f"   • Category: {product['product_category']}\n"
            response += f"   • Similarity: {product['similarity']:.2%}\n"
            if product.get('description'):
                response += f"   • Description: {product['description']}\n"
            response += "\n"
        
        response += f"*Found {len(results)} similar products using pgvector semantic search*"
        return response
    
    def format_customer_results(self, results: list, original_query: str) -> str:
        """Format customer search results"""
        if not results:
            return f"👥 No similar customers found for '{original_query}'"
        
        response = f"👥 **Semantic Search Results for '{original_query}':**\n\n"
        
        for i, customer in enumerate(results, 1):
            response += f"**{i}. Customer {customer['customer_id']}**\n"
            response += f"   • Preferences: {customer.get('preferences', 'N/A')}\n"
            response += f"   • Similarity: {customer['similarity']:.2%}\n"
            if customer.get('purchase_history'):
                response += f"   • History: {customer['purchase_history']}\n"
            response += "\n"
        
        response += f"*Found {len(results)} similar customers using pgvector semantic search*"
        return response
    
    def format_hybrid_results(self, results: list, original_query: str) -> str:
        """Format hybrid search results"""
        if not results:
            return f"🔍 No results found for '{original_query}'"
        
        response = f"🔍 **Hybrid Search Results for '{original_query}':**\n\n"
        
        for i, product in enumerate(results, 1):
            response += f"**{i}. {product['product_name']}**\n"
            response += f"   • Category: {product['product_category']}\n"
            response += f"   • Similarity: {product['similarity']:.2%}\n"
            if product.get('description'):
                response += f"   • Description: {product['description']}\n"
            response += "\n"
        
        response += f"*Found {len(results)} products using pgvector hybrid search*"
        return response