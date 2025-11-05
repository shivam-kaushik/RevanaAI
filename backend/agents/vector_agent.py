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
                # Extract category filter if present
                category = self.extract_category_filter(user_query)
                # Extract city filter if present
                city = self.extract_city_filter(user_query)
                if city:
                    print(f"📍 City filter detected: {city}")
                results = self.vector_store.semantic_search_products(user_query, limit=10, category_filter=category, city_filter=city)
                print(f"📦 Found {len(results)} similar products")
                return self.format_product_results(results, user_query)
            
            elif query_type == "customer_search":
                print("👥 Performing customer semantic search...")
                # Use dataset_id if available
                dataset_id = getattr(Config, 'ACTIVE_DATASET_ID', None) or 'default'
                results = self.vector_store.semantic_search_customers(user_query, limit=10, dataset_id=dataset_id)
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
    
    def extract_city_filter(self, query: str) -> str:
        """Extract city/location from query"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    Extract the city or location name from the query. Common cities: 
                    Chicago, New York, Los Angeles, Phoenix, Houston, etc.
                    Return only the city name or "None" if no city found.
                    """},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            city = response.choices[0].message.content.strip()
            return None if city.lower() == "none" else city
            
        except Exception as e:
            logger.error(f"City extraction error: {e}")
            return None
    
    def format_product_results(self, results: list, original_query: str) -> str:
        """Format product search results with rich details"""
        if not results:
            return f"🔍 No similar products found for '{original_query}'"
        
        response = f"🔍 **Top {len(results)} Products for '{original_query}':**\n\n"
        
        for i, product in enumerate(results, 1):
            response += f"**{i}. {product.get('product_name', 'Unknown')}**\n"
            
            if product.get('product_category') and product.get('product_category') != 'N/A':
                response += f"   📁 Category: {product['product_category']}\n"
            
            response += f"   📊 Similarity: {product.get('similarity', 0):.1f}%\n"
            
            if product.get('description') and product.get('description') != 'N/A':
                desc = product['description']
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                response += f"   📝 {desc}\n"
            
            # Show metadata insights if available
            if product.get('metadata') and isinstance(product['metadata'], dict):
                meta = product['metadata']
                if meta.get('most_popular_city'):
                    response += f"   📍 Most Popular: {meta['most_popular_city']}\n"
                if meta.get('purchase_count'):
                    response += f"   🛒 Purchased {meta['purchase_count']} times\n"
                if meta.get('total_revenue'):
                    response += f"   💰 Revenue: ${float(meta['total_revenue']):.2f}\n"
                if meta.get('total_quantity_sold'):
                    response += f"   � Total Sold: {meta['total_quantity_sold']} units\n"
            
            response += "\n"
        
        response += f"*Semantic search powered by pgvector & OpenAI embeddings*"
        return response
    
    def format_customer_results(self, results: list, original_query: str) -> str:
        """Format customer search results with rich details"""
        if not results:
            return f"👥 No similar customers found for '{original_query}'"
        
        response = f"👥 **Top {len(results)} Similar Customers for '{original_query}':**\n\n"
        
        for i, customer in enumerate(results, 1):
            response += f"**{i}. {customer.get('customer_id', 'Unknown')}**\n"
            response += f"   🎯 Similarity: {customer.get('similarity', 0):.1f}%\n"
            
            if customer.get('preferences') and customer.get('preferences') != 'N/A':
                response += f"   ❤️ Preferences: {customer['preferences']}\n"
            
            if customer.get('purchase_history') and customer.get('purchase_history') != 'N/A':
                history = customer['purchase_history']
                if len(history) > 150:
                    history = history[:150] + "..."
                response += f"   📜 {history}\n"
            
            # Show metadata insights if available
            if customer.get('metadata') and isinstance(customer['metadata'], dict):
                meta = customer['metadata']
                if meta.get('transaction_count'):
                    response += f"   🛍️ Transactions: {meta['transaction_count']}\n"
                if meta.get('total_spent'):
                    try:
                        spent = float(meta['total_spent'])
                        response += f"   💵 Total Spent: ${spent:.2f}\n"
                    except:
                        pass
                if meta.get('primary_city'):
                    response += f"   📍 Location: {meta['primary_city']}\n"
            
            response += "\n"
        
        response += f"*Semantic search powered by pgvector & OpenAI embeddings*"
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