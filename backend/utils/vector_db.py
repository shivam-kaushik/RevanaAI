import chromadb
from sentence_transformers import SentenceTransformer
import os
from backend.config import Config

class VectorDBManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.client.get_or_create_collection("sales_assistant_knowledge")
        
        # Initialize with sample data if empty
        if self.collection.count() == 0:
            self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize vector DB with schema information and examples"""
        
        # Schema information
        schema_docs = [
            "Table: retail_transactions - Contains retail sales transaction data",
            "Columns: date (DATE) - Transaction date, customer_name (VARCHAR) - Customer name, product_category (VARCHAR) - Product category, product_name (VARCHAR) - Product name, quantity (INTEGER) - Quantity sold, unit_price (DECIMAL) - Price per unit, total_cost (DECIMAL) - Total transaction amount, payment_method (VARCHAR) - Payment method used, city (VARCHAR) - City where transaction occurred, store_type (VARCHAR) - Type of store (Physical/Online), discount_applied (BOOLEAN) - Whether discount was applied",
            "Business terms: Revenue = sum of total_cost, Monthly sales = total_cost grouped by month, Top products = products with highest total_cost, Customer segmentation = grouping by customer behavior"
        ]
        
        # Example queries
        example_queries = [
            "What were the total sales last month?|SELECT SUM(total_cost) as total_sales FROM retail_transactions WHERE date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND date < DATE_TRUNC('month', CURRENT_DATE);",
            "Show me sales by product category for the last quarter|SELECT product_category, SUM(total_cost) as revenue FROM retail_transactions WHERE date >= DATE_TRUNC('quarter', CURRENT_DATE) - INTERVAL '3 months' AND date < DATE_TRUNC('quarter', CURRENT_DATE) GROUP BY product_category ORDER BY revenue DESC;",
            "What are the top 5 products by revenue?|SELECT product_name, SUM(total_cost) as revenue FROM retail_transactions GROUP BY product_name ORDER BY revenue DESC LIMIT 5;",
            "How have sales trended over the past 6 months?|SELECT DATE_TRUNC('month', date) as month, SUM(total_cost) as monthly_sales FROM retail_transactions WHERE date >= CURRENT_DATE - INTERVAL '6 months' GROUP BY month ORDER BY month;",
            "Which city has the highest sales?|SELECT city, SUM(total_cost) as total_sales FROM retail_transactions GROUP BY city ORDER BY total_sales DESC LIMIT 1;"
        ]
        
        documents = schema_docs + example_queries
        embeddings = self.embedding_model.encode(documents).tolist()
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
    
    def search_similar(self, query, n_results=3):
        """Search for similar documents in vector DB"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results['documents'][0] if results['documents'] else []

# Singleton instance
vector_db = VectorDBManager()