import logging
import psycopg2
import numpy as np
from typing import List, Dict, Any
import openai
from backend.config import Config

logger = logging.getLogger(__name__)

class PostgresVectorStore:
    def __init__(self):
        self.db_config = {
            'dbname': 'Revana',
            'user': 'postgres',
            'password': 'Password1!',  # Update with your actual password
            'host': 'localhost',
            'port': '5432'
        }
        self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.vector_available = False
        self.initialize_vector_store()
    
    def initialize_vector_store(self):
        """Initialize vector store gracefully - don't crash if pgvector not available"""
        try:
            self.ensure_vector_tables()
            self.vector_available = True
            logger.info("✅ pgvector store initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ pgvector not available: {e}")
            logger.info("⚠️ Semantic search features will be disabled")
            self.vector_available = False
    
    def get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.db_config)
    
    def ensure_vector_tables(self):
        """Create tables for vector storage if they don't exist"""
        print("🔍 Vector Store Debug - Starting ensure_vector_tables")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Debug: Show current database info
            cursor.execute("SELECT current_database(), current_user")
            db_info = cursor.fetchone()
            print(f"🔍 Vector Store - Connected to: {db_info[0]} as {db_info[1]}")
            
            # List all extensions
            cursor.execute("SELECT extname FROM pg_extension ORDER BY extname")
            extensions = [row[0] for row in cursor.fetchall()]
            print(f"🔍 Vector Store - Available extensions: {extensions}")
            
            # Check if vector extension is available
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """)
            has_vector_extension = cursor.fetchone()[0]
            
            print(f"🔍 Vector Store - Has vector extension: {has_vector_extension}")
            
            if not has_vector_extension:
                logger.error("❌ pgvector extension not installed")
                # Try to create it
                try:
                    print("🔄 Attempting to create vector extension...")
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.commit()
                    print("✅ Vector extension creation attempted")
                    
                    # Check again
                    cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                    has_vector_extension = cursor.fetchone()[0]
                    print(f"🔍 Vector Store - After creation attempt: {has_vector_extension}")
                    
                except Exception as create_error:
                    print(f"❌ Failed to create vector extension: {create_error}")
                
                if not has_vector_extension:
                    raise Exception("pgvector extension not available. Please install it first.")
            
            logger.info("✅ pgvector extension is available")
            
            # Rest of your table creation code...
            print("🔄 Creating vector tables...")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS product_embeddings (
                    id SERIAL PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    product_category TEXT,
                    description TEXT,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # ... rest of your table creation code
            
            conn.commit()
            print("✅ Vector tables created successfully")
            
        except Exception as e:
            print(f"❌ Error in ensure_vector_tables: {e}")
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
    
    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def store_product_embeddings(self, products_data: List[Dict]):
        """Store product data with embeddings"""
        if not self.vector_available:
            logger.warning("⚠️ Vector store not available - skipping product embeddings")
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            for product in products_data:
                # Create descriptive text for embedding
                description_text = f"""
                Product: {product.get('product_name', '')}
                Category: {product.get('product_category', '')}
                Description: {product.get('description', '')}
                Price: {product.get('price', '')}
                """
                
                embedding = self.get_embedding(description_text)
                
                if embedding:
                    # Convert to PostgreSQL array format for vector type
                    embedding_array = "[" + ",".join(map(str, embedding)) + "]"
                    
                    cursor.execute("""
                        INSERT INTO product_embeddings 
                        (product_name, product_category, description, embedding, metadata)
                        VALUES (%s, %s, %s, %s::vector, %s)
                    """, (
                        product.get('product_name', ''),
                        product.get('product_category', ''),
                        product.get('description', ''),
                        embedding_array,
                        product
                    ))
            
            conn.commit()
            logger.info(f"✅ Stored embeddings for {len(products_data)} products")
            
        except Exception as e:
            logger.error(f"Error storing product embeddings: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def store_customer_embeddings(self, customers_data: List[Dict]):
        """Store customer data with embeddings"""
        if not self.vector_available:
            logger.warning("⚠️ Vector store not available - skipping customer embeddings")
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            for customer in customers_data:
                # Create descriptive text for embedding
                description_text = f"""
                Customer: {customer.get('customer_id', '')}
                Age: {customer.get('age', '')}
                Gender: {customer.get('gender', '')}
                Purchase History: {customer.get('purchase_history', '')}
                Preferences: {customer.get('preferences', '')}
                """
                
                embedding = self.get_embedding(description_text)
                
                if embedding:
                    # Convert to PostgreSQL array format for vector type
                    embedding_array = "[" + ",".join(map(str, embedding)) + "]"
                    
                    cursor.execute("""
                        INSERT INTO customer_embeddings 
                        (customer_id, purchase_history, preferences, embedding, metadata)
                        VALUES (%s, %s, %s, %s::vector, %s)
                    """, (
                        customer.get('customer_id', ''),
                        customer.get('purchase_history', ''),
                        customer.get('preferences', ''),
                        embedding_array,
                        customer
                    ))
            
            conn.commit()
            logger.info(f"✅ Stored embeddings for {len(customers_data)} customers")
            
        except Exception as e:
            logger.error(f"Error storing customer embeddings: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def semantic_search_products(self, query: str, limit: int = 5) -> List[Dict]:
        """Find similar products using semantic search"""
        if not self.vector_available:
            logger.warning("⚠️ Vector store not available - semantic search disabled")
            return []
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                return []
            
            # Convert to PostgreSQL array format
            query_embedding_array = "[" + ",".join(map(str, query_embedding)) + "]"
            
            cursor.execute("""
                SELECT 
                    product_name,
                    product_category,
                    description,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM product_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding_array, query_embedding_array, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'product_name': row[0],
                    'product_category': row[1],
                    'description': row[2],
                    'metadata': row[3],
                    'similarity': float(row[4])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic product search: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def semantic_search_customers(self, query: str, limit: int = 5) -> List[Dict]:
        """Find similar customers using semantic search"""
        if not self.vector_available:
            logger.warning("⚠️ Vector store not available - semantic search disabled")
            return []
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                return []
            
            # Convert to PostgreSQL array format
            query_embedding_array = "[" + ",".join(map(str, query_embedding)) + "]"
            
            cursor.execute("""
                SELECT 
                    customer_id,
                    purchase_history,
                    preferences,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM customer_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding_array, query_embedding_array, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'customer_id': row[0],
                    'purchase_history': row[1],
                    'preferences': row[2],
                    'metadata': row[3],
                    'similarity': float(row[4])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic customer search: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def hybrid_search(self, semantic_query: str, category_filter: str = None, limit: int = 5) -> List[Dict]:
        """Combine semantic search with structured filters"""
        if not self.vector_available:
            logger.warning("⚠️ Vector store not available - hybrid search disabled")
            return []
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query_embedding = self.get_embedding(semantic_query)
            
            if not query_embedding:
                return []
            
            # Convert to PostgreSQL array format
            query_embedding_array = "[" + ",".join(map(str, query_embedding)) + "]"
            
            if category_filter:
                cursor.execute("""
                    SELECT 
                        product_name,
                        product_category,
                        description,
                        metadata,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM product_embeddings
                    WHERE product_category = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding_array, category_filter, query_embedding_array, limit))
            else:
                cursor.execute("""
                    SELECT 
                        product_name,
                        product_category,
                        description,
                        metadata,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM product_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding_array, query_embedding_array, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'product_name': row[0],
                    'product_category': row[1],
                    'description': row[2],
                    'metadata': row[3],
                    'similarity': float(row[4])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def get_vector_stats(self):
        """Get statistics about vector data"""
        if not self.vector_available:
            return {"vector_available": False, "message": "pgvector extension not available"}
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM product_embeddings")
            product_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM customer_embeddings")
            customer_count = cursor.fetchone()[0]
            
            return {
                "vector_available": True,
                "products_count": product_count,
                "customers_count": customer_count,
                "vector_extension": "pgvector"
            }
            
        except Exception as e:
            logger.error(f"Error getting vector stats: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    def is_available(self):
        """Check if vector store is available"""
        return self.vector_available