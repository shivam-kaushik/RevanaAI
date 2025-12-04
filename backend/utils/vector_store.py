import logging
import psycopg2
import numpy as np
import re
import ast
from typing import List, Dict, Any
import openai
from backend.config import Config

logger = logging.getLogger(__name__)

class PostgresVectorStore:
    def __init__(self):
        self.db_config = {
            'dbname': 'Revana',
            'user': 'postgres',
            'password': 'Adilet123!',  # Update with your actual password
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
            
            # Ensure customer embeddings table exists as well
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customer_embeddings (
                    customer_id TEXT PRIMARY KEY,
                    purchase_history TEXT,
                    preferences TEXT,
                    metadata JSONB,
                    dataset_id TEXT,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create optional ivfflat indexes for speed (may require admin permissions)
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_embeddings_embedding ON product_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            except Exception:
                logger.debug("Could not create ivfflat index for product_embeddings; skipping")

            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_embeddings_embedding ON customer_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            except Exception:
                logger.debug("Could not create ivfflat index for customer_embeddings; skipping")

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
        """Get OpenAI embedding for text using standardized model"""
        try:
            # Use text-embedding-3-small for consistency across the project
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
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
    
    def semantic_search_products(self, query: str, limit: int = 5, category_filter: str = None, city_filter: str = None) -> List[Dict]:
        """Find similar products using semantic search with optional category and city filters"""
        if not self.vector_available:
            logger.warning("⚠️ Vector store not available - semantic search disabled")
            return []
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                return []
            
            # Convert to PostgreSQL array format with full precision
            query_embedding = [float(x) for x in query_embedding]
            query_embedding_array = "[" + ",".join(format(float(x), '.18g') for x in query_embedding) + "]"
            
            # Inspect available columns
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", ('product_embeddings',))
            available_cols = [r[0] for r in cursor.fetchall()]
            
            if 'embedding' not in available_cols:
                logger.error("product_embeddings table does not have an 'embedding' column")
                return []
            
            # Build dynamic SELECT clause
            select_cols = []
            for c in ('product_name', 'product_category', 'description', 'metadata'):
                if c in available_cols:
                    select_cols.append(c)
            
            select_clause = ', '.join(select_cols) if select_cols else ''
            if select_clause:
                select_clause = select_clause + ', '
            select_clause = select_clause + "GREATEST(0, 1 - (embedding <=> %s::vector)) as similarity"
            
            # Build WHERE clause for category and city filters
            where_clauses = []
            params = []
            
            if category_filter and 'product_category' in available_cols:
                where_clauses.append("product_category ILIKE %s")
                params.append(f"%{category_filter}%")
            
            # City filter: check metadata JSONB field for most_popular_city
            if city_filter and 'metadata' in available_cols:
                where_clauses.append("(metadata->>'most_popular_city')::text ILIKE %s")
                params.append(f"%{city_filter}%")
            
            where_clause = ""
            if where_clauses:
                where_clause = "WHERE " + " AND ".join(where_clauses)
            
            sql = f"""
                SELECT {select_clause}
                FROM product_embeddings
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            
            # Build parameters: [vector_for_SELECT, category_filter (optional), vector_for_ORDER, limit]
            exec_params = [query_embedding_array] + params + [query_embedding_array, limit]
            cursor.execute(sql, exec_params)
            
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = {}
                for idx, col in enumerate(select_cols):
                    item[col] = row[idx] if row[idx] is not None else 'N/A'
                # Similarity is last column, convert to percentage
                sim = float(row[len(select_cols)])
                item['similarity'] = round(sim * 100.0, 2)
                results.append(item)
            
            # Sort by similarity descending
            results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic product search: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def semantic_search_customers(self, query: str, limit: int = 5, dataset_id: str = None) -> List[Dict]:
        """Find similar customers using semantic search"""
        if not self.vector_available:
            logger.warning("⚠️ Vector store not available - semantic search disabled")
            return []
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # If the user asked for customers similar to a specific customer id like 'CUST010',
            # normalize it to the repo's Customer_<n> ids and, if present, use that customer's
            # stored embedding as the query vector (more accurate than re-embedding the id string).
            query_embedding = None
            cust_match = re.search(r"\bCUST0*(\d+)\b", query, re.IGNORECASE)
            if cust_match:
                cust_num = int(cust_match.group(1))
                cust_id = f"Customer_{cust_num}"
                try:
                    cursor.execute("SELECT embedding FROM customer_embeddings WHERE customer_id = %s", (cust_id,))
                    row = cursor.fetchone()
                    if row and row[0]:
                        # row[0] may come back as a Python list, tuple, numpy array, memoryview or string
                        raw = row[0]
                        # If it's already a list/tuple/ndarray, convert to list of floats
                        if isinstance(raw, (list, tuple, np.ndarray)):
                            query_embedding = [float(x) for x in raw]
                        elif isinstance(raw, memoryview):
                            try:
                                s = raw.tobytes().decode('utf-8')
                            except Exception:
                                s = str(raw)
                            try:
                                query_embedding = [float(x) for x in ast.literal_eval(s)]
                            except Exception:
                                nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s)
                                query_embedding = [float(n) for n in nums]
                        elif isinstance(raw, str):
                            # Try to parse a Python list string like '[0.01, 0.23, ...]'
                            try:
                                query_embedding = [float(x) for x in ast.literal_eval(raw)]
                            except Exception:
                                nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", raw)
                                query_embedding = [float(n) for n in nums]
                        else:
                            # Fallback: try to coerce to list
                            try:
                                query_embedding = [float(x) for x in list(raw)]
                            except Exception:
                                logger.debug("Unable to coerce stored embedding for %s; will re-embed: %s", cust_id, type(raw))
                                query_embedding = None
                except Exception as e:
                    logger.debug("Could not fetch embedding for %s: %s", cust_id, e)

            # Fall back to generating an embedding for the query text
            if query_embedding is None:
                query_embedding = self.get_embedding(query)

            if not query_embedding:
                return []

            # Convert to PostgreSQL array format
            # Ensure we have a plain list of floats; format with full precision
            query_embedding = [float(x) for x in query_embedding]
            query_embedding_array = "[" + ",".join(format(float(x), '.18g') for x in query_embedding) + "]"

            # Inspect available columns in customer_embeddings to avoid missing column errors
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", ('customer_embeddings',))
            available_cols = [r[0] for r in cursor.fetchall()]

            # Ensure embedding column exists
            if 'embedding' not in available_cols:
                logger.error("customer_embeddings table does not have an 'embedding' column")
                return []

            select_cols = []
            # Prefer to include these if present
            for c in ('customer_id', 'purchase_history', 'preferences', 'metadata'):
                if c in available_cols:
                    select_cols.append(c)

            # Build select clause dynamically and append similarity
            select_clause = ', '.join(select_cols) if select_cols else ''
            if select_clause:
                select_clause = select_clause + ', '
            # Use GREATEST to clip negative similarities to zero (distance may be > 1)
            select_clause = select_clause + "GREATEST(0, 1 - (embedding <=> %s::vector)) as similarity"

            # Build WHERE clause for optional dataset scoping
            where_clause = ""
            params = []
            if dataset_id and 'dataset_id' in available_cols:
                where_clause = "WHERE dataset_id = %s"
                params.append(dataset_id)

            sql = f"""
                SELECT {select_clause}
                FROM customer_embeddings
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            # Build exec_params so that the first vector placeholder (in SELECT similarity)
            # receives the query vector, then dataset_id (if present), then the ORDER BY vector and limit.
            # SQL parameter order is:
            #   1) SELECT similarity vector placeholder
            #   2) WHERE dataset_id (optional)
            #   3) ORDER BY vector placeholder
            #   4) LIMIT
            exec_params = [query_embedding_array] + params + [query_embedding_array, limit]
            cursor.execute(sql, exec_params)

            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = {}
                # Map returned columns to dict keys; replace None with 'N/A' for readability
                for idx, col in enumerate(select_cols):
                    item[col] = row[idx] if row[idx] is not None else 'N/A'
                # similarity is always last and is in [0,1]
                sim = float(row[len(select_cols)])
                # convert to percentage for display
                item['similarity'] = round(sim * 100.0, 2)
                results.append(item)

            # Results are already ordered by distance (most similar first). Ensure sorting by similarity desc
            results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
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