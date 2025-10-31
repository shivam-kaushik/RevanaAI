"""
Flexible Embedding Generator for Any Dataset Schema

This module automatically detects dataset schema and generates appropriate embeddings
for different entity types (customers, products, transactions) regardless of column structure.
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from backend.config import Config
from backend.utils.database import db_manager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Standardized embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIM = 1536


class FlexibleEmbeddingGenerator:
    """Generate embeddings dynamically based on dataset schema"""
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.dataset_id = getattr(Config, 'ACTIVE_DATASET_ID', None) or 'default'
    
    def analyze_schema(self, table_name: str) -> Dict[str, List[str]]:
        """
        Analyze table schema and categorize columns by type.
        Returns dictionary of column types and their column names.
        """
        try:
            query = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            df = db_manager.execute_query(query)
            
            if df is None or df.empty:
                logger.error(f"Could not analyze schema for table: {table_name}")
                return {}
            
            # Categorize columns
            schema_info = {
                'id_columns': [],
                'customer_columns': [],
                'product_columns': [],
                'transaction_columns': [],
                'demographic_columns': [],
                'location_columns': [],
                'temporal_columns': [],
                'numeric_columns': [],
                'categorical_columns': []
            }
            
            for _, row in df.iterrows():
                col_name = row['column_name'].lower()
                col_type = row['data_type'].lower()
                
                # Identify column types based on name patterns
                if any(x in col_name for x in ['id', '_id']):
                    schema_info['id_columns'].append(row['column_name'])
                
                if any(x in col_name for x in ['customer', 'client', 'user', 'buyer']):
                    schema_info['customer_columns'].append(row['column_name'])
                
                if any(x in col_name for x in ['product', 'item', 'sku', 'goods']):
                    schema_info['product_columns'].append(row['column_name'])
                
                if any(x in col_name for x in ['transaction', 'order', 'purchase', 'sale']):
                    schema_info['transaction_columns'].append(row['column_name'])
                
                if any(x in col_name for x in ['age', 'gender', 'sex', 'occupation', 'income']):
                    schema_info['demographic_columns'].append(row['column_name'])
                
                if any(x in col_name for x in ['city', 'state', 'country', 'location', 'address', 'region']):
                    schema_info['location_columns'].append(row['column_name'])
                
                if any(x in col_name for x in ['date', 'time', 'timestamp', 'season', 'month', 'year']):
                    schema_info['temporal_columns'].append(row['column_name'])
                
                if 'numeric' in col_type or 'integer' in col_type or 'decimal' in col_type or 'float' in col_type:
                    if any(x in col_name for x in ['quantity', 'amount', 'price', 'cost', 'total', 'count']):
                        schema_info['numeric_columns'].append(row['column_name'])
                
                if col_type in ['character varying', 'text', 'varchar']:
                    if col_name not in schema_info['id_columns']:
                        schema_info['categorical_columns'].append(row['column_name'])
            
            logger.info(f"Schema analysis for {table_name}: {schema_info}")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error analyzing schema: {e}")
            return {}
    
    def detect_entity_types(self, schema_info: Dict[str, List[str]]) -> List[str]:
        """
        Determine what types of entities can be extracted from this dataset.
        Returns list of entity types: 'customer', 'product', 'transaction', 'location'
        """
        entity_types = []
        
        # Check for customer entities
        if schema_info.get('customer_columns') or schema_info.get('demographic_columns'):
            entity_types.append('customer')
        
        # Check for product entities
        if schema_info.get('product_columns'):
            entity_types.append('product')
        
        # Check for transaction/behavioral patterns
        if schema_info.get('transaction_columns') or schema_info.get('temporal_columns'):
            entity_types.append('transaction')
        
        # Check for location-based entities
        if schema_info.get('location_columns'):
            entity_types.append('location')
        
        logger.info(f"Detected entity types: {entity_types}")
        return entity_types
    
    def get_customer_identifier(self, schema_info: Dict[str, List[str]]) -> Optional[str]:
        """Find the primary customer identifier column"""
        # Priority order for customer identification
        priority_patterns = [
            'customer_id', 'customer_name', 'customerid', 'cust_id', 'custid',
            'client_id', 'user_id', 'buyer_id'
        ]
        
        all_columns = schema_info.get('customer_columns', []) + schema_info.get('id_columns', [])
        
        for pattern in priority_patterns:
            for col in all_columns:
                if pattern in col.lower():
                    logger.info(f"Using customer identifier: {col}")
                    return col
        
        # Fallback: use first customer column
        if schema_info.get('customer_columns'):
            col = schema_info['customer_columns'][0]
            logger.info(f"Using fallback customer identifier: {col}")
            return col
        
        return None
    
    def get_product_identifier(self, schema_info: Dict[str, List[str]]) -> Optional[str]:
        """Find the primary product identifier column"""
        priority_patterns = [
            'product_name', 'product_category', 'product_id', 'productname',
            'item_name', 'product_item', 'sku'
        ]
        
        all_columns = schema_info.get('product_columns', []) + schema_info.get('categorical_columns', [])
        
        for pattern in priority_patterns:
            for col in all_columns:
                if pattern in col.lower():
                    logger.info(f"Using product identifier: {col}")
                    return col
        
        # Fallback
        if schema_info.get('product_columns'):
            col = schema_info['product_columns'][0]
            logger.info(f"Using fallback product identifier: {col}")
            return col
        
        return None
    
    def build_customer_aggregation_query(self, table_name: str, schema_info: Dict[str, List[str]]) -> Optional[str]:
        """
        Build a SQL query to aggregate customer data from the table.
        Creates a rich text representation for each customer.
        """
        customer_col = self.get_customer_identifier(schema_info)
        if not customer_col:
            logger.warning("No customer identifier found")
            return None
        
        # Build select clause dynamically
        select_parts = [f"{customer_col} AS customer_id"]
        
        # Add demographic info if available
        for col in schema_info.get('demographic_columns', []):
            if 'age' in col.lower():
                select_parts.append(f"AVG({col})::text AS avg_age")
            elif 'gender' in col.lower():
                select_parts.append(f"MODE() WITHIN GROUP (ORDER BY {col}) AS gender")
        
        # Add product preferences
        product_col = self.get_product_identifier(schema_info)
        if product_col:
            select_parts.append(f"string_agg(DISTINCT {product_col}, ', ') AS product_preferences")
        
        # Add location info
        for col in schema_info.get('location_columns', []):
            if 'city' in col.lower():
                select_parts.append(f"MODE() WITHIN GROUP (ORDER BY {col}) AS primary_city")
            elif 'store_type' in col.lower():
                select_parts.append(f"string_agg(DISTINCT {col}, ', ') AS store_types")
        
        # Add transaction metrics
        select_parts.append("COUNT(*) AS transaction_count")
        
        # Add monetary values if available
        for col in schema_info.get('numeric_columns', []):
            if any(x in col.lower() for x in ['total', 'amount', 'cost', 'price']):
                select_parts.append(f"SUM({col})::text AS total_spent")
                select_parts.append(f"AVG({col})::text AS avg_transaction_value")
                break
        
        # Add temporal patterns if available
        for col in schema_info.get('temporal_columns', []):
            if 'season' in col.lower():
                select_parts.append(f"string_agg(DISTINCT {col}, ', ') AS seasons")
                break
        
        # Add payment methods if available
        for col in schema_info.get('categorical_columns', []):
            if 'payment' in col.lower():
                select_parts.append(f"string_agg(DISTINCT {col}, ', ') AS payment_methods")
                break
        
        query = f"""
        SELECT {', '.join(select_parts)}
        FROM {table_name}
        GROUP BY {customer_col}
        HAVING COUNT(*) >= 1
        ORDER BY COUNT(*) DESC
        """
        
        logger.info(f"Built customer aggregation query:\n{query}")
        return query
    
    def build_product_aggregation_query(self, table_name: str, schema_info: Dict[str, List[str]]) -> Optional[str]:
        """
        Build a SQL query to aggregate product data from the table.
        Creates a rich text representation for each product/category.
        """
        product_col = self.get_product_identifier(schema_info)
        if not product_col:
            logger.warning("No product identifier found")
            return None
        
        select_parts = [f"{product_col} AS product_name"]
        
        # Add category if different from product name
        category_col = None
        for col in schema_info.get('product_columns', []):
            if 'category' in col.lower() and col != product_col:
                select_parts.append(f"{col} AS product_category")
                category_col = col
                break
        
        # Add purchase metrics
        select_parts.append("COUNT(*) AS purchase_count")
        
        # Add quantity if available
        for col in schema_info.get('numeric_columns', []):
            if 'quantity' in col.lower():
                select_parts.append(f"SUM({col})::text AS total_quantity_sold")
                break
        
        # Add revenue metrics
        for col in schema_info.get('numeric_columns', []):
            if any(x in col.lower() for x in ['total', 'amount', 'cost']):
                select_parts.append(f"SUM({col})::text AS total_revenue")
                select_parts.append(f"AVG({col})::text AS avg_price")
                break
        
        # Add customer demographics who buy this product
        for col in schema_info.get('demographic_columns', []):
            if 'age' in col.lower():
                select_parts.append(f"AVG({col})::text AS avg_customer_age")
            elif 'gender' in col.lower():
                select_parts.append(f"MODE() WITHIN GROUP (ORDER BY {col}) AS primary_gender")
        
        # Add location popularity
        for col in schema_info.get('location_columns', []):
            if 'city' in col.lower():
                select_parts.append(f"MODE() WITHIN GROUP (ORDER BY {col}) AS most_popular_city")
                break
        
        # Add seasonal patterns
        for col in schema_info.get('temporal_columns', []):
            if 'season' in col.lower():
                select_parts.append(f"MODE() WITHIN GROUP (ORDER BY {col}) AS popular_season")
                break
        
        # Build GROUP BY clause - include product_name and category if present
        group_by_cols = [product_col]
        if category_col:
            group_by_cols.append(category_col)
        
        query = f"""
        SELECT {', '.join(select_parts)}
        FROM {table_name}
        GROUP BY {', '.join(group_by_cols)}
        HAVING COUNT(*) >= 1
        ORDER BY COUNT(*) DESC
        """
        
        logger.info(f"Built product aggregation query:\n{query}")
        return query
    
    def create_customer_text_representation(self, row: pd.Series, schema_info: Dict[str, List[str]]) -> str:
        """
        Create rich text representation for a customer based on available data.
        """
        parts = []
        
        # Start with customer ID
        if 'customer_id' in row:
            parts.append(f"Customer: {row['customer_id']}")
        
        # Add demographics
        if 'avg_age' in row and pd.notna(row['avg_age']):
            parts.append(f"Age: {float(row['avg_age']):.0f}")
        
        if 'gender' in row and pd.notna(row['gender']):
            parts.append(f"Gender: {row['gender']}")
        
        # Add purchase behavior
        if 'transaction_count' in row:
            parts.append(f"Transaction Count: {row['transaction_count']}")
        
        if 'total_spent' in row and pd.notna(row['total_spent']):
            parts.append(f"Total Spent: ${float(row['total_spent']):.2f}")
        
        if 'avg_transaction_value' in row and pd.notna(row['avg_transaction_value']):
            parts.append(f"Average Purchase: ${float(row['avg_transaction_value']):.2f}")
        
        # Add preferences
        if 'product_preferences' in row and pd.notna(row['product_preferences']):
            parts.append(f"Product Preferences: {row['product_preferences']}")
        
        # Add location
        if 'primary_city' in row and pd.notna(row['primary_city']):
            parts.append(f"Location: {row['primary_city']}")
        
        if 'store_types' in row and pd.notna(row['store_types']):
            parts.append(f"Store Types: {row['store_types']}")
        
        # Add temporal patterns
        if 'seasons' in row and pd.notna(row['seasons']):
            parts.append(f"Shopping Seasons: {row['seasons']}")
        
        # Add payment preferences
        if 'payment_methods' in row and pd.notna(row['payment_methods']):
            parts.append(f"Payment Methods: {row['payment_methods']}")
        
        text = " | ".join(parts)
        logger.debug(f"Customer text: {text[:200]}...")
        return text
    
    def create_product_text_representation(self, row: pd.Series, schema_info: Dict[str, List[str]]) -> str:
        """
        Create rich text representation for a product based on available data.
        """
        parts = []
        
        # Start with product name
        if 'product_name' in row:
            parts.append(f"Product: {row['product_name']}")
        
        if 'product_category' in row and pd.notna(row['product_category']):
            parts.append(f"Category: {row['product_category']}")
        
        # Add sales metrics
        if 'purchase_count' in row:
            parts.append(f"Times Purchased: {row['purchase_count']}")
        
        if 'total_quantity_sold' in row and pd.notna(row['total_quantity_sold']):
            parts.append(f"Total Units Sold: {float(row['total_quantity_sold']):.0f}")
        
        if 'total_revenue' in row and pd.notna(row['total_revenue']):
            parts.append(f"Total Revenue: ${float(row['total_revenue']):.2f}")
        
        if 'avg_price' in row and pd.notna(row['avg_price']):
            parts.append(f"Average Price: ${float(row['avg_price']):.2f}")
        
        # Add customer insights
        if 'avg_customer_age' in row and pd.notna(row['avg_customer_age']):
            parts.append(f"Avg Customer Age: {float(row['avg_customer_age']):.0f}")
        
        if 'primary_gender' in row and pd.notna(row['primary_gender']):
            parts.append(f"Primary Gender: {row['primary_gender']}")
        
        # Add location insights
        if 'most_popular_city' in row and pd.notna(row['most_popular_city']):
            parts.append(f"Popular in: {row['most_popular_city']}")
        
        # Add temporal patterns
        if 'popular_season' in row and pd.notna(row['popular_season']):
            parts.append(f"Popular Season: {row['popular_season']}")
        
        text = " | ".join(parts)
        logger.debug(f"Product text: {text[:200]}...")
        return text
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def upsert_customer_embedding(self, customer_id: str, embedding: List[float], 
                                   additional_data: Dict = None):
        """Upsert customer embedding into customer_embeddings table"""
        import json
        
        vector_literal = '[' + ','.join(str(float(x)) for x in embedding) + ']'
        
        sql = """
        INSERT INTO customer_embeddings 
            (customer_id, embedding, dataset_id, purchase_history, preferences, metadata)
        VALUES (%s, %s::vector, %s, %s, %s, %s::jsonb)
        ON CONFLICT (customer_id)
        DO UPDATE SET 
            embedding = EXCLUDED.embedding,
            dataset_id = EXCLUDED.dataset_id,
            purchase_history = EXCLUDED.purchase_history,
            preferences = EXCLUDED.preferences,
            metadata = EXCLUDED.metadata
        """
        
        purchase_history = additional_data.get('purchase_history', '') if additional_data else ''
        preferences = additional_data.get('preferences', '') if additional_data else ''
        metadata_dict = additional_data.get('metadata', {}) if additional_data else {}
        # Convert metadata dict to JSON string for psycopg2
        metadata = json.dumps(metadata_dict)
        
        conn = None
        cursor = None
        try:
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (customer_id, vector_literal, self.dataset_id, 
                                purchase_history, preferences, metadata))
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to upsert customer embedding for {customer_id}: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def upsert_product_embedding(self, product_name: str, embedding: List[float],
                                  additional_data: Dict = None):
        """Upsert product embedding into product_embeddings table"""
        import json
        
        vector_literal = '[' + ','.join(str(float(x)) for x in embedding) + ']'
        
        sql = """
        INSERT INTO product_embeddings 
            (product_name, embedding, product_category, description, metadata)
        VALUES (%s, %s::vector, %s, %s, %s::jsonb)
        ON CONFLICT (product_name)
        DO UPDATE SET 
            embedding = EXCLUDED.embedding,
            product_category = EXCLUDED.product_category,
            description = EXCLUDED.description,
            metadata = EXCLUDED.metadata
        """
        
        product_category = additional_data.get('product_category', '') if additional_data else ''
        description = additional_data.get('description', '') if additional_data else ''
        metadata_dict = additional_data.get('metadata', {}) if additional_data else {}
        # Convert metadata dict to JSON string for psycopg2
        metadata = json.dumps(metadata_dict)
        
        conn = None
        cursor = None
        try:
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            
            # First ensure product_name is unique constraint (add if needed)
            try:
                cursor.execute("""
                    ALTER TABLE product_embeddings 
                    DROP CONSTRAINT IF EXISTS product_embeddings_product_name_key
                """)
                cursor.execute("""
                    ALTER TABLE product_embeddings 
                    ADD CONSTRAINT product_embeddings_product_name_key 
                    UNIQUE (product_name)
                """)
                conn.commit()
            except Exception as constraint_error:
                logger.debug(f"Constraint modification info: {constraint_error}")
                conn.rollback()
            
            cursor.execute(sql, (product_name, vector_literal, product_category,
                                description, metadata))
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to upsert product embedding for {product_name}: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def populate_embeddings_from_table(self, table_name: str, 
                                       entity_types: List[str] = None) -> Dict[str, int]:
        """
        Main method: Analyze table, extract entities, and generate embeddings.
        
        Args:
            table_name: Name of the table to process
            entity_types: List of entity types to generate ('customer', 'product', etc.)
                         If None, auto-detects all possible types
        
        Returns:
            Dictionary with counts of generated embeddings per entity type
        """
        results = {'customers': 0, 'products': 0, 'errors': 0}
        
        try:
            # Step 1: Analyze schema
            logger.info(f"=== Analyzing schema for table: {table_name} ===")
            schema_info = self.analyze_schema(table_name)
            
            if not schema_info:
                logger.error("Could not analyze table schema")
                return results
            
            # Step 2: Detect entity types
            detected_types = self.detect_entity_types(schema_info)
            
            if entity_types is None:
                entity_types = detected_types
            else:
                # Use only requested types that are also detected
                entity_types = [t for t in entity_types if t in detected_types]
            
            logger.info(f"Will generate embeddings for: {entity_types}")
            
            # Step 3: Generate customer embeddings
            if 'customer' in entity_types:
                logger.info("\n=== Generating CUSTOMER embeddings ===")
                customer_query = self.build_customer_aggregation_query(table_name, schema_info)
                
                if customer_query:
                    customer_df = db_manager.execute_query(customer_query)
                    
                    if customer_df is not None and not customer_df.empty:
                        logger.info(f"Found {len(customer_df)} customers to embed")
                        
                        for idx, row in customer_df.iterrows():
                            try:
                                customer_id = str(row['customer_id'])
                                text = self.create_customer_text_representation(row, schema_info)
                                
                                embedding = self.get_embedding(text)
                                if embedding:
                                    # Prepare additional metadata
                                    additional_data = {
                                        'purchase_history': text[:500],
                                        'preferences': row.get('product_preferences', 'N/A'),
                                        'metadata': row.to_dict()
                                    }
                                    
                                    self.upsert_customer_embedding(customer_id, embedding, additional_data)
                                    results['customers'] += 1
                                    
                                    if (idx + 1) % 10 == 0:
                                        logger.info(f"  Processed {idx + 1}/{len(customer_df)} customers")
                                
                            except Exception as e:
                                logger.error(f"Error processing customer {row.get('customer_id')}: {e}")
                                results['errors'] += 1
                        
                        logger.info(f"✅ Completed {results['customers']} customer embeddings")
            
            # Step 4: Generate product embeddings
            if 'product' in entity_types:
                logger.info("\n=== Generating PRODUCT embeddings ===")
                product_query = self.build_product_aggregation_query(table_name, schema_info)
                
                if product_query:
                    product_df = db_manager.execute_query(product_query)
                    
                    if product_df is not None and not product_df.empty:
                        logger.info(f"Found {len(product_df)} products to embed")
                        
                        for idx, row in product_df.iterrows():
                            try:
                                product_name = str(row['product_name'])
                                text = self.create_product_text_representation(row, schema_info)
                                
                                embedding = self.get_embedding(text)
                                if embedding:
                                    # Prepare additional metadata
                                    additional_data = {
                                        'product_category': row.get('product_category', ''),
                                        'description': text[:500],
                                        'metadata': row.to_dict()
                                    }
                                    
                                    self.upsert_product_embedding(product_name, embedding, additional_data)
                                    results['products'] += 1
                                    
                                    if (idx + 1) % 10 == 0:
                                        logger.info(f"  Processed {idx + 1}/{len(product_df)} products")
                                
                            except Exception as e:
                                logger.error(f"Error processing product {row.get('product_name')}: {e}")
                                results['errors'] += 1
                        
                        logger.info(f"✅ Completed {results['products']} product embeddings")
            
            logger.info(f"\n=== SUMMARY ===")
            logger.info(f"Customer embeddings: {results['customers']}")
            logger.info(f"Product embeddings: {results['products']}")
            logger.info(f"Errors: {results['errors']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in populate_embeddings_from_table: {e}")
            results['errors'] += 1
            return results
