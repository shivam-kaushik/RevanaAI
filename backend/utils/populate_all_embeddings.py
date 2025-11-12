"""
Universal Embedding Population Script

This script can generate embeddings for ANY dataset structure.
It automatically detects the schema and generates appropriate embeddings
for customers, products, and other entities.

Usage:
    # Activate venv first
    # Windows PowerShell:
    .venv/Scripts/Activate.ps1
    
    # Populate from a specific table
    python backend/utils/populate_all_embeddings.py --table retail_transactions
    
    # Populate specific entity types only
    python backend/utils/populate_all_embeddings.py --table retail_sales --entities customer product
    
    # Populate from all tables in database
    python backend/utils/populate_all_embeddings.py --all
"""

import argparse
import logging
import sys
from backend.utils.flexible_embeddings import FlexibleEmbeddingGenerator
from backend.utils.database import db_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_embedding_tables():
    """Ensure customer_embeddings and product_embeddings tables exist with proper schema"""
    conn = None
    cursor = None
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        # Ensure customer_embeddings table
        logger.info("Ensuring customer_embeddings table exists...")
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
        
        # Add missing columns if table already exists (migration)
        logger.info("Adding missing columns to customer_embeddings...")
        cursor.execute("""
            ALTER TABLE customer_embeddings 
            ADD COLUMN IF NOT EXISTS dataset_id TEXT
        """)
        cursor.execute("""
            ALTER TABLE customer_embeddings 
            ADD COLUMN IF NOT EXISTS purchase_history TEXT
        """)
        cursor.execute("""
            ALTER TABLE customer_embeddings 
            ADD COLUMN IF NOT EXISTS preferences TEXT
        """)
        cursor.execute("""
            ALTER TABLE customer_embeddings 
            ADD COLUMN IF NOT EXISTS metadata JSONB
        """)
        
        # Ensure product_embeddings table
        logger.info("Ensuring product_embeddings table exists...")
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
        
        # Add unique constraint on product_name if not exists
        cursor.execute("""
            ALTER TABLE product_embeddings 
            DROP CONSTRAINT IF EXISTS product_embeddings_product_name_key
        """)
        cursor.execute("""
            ALTER TABLE product_embeddings 
            ADD CONSTRAINT product_embeddings_product_name_key 
            UNIQUE (product_name)
        """)
        
        # Create indexes for performance
        logger.info("Creating indexes...")
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_customer_embeddings_embedding 
                ON customer_embeddings 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100)
            """)
        except Exception as idx_err:
            logger.debug(f"Index creation info: {idx_err}")
        
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_product_embeddings_embedding 
                ON product_embeddings 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100)
            """)
        except Exception as idx_err:
            logger.debug(f"Index creation info: {idx_err}")
        
        conn.commit()
        logger.info("✅ Embedding tables ready")
        
    except Exception as e:
        logger.error(f"Error ensuring embedding tables: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_available_tables():
    """Get list of user tables in the database (excluding system tables)"""
    try:
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        AND table_name NOT IN ('customer_embeddings', 'product_embeddings')
        ORDER BY table_name
        """
        df = db_manager.execute_query(query)
        
        if df is not None and not df.empty:
            return df['table_name'].tolist()
        return []
        
    except Exception as e:
        logger.error(f"Error getting table list: {e}")
        return []


def populate_table(table_name: str, entity_types: list = None):
    """Populate embeddings for a specific table"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing table: {table_name}")
    logger.info(f"{'='*60}\n")
    
    try:
        generator = FlexibleEmbeddingGenerator()
        results = generator.populate_embeddings_from_table(table_name, entity_types)
        
        logger.info(f"\n✅ Successfully processed {table_name}")
        logger.info(f"   - Customer embeddings: {results['customers']}")
        logger.info(f"   - Product embeddings: {results['products']}")
        logger.info(f"   - Errors: {results['errors']}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error processing table {table_name}: {e}")
        return {'customers': 0, 'products': 0, 'errors': 1}


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for any dataset structure'
    )
    parser.add_argument(
        '--table',
        type=str,
        help='Specific table name to process (e.g., retail_transactions)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all tables in the database'
    )
    parser.add_argument(
        '--entities',
        nargs='+',
        choices=['customer', 'product', 'transaction', 'location'],
        help='Specific entity types to generate (default: auto-detect all)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available tables and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # Step 1: Ensure embedding tables exist
        ensure_embedding_tables()
        
        # Step 2: Get available tables
        available_tables = get_available_tables()
        
        if args.list:
            logger.info("\n📋 Available tables:")
            for table in available_tables:
                logger.info(f"   - {table}")
            return
        
        if not available_tables:
            logger.error("❌ No tables found in database")
            return
        
        # Step 3: Determine which tables to process
        tables_to_process = []
        
        if args.all:
            tables_to_process = available_tables
            logger.info(f"Processing all {len(tables_to_process)} tables")
        elif args.table:
            if args.table in available_tables:
                tables_to_process = [args.table]
            else:
                logger.error(f"❌ Table '{args.table}' not found")
                logger.info(f"Available tables: {', '.join(available_tables)}")
                return
        else:
            # Interactive mode: ask user to select
            logger.info("\n📋 Available tables:")
            for i, table in enumerate(available_tables, 1):
                logger.info(f"   {i}. {table}")
            
            choice = input("\nEnter table number to process (or 'all'): ").strip()
            
            if choice.lower() == 'all':
                tables_to_process = available_tables
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_tables):
                        tables_to_process = [available_tables[idx]]
                    else:
                        logger.error("Invalid choice")
                        return
                except ValueError:
                    logger.error("Invalid input")
                    return
        
        # Step 4: Process each table
        total_results = {'customers': 0, 'products': 0, 'errors': 0}
        
        for table in tables_to_process:
            results = populate_table(table, args.entities)
            total_results['customers'] += results['customers']
            total_results['products'] += results['products']
            total_results['errors'] += results['errors']
        
        # Step 5: Final summary
        logger.info(f"\n{'='*60}")
        logger.info("🎉 FINAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Tables processed: {len(tables_to_process)}")
        logger.info(f"Total customer embeddings: {total_results['customers']}")
        logger.info(f"Total product embeddings: {total_results['products']}")
        logger.info(f"Total errors: {total_results['errors']}")
        logger.info(f"{'='*60}\n")
        
        # Verify final counts
        try:
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM customer_embeddings")
            cust_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM product_embeddings")
            prod_count = cursor.fetchone()[0]
            
            logger.info(f"📊 Database verification:")
            logger.info(f"   - customer_embeddings table: {cust_count} rows")
            logger.info(f"   - product_embeddings table: {prod_count} rows")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Could not verify counts: {e}")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
