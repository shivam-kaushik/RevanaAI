# Populate customer_embeddings table with embeddings.
#
# Usage:
#   # activate your venv first
#   # Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1
#   python backend\utils\populate_customer_embeddings.py

import json
import logging
from openai import OpenAI
from backend.config import Config
from backend.utils.database import db_manager
from sqlalchemy import text
# Optionally import any helper used to build customer text (e.g., orders table)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EMBEDDING_MODEL = "text-embedding-3-small"  # or the model you use; adjust if different
VECTOR_DIM = 1536  # set to your model dimension

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def get_customers():
    """
    Adjust the query to get unique customer IDs and a text field to embed.
    Here we create a representative text per customer by concatenating product_category and total_amount etc.
    """
    # Use the canonical sample table created by database/init.py
    query = """
    SELECT customer_name AS customer_id,
           string_agg(distinct product_category, ', ') AS categories,
           SUM(total_cost)::text AS total_spent
    FROM retail_transactions
    GROUP BY customer_name
    """
    try:
        return db_manager.execute_query(query)
    except Exception as e:
        logger.error("Failed to query retail transactions: %s", e)
        logger.info("If you haven't initialized the sample DB, run: python database/init.py")
        return None

def embed_text(text):
    # Use your project's OpenAI client pattern
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    # OpenAI Python client returns embedding at resp.data[0].embedding or similar depending on SDK version
    # The modern openai.OpenAI client returns resp.data[0].embedding
    emb = resp.data[0].embedding
    return emb

def upsert_embedding(customer_id, vector):
    # Convert embedding list -> string for pgvector literal like: [0.1,0.2,0.3]
    vector_literal = '[' + ','.join(str(float(x)) for x in vector) + ']'

    sql = """
    INSERT INTO customer_embeddings (customer_id, embedding, dataset_id)
    VALUES (%s, %s::vector, %s)
    ON CONFLICT (customer_id)
        DO UPDATE SET embedding = EXCLUDED.embedding, dataset_id = EXCLUDED.dataset_id;
    """

    conn = None
    cursor = None
    try:
        # get dataset id from environment or default
        dataset_id = getattr(Config, 'ACTIVE_DATASET_ID', None) or 'default'
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, (customer_id, vector_literal, dataset_id))
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error("Upsert failed for %s: %s", customer_id, e)
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def main():
    # Ensure the customer_embeddings table has a dataset_id column (migration for older DBs)
    try:
        conn = db_manager.get_connection()
        cur = conn.cursor()
        cur.execute("ALTER TABLE customer_embeddings ADD COLUMN IF NOT EXISTS dataset_id TEXT;")
        conn.commit()
    except Exception as e:
        logger.debug("Could not ensure dataset_id column: %s", e)
        if conn:
            conn.rollback()
    finally:
        try:
            if cur:
                cur.close()
        except Exception:
            pass
        try:
            if conn:
                conn.close()
        except Exception:
            pass

    rows = get_customers()
    if rows is None or rows.empty:
        logger.info("No customers found to embed.")
        return

    logger.info("Found %d customers. Generating embeddings...", len(rows))
    for _, r in rows.iterrows():
        customer_id = r['customer_id']
        # Build a text representation (tweak as needed)
        text = f"Customer {customer_id} - categories: {r.get('categories','')}; total_spent: {r.get('total_spent','0')}"
        try:
            emb = embed_text(text)
            upsert_embedding(customer_id, emb)
            logger.info("Upserted embedding for %s", customer_id)
        except Exception as e:
            logger.exception("Failed for %s: %s", customer_id, e)

    logger.info("Done.")

if __name__ == "__main__":
    main()