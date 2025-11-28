import logging
import re
from openai import OpenAI
from backend.config import Config, USE_QWEN_MODEL
from backend.utils.dataset_manager import DatasetManager
from backend.utils.database import db_manager
from backend.utils.vector_db import vector_db

logger = logging.getLogger(__name__)


class SQLAgent:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.dataset_manager = DatasetManager()

        # Try loading embedded Qwen Text2SQL
        try:
            from backend.agents.text2sql_qwen.model_loader import get_model
            from backend.agents.text2sql_qwen.prompting import build_messages
            self.use_qwen = bool(USE_QWEN_MODEL)
            if self.use_qwen:
                self.qwen_model = get_model()
                self.qwen_prompt_builder = build_messages
                logger.info("✅ Qwen Text2SQL model initialized successfully.")
            else:
                self.qwen_model = None
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Qwen model: {e}")
            self.qwen_model = None
            self.use_qwen = False

        logger.info(
            f"USE_QWEN_MODEL={getattr(self, 'use_qwen', False)} | qwen_model_is_none={self.qwen_model is None}")

    def generate_sql(self, user_query, active_table=None):
        """Generate SQL query using Qwen first, then GPT fallback"""
        try:
            logger.info(f"🔍 SQL_AGENT: Generating SQL for: {user_query}")

            # Resolve table name
            if not active_table:
                active_info = self.dataset_manager.get_active_dataset(
                    force_refresh=True)
                active_table = (
                    active_info["table_name"]
                    if isinstance(active_info, dict) and active_info.get("table_name")
                    else None
                )
                logger.info(f"📊 SQL_AGENT: Using active table: {active_table}")

            if not active_table:
                return None, "No active dataset. Please upload a CSV file first."

            # Detect anomaly/unusual pattern queries - generate aggregation SQL (optionally by group like category/product)
            anomaly_keywords = ["anomal", "outlier", "unusual", "drop", "spike", "irregular", "abnormal", "unexpected"]
            uq_lower = user_query.lower()
            is_anomaly = any(keyword in uq_lower for keyword in anomaly_keywords)
            # Detect desire for grouping (category/product/item/brand)
            group_phrases = [
                "by product", "per product", "each product",
                "by category", "per category", "each category", "by product category", "product categories",
                "by item", "per item", "each item",
                "by sku", "per sku",
                "by brand", "per brand"
            ]
            wants_group = any(p in uq_lower for p in group_phrases) or (
                ("category" in uq_lower or "product" in uq_lower or "item" in uq_lower or "sku" in uq_lower or "brand" in uq_lower)
                and "anomal" in uq_lower
            )

            # Helper: pick best grouping column present in the table
            def _get_group_column():
                try:
                    cols = db_manager.execute_query_dict(
                        f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = '{active_table}'
                        ORDER BY ordinal_position
                        """
                    )
                    column_names = [c['column_name'] for c in cols]
                except Exception:
                    column_names = []

                preferred = [
                    # product-first
                    'product', 'product_name', 'product_title', 'product_id', 'sku',
                    # item variants
                    'item', 'item_name',
                    # brand
                    'brand', 'brand_name',
                    # categories
                    'product_category', 'category', 'category_name'
                ]
                for c in preferred:
                    if c in column_names:
                        return c
                # Fallback: None (no grouping available)
                return None

            if is_anomaly:
                logger.info("🚨 Detected anomaly query - generating monthly aggregation SQL")
                
                # Grouped anomaly detection (category/product/etc.)
                if wants_group:
                    group_col = _get_group_column()
                    if not group_col:
                        logger.info("ℹ️ Grouping requested but no suitable column found; falling back to non-grouped")
                    else:
                        logger.info(f"🧩 Detected grouping dimension for anomaly query: {group_col}")
                    # Timeframe filters for category grouping
                    if group_col and ("last 3 months" in uq_lower or "last three months" in uq_lower):
                        logger.info("🗓️ Applying 'last 3 months' timeframe with category grouping")
                        sql = f"""
                        WITH bounds AS (
                            SELECT 
                                (DATE_TRUNC('month', MAX(CAST(date AS DATE))) - INTERVAL '2 month') AS m_start,
                                (DATE_TRUNC('month', MAX(CAST(date AS DATE))) + INTERVAL '1 month') AS m_end
                            FROM {active_table}
                            WHERE date IS NOT NULL
                        ),
                        monthly_sales AS (
                            SELECT 
                                DATE_TRUNC('month', CAST(t.date AS DATE)) as date,
                                t.{group_col},
                                SUM(t.total_amount) as total_amount,
                                COUNT(*) as transaction_count
                            FROM {active_table} t
                            CROSS JOIN bounds b
                            WHERE t.date IS NOT NULL
                              AND CAST(t.date AS DATE) >= b.m_start
                              AND CAST(t.date AS DATE) < b.m_end
                            GROUP BY DATE_TRUNC('month', CAST(t.date AS DATE)), t.{group_col}
                        )
                        SELECT 
                            date,
                            {group_col},
                            total_amount,
                            transaction_count
                        FROM monthly_sales
                        ORDER BY {group_col}, date;
                        """
                        logger.info("✅ Generated category anomaly SQL for last 3 months")
                        logger.info(f"🔍 SQL Query: {sql}")
                        return sql, None
                    
                    if group_col and ("last 6 months" in uq_lower or "last six months" in uq_lower):
                        logger.info("🗓️ Applying 'last 6 months' timeframe with category grouping")
                        sql = f"""
                        WITH bounds AS (
                            SELECT 
                                (DATE_TRUNC('month', MAX(CAST(date AS DATE))) - INTERVAL '5 month') AS m_start,
                                (DATE_TRUNC('month', MAX(CAST(date AS DATE))) + INTERVAL '1 month') AS m_end
                            FROM {active_table}
                            WHERE date IS NOT NULL
                        ),
                        monthly_sales AS (
                            SELECT 
                                DATE_TRUNC('month', CAST(t.date AS DATE)) as date,
                                t.{group_col},
                                SUM(t.total_amount) as total_amount,
                                COUNT(*) as transaction_count
                            FROM {active_table} t
                            CROSS JOIN bounds b
                            WHERE t.date IS NOT NULL
                              AND CAST(t.date AS DATE) >= b.m_start
                              AND CAST(t.date AS DATE) < b.m_end
                            GROUP BY DATE_TRUNC('month', CAST(t.date AS DATE)), t.{group_col}
                        )
                        SELECT 
                            date,
                            {group_col},
                            total_amount,
                            transaction_count
                        FROM monthly_sales
                        ORDER BY {group_col}, date;
                        """
                        logger.info("✅ Generated category anomaly SQL for last 6 months")
                        logger.info(f"🔍 SQL Query: {sql}")
                        return sql, None
                    
                    if group_col and ("last quarter" in uq_lower or "previous quarter" in uq_lower):
                        logger.info("🗓️ Applying 'last quarter' timeframe with category grouping (previous completed quarter)")
                        sql = f"""
                        WITH latest AS (
                            SELECT DATE_TRUNC('quarter', MAX(CAST(date AS DATE))) AS curr_q
                            FROM {active_table}
                            WHERE date IS NOT NULL
                        ), bounds AS (
                            SELECT (curr_q - INTERVAL '3 month') AS q_start,
                                   curr_q AS q_end
                            FROM latest
                        ),
                        monthly_sales AS (
                            SELECT 
                                DATE_TRUNC('month', CAST(t.date AS DATE)) as date,
                                t.{group_col},
                                SUM(t.total_amount) as total_amount,
                                COUNT(*) as transaction_count
                            FROM {active_table} t
                            CROSS JOIN bounds b
                            WHERE t.date IS NOT NULL
                              AND CAST(t.date AS DATE) >= b.q_start
                              AND CAST(t.date AS DATE) < b.q_end
                            GROUP BY DATE_TRUNC('month', CAST(t.date AS DATE)), t.{group_col}
                        )
                        SELECT 
                            date,
                            {group_col},
                            total_amount,
                            transaction_count
                        FROM monthly_sales
                        ORDER BY {group_col}, date;
                        """
                        logger.info("✅ Generated category anomaly SQL for last quarter")
                        logger.info(f"🔍 SQL Query: {sql}")
                        return sql, None
                    
                    if group_col:
                        # Full-range grouped monthly aggregation
                        sql = f"""
                        WITH monthly_sales AS (
                            SELECT 
                                DATE_TRUNC('month', CAST(date AS DATE)) as date,
                                {group_col},
                                SUM(total_amount) as total_amount,
                                COUNT(*) as transaction_count
                            FROM {active_table}
                            WHERE date IS NOT NULL
                            GROUP BY DATE_TRUNC('month', CAST(date AS DATE)), {group_col}
                        )
                        SELECT 
                            date,
                            {group_col},
                            total_amount,
                            transaction_count
                        FROM monthly_sales
                        ORDER BY {group_col}, date;
                        """
                        logger.info("✅ Generated grouped anomaly SQL without timeframe filter")
                        logger.info(f"🔍 SQL Query: {sql}")
                        return sql, None

                # Optional timeframe filters (non-category case)
                if "last 3 months" in uq_lower or "last three months" in uq_lower:
                    logger.info("🗓️ Applying 'last 3 months' timeframe filter")
                    sql = f"""
                    WITH bounds AS (
                        SELECT 
                            (DATE_TRUNC('month', MAX(CAST(date AS DATE))) - INTERVAL '2 month') AS m_start,
                            (DATE_TRUNC('month', MAX(CAST(date AS DATE))) + INTERVAL '1 month') AS m_end
                        FROM {active_table}
                        WHERE date IS NOT NULL
                    ),
                    monthly_sales AS (
                        SELECT 
                            DATE_TRUNC('month', CAST(t.date AS DATE)) as date,
                            SUM(t.total_amount) as total_amount,
                            COUNT(*) as transaction_count
                        FROM {active_table} t
                        CROSS JOIN bounds b
                        WHERE t.date IS NOT NULL
                          AND CAST(t.date AS DATE) >= b.m_start
                          AND CAST(t.date AS DATE) < b.m_end
                        GROUP BY DATE_TRUNC('month', CAST(t.date AS DATE))
                    )
                    SELECT 
                        date,
                        total_amount,
                        transaction_count,
                        AVG(total_amount) OVER () as avg_amount,
                        STDDEV(total_amount) OVER () as stddev_amount
                    FROM monthly_sales
                    ORDER BY date;
                    """
                    logger.info("✅ Generated anomaly detection SQL for last 3 months")
                    logger.info(f"🔍 SQL Query: {sql}")
                    return sql, None
                
                if "last 6 months" in uq_lower or "last six months" in uq_lower:
                    logger.info("🗓️ Applying 'last 6 months' timeframe filter")
                    sql = f"""
                    WITH bounds AS (
                        SELECT 
                            (DATE_TRUNC('month', MAX(CAST(date AS DATE))) - INTERVAL '5 month') AS m_start,
                            (DATE_TRUNC('month', MAX(CAST(date AS DATE))) + INTERVAL '1 month') AS m_end
                        FROM {active_table}
                        WHERE date IS NOT NULL
                    ),
                    monthly_sales AS (
                        SELECT 
                            DATE_TRUNC('month', CAST(t.date AS DATE)) as date,
                            SUM(t.total_amount) as total_amount,
                            COUNT(*) as transaction_count
                        FROM {active_table} t
                        CROSS JOIN bounds b
                        WHERE t.date IS NOT NULL
                          AND CAST(t.date AS DATE) >= b.m_start
                          AND CAST(t.date AS DATE) < b.m_end
                        GROUP BY DATE_TRUNC('month', CAST(t.date AS DATE))
                    )
                    SELECT 
                        date,
                        total_amount,
                        transaction_count,
                        AVG(total_amount) OVER () as avg_amount,
                        STDDEV(total_amount) OVER () as stddev_amount
                    FROM monthly_sales
                    ORDER BY date;
                    """
                    logger.info("✅ Generated anomaly detection SQL for last 6 months")
                    logger.info(f"🔍 SQL Query: {sql}")
                    return sql, None

                if "last quarter" in uq_lower or "previous quarter" in uq_lower:
                    logger.info("🗓️ Applying 'last quarter' timeframe filter (previous completed quarter)")
                    sql = f"""
                    WITH latest AS (
                        SELECT DATE_TRUNC('quarter', MAX(CAST(date AS DATE))) AS curr_q
                        FROM {active_table}
                        WHERE date IS NOT NULL
                    ), bounds AS (
                        SELECT (curr_q - INTERVAL '3 month') AS q_start,
                               curr_q AS q_end
                        FROM latest
                    ),
                    monthly_sales AS (
                        SELECT 
                            DATE_TRUNC('month', CAST(t.date AS DATE)) as date,
                            SUM(t.total_amount) as total_amount,
                            COUNT(*) as transaction_count
                        FROM {active_table} t
                        CROSS JOIN bounds b
                        WHERE t.date IS NOT NULL
                          AND CAST(t.date AS DATE) >= b.q_start
                          AND CAST(t.date AS DATE) < b.q_end
                        GROUP BY DATE_TRUNC('month', CAST(t.date AS DATE))
                    )
                    SELECT 
                        date,
                        total_amount,
                        transaction_count,
                        AVG(total_amount) OVER () as avg_amount,
                        STDDEV(total_amount) OVER () as stddev_amount
                    FROM monthly_sales
                    ORDER BY date;
                    """
                    logger.info("✅ Generated anomaly detection SQL for last quarter")
                    logger.info(f"🔍 SQL Query: {sql}")
                    return sql, None
                
                # Default: full-range monthly aggregation
                sql = f"""
                WITH monthly_sales AS (
                    SELECT 
                        DATE_TRUNC('month', CAST(date AS DATE)) as date,
                        SUM(total_amount) as total_amount,
                        COUNT(*) as transaction_count
                    FROM {active_table}
                    WHERE date IS NOT NULL
                    GROUP BY DATE_TRUNC('month', CAST(date AS DATE))
                )
                SELECT 
                    date,
                    total_amount,
                    transaction_count,
                    AVG(total_amount) OVER () as avg_amount,
                    STDDEV(total_amount) OVER () as stddev_amount
                FROM monthly_sales
                ORDER BY date;
                """
                logger.info("✅ Generated anomaly detection SQL without timeframe filter")
                logger.info(f"🔍 SQL Query: {sql}")
                return sql, None

            # Get schema info and context
            schema_info = self._get_table_schema(active_table)
            schema_context = vector_db.get_schema_context(user_query)

            # ---------- QWEN TRY FIRST ----------
            if self.qwen_model and self.use_qwen:
                try:
                    logger.info("🚀 Trying Qwen Text2SQL model first...")
                    messages = self.qwen_prompt_builder(
                        question=user_query,
                        db_schema=f"{schema_info}\n{schema_context}",
                    )

                    qresp = self.qwen_model.create_chat_completion(
                        messages=messages,
                        temperature=0,
                        max_tokens=512,
                    )

                    content = (
                        qresp.get("choices", [{}])[0]
                             .get("message", {})
                             .get("content", "")
                    ).strip()

                    logger.info(
                        f"🧾 Qwen raw output: {content[:200]}{'...' if len(content) > 200 else ''}")

                    sql_query = self._clean_sql(content)

                    # If the model added extra text, extract the first SELECT ... ;
                    if not sql_query.lower().startswith("select"):
                        m = re.search(r"(?is)\bselect\b.*?;", sql_query)
                        if m:
                            sql_query = m.group(0).strip()

                    logger.info(f"🧪 Qwen cleaned candidate: {sql_query}")

                    validated_sql = self._validate_sql(sql_query)
                    if validated_sql:
                        logger.info("✅ Qwen succeeded (using model output).")
                        return validated_sql, None
                    else:
                        logger.warning(
                            "⚠️ Qwen produced invalid SQL; will fall back to GPT.")

                except Exception as qe:
                    logger.warning(
                        f"⚠️ Qwen exception; falling back to GPT. Details: {qe}")

            # ---------- GPT FALLBACK ----------
            logger.info("💬 Using GPT fallback...")
            sql_query = self._generate_sql_with_gpt(
                user_query, active_table, schema_info, schema_context)
            validated_sql = self._validate_sql(sql_query)
            if validated_sql:
                return validated_sql, None
            else:
                return None, "Generated SQL query is invalid."

        except Exception as e:
            logger.error(f"❌ SQL_AGENT Error: {e}")
            return None, f"Error generating SQL: {str(e)}"

    def _generate_sql_with_gpt(self, user_query, active_table, schema_info, schema_context):
        """Existing GPT-based SQL generation"""
        system_prompt = f"""
        You are an expert SQL query generator. Convert natural language to PostgreSQL queries.
        Active table: {active_table}
        Schema:
        {schema_info}
        Context:
        {schema_context}
        Rules: SELECT only. No updates/deletes. Output SQL only.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate SQL query for: {user_query}"},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        sql_query = response.choices[0].message.content.strip()
        return self._clean_sql(sql_query)

    def _clean_sql(self, sql_query: str) -> str:
        if not sql_query:
            return ""
        s = sql_query.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s.replace("sql\n", "").replace("SQL\n", "")
        return s.strip()

    def _get_table_schema(self, table_name):
        """Get table schema information"""
        try:
            query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            columns_info = db_manager.execute_query_dict(query)
            schema_description = f"Table: {table_name}\nColumns:\n"
            for row in columns_info:
                schema_description += f"- {row['column_name']} ({row['data_type']})\n"
            return schema_description
        except Exception as e:
            logger.error(f"Schema retrieval error: {e}")
            return f"Table: {table_name} (schema unavailable)"

    def _validate_sql(self, sql_query):
        """Basic SQL validation"""
        if not sql_query:
            return None
        if not sql_query.strip().upper().startswith('SELECT'):
            return None
        dangerous_keywords = ['DROP', 'DELETE',
                              'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
            return None
        return sql_query
