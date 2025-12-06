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
        # OpenAI client for GPT fallback
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.dataset_manager = DatasetManager()

        # --- Try loading embedded Qwen Text2SQL model ---
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
            f"USE_QWEN_MODEL={getattr(self, 'use_qwen', False)} | "
            f"qwen_model_is_none={self.qwen_model is None}"
        )

    def generate_sql(self, user_query, active_table=None):
        """
        Generate SQL query from natural language:
        1) Try local Qwen Text2SQL model
        2) If invalid / error, fall back to GPT
        """
        try:
            logger.info(f"🔍 SQL_AGENT: Generating SQL for: {user_query}")

            # --- Resolve active table ---
            if not active_table:
                active_info = self.dataset_manager.get_active_dataset(force_refresh=True)
                active_table = (
                    active_info["table_name"]
                    if isinstance(active_info, dict) and active_info.get("table_name")
                    else None
                )
                logger.info(f"📊 SQL_AGENT: Using active table: {active_table}")

            if not active_table:
                return None, "No active dataset. Please upload a CSV file first."

            # --- Auto-fix schema types if needed (One-time check per session/table could be better, but doing it here for safety) ---
            try:
                self._ensure_date_columns_are_typed(active_table)
            except Exception as e:
                logger.warning(f"⚠️ Auto-fixing date columns failed: {e}")

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

            # Helper: get column names and types from schema
            def _get_table_columns():
                try:
                    cols = db_manager.execute_query_dict(
                        f"""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = '{active_table}'
                        ORDER BY ordinal_position
                        """
                    )
                    return cols
                except Exception:
                    return []
            
            # Helper: detect date/time column and its data type
            def _get_date_column_info():
                cols = _get_table_columns()
                column_names = [c['column_name'] for c in cols]
                
                # Preferred date column names
                date_keywords = ['date', 'invoicedate', 'orderdate', 'transactiondate', 'timestamp', 'time']
                for keyword in date_keywords:
                    for col in column_names:
                        if keyword == col.lower().replace('_', ''):
                            # Find the data type for this column
                            col_info = next((c for c in cols if c['column_name'] == col), None)
                            return {'name': col, 'type': col_info['data_type'] if col_info else 'unknown'}
                
                # Fallback: check data types
                for col_info in cols:
                    if 'timestamp' in col_info['data_type'].lower() or 'date' in col_info['data_type'].lower():
                        return {'name': col_info['column_name'], 'type': col_info['data_type']}
                
                return {'name': 'date', 'type': 'unknown'}  # ultimate fallback
            
            # Helper: get SQL expression to convert column to DATE based on its type
            def _get_date_cast_expr(col_name, col_type):
                col_type_lower = col_type.lower()
                
                # If already a date/timestamp type, just cast it
                if 'timestamp' in col_type_lower or 'date' in col_type_lower:
                    return f"CAST({col_name} AS DATE)"
                
                # If it's text/varchar/character varying, detect the format by sampling
                if any(t in col_type_lower for t in ['text', 'varchar', 'character varying', 'char']):
                    # Sample a few values to detect format
                    try:
                        sample_query = f"SELECT {col_name} FROM {active_table} WHERE {col_name} IS NOT NULL LIMIT 1"
                        result = db_manager.execute_query_dict(sample_query)
                        if result and len(result) > 0:
                            sample_value = str(result[0][col_name])
                            
                            # Detect format based on sample
                            if '/' in sample_value:
                                # Could be MM/DD/YYYY or DD/MM/YYYY
                                parts = sample_value.split('/')
                                if len(parts) >= 3:
                                    # If first part > 12, it's DD/MM/YYYY, else MM/DD/YYYY
                                    first_num = int(parts[0].split()[0])  # Handle time if present
                                    if first_num > 12:
                                        return f"TO_DATE({col_name}, 'DD/MM/YYYY')"
                                    else:
                                        return f"TO_DATE({col_name}, 'MM/DD/YYYY')"
                            elif '-' in sample_value:
                                # Could be YYYY-MM-DD or DD-MM-YYYY
                                parts = sample_value.split('-')
                                if len(parts) >= 3:
                                    # If first part is 4 digits, it's YYYY-MM-DD
                                    if len(parts[0]) == 4:
                                        return f"TO_DATE({col_name}, 'YYYY-MM-DD')"
                                    else:
                                        return f"TO_DATE({col_name}, 'DD-MM-YYYY')"
                    except Exception as e:
                        logger.warning(f"⚠️ Could not detect date format, using default: {e}")
                    
                    # Fallback to MM/DD/YYYY
                    return f"TO_DATE({col_name}, 'MM/DD/YYYY')"
                
                # Default fallback
                return f"CAST({col_name} AS DATE)"
            
            # Helper: detect value/amount column
            def _get_value_column():
                cols = _get_table_columns()
                column_names = [c['column_name'] for c in cols]
                
                # Preferred value column names in priority order
                value_keywords = ['total_amount', 'totalamount', 'amount', 'total', 'quantity', 'qty', 'unitprice', 'price', 'revenue', 'sales', 'value']
                for keyword in value_keywords:
                    for col in column_names:
                        if keyword == col.lower().replace('_', '').replace(' ', ''):
                            return col
                
                # Fallback: find first numeric column
                for col_info in cols:
                    data_type = col_info['data_type'].lower()
                    if any(t in data_type for t in ['int', 'numeric', 'decimal', 'float', 'double', 'real']):
                        return col_info['column_name']
                
                return 'total_amount'  # ultimate fallback
            
            # Helper: pick best grouping column present in the table
            def _get_group_column():
                cols = _get_table_columns()
                column_names = [c['column_name'] for c in cols]

                preferred = [
                    # product-first
                    'product', 'product_name', 'product_title', 'product_id', 'sku',
                    # item variants
                    'item', 'item_name', 'description',
                    # brand
                    'brand', 'brand_name',
                    # categories
                    'product_category', 'category', 'category_name', 'productcategory'
                ]
                for c in preferred:
                    if c in column_names:
                        return c
                # Fallback: None (no grouping available)
                return None

                # Fallback: None (no grouping available)
                return None

            # Detect correlation/scatter intent
            correlation_keywords = ["correlation", "scatter", "relationship", "vs", "versus"]
            is_correlation = any(keyword in uq_lower for keyword in correlation_keywords)
            
            if is_correlation and not is_anomaly:
                logger.info("📈 Detected correlation/scatter query - identifying columns")
                
                cols = _get_table_columns()
                col_names = [c['column_name'] for c in cols]
                
                # Identify numeric columns mentioned in query
                mentioned_cols = []
                for col in col_names:
                    # Check partial matches too, e.g. "price" matches "unitprice"
                    if col.lower() in uq_lower or col.lower().replace('_', '') in uq_lower.replace(' ', ''):
                        # Verify it's numeric
                        col_info = next((c for c in cols if c['column_name'] == col), None)
                        if col_info and any(t in col_info['data_type'].lower() for t in ['int', 'numeric', 'float', 'double', 'real', 'decimal']):
                             mentioned_cols.append(col)
                    # Handle "price" -> "unitprice" mapping specifically if needed
                    elif "price" in uq_lower and "price" in col.lower():
                         col_info = next((c for c in cols if c['column_name'] == col), None)
                         if col_info and any(t in col_info['data_type'].lower() for t in ['int', 'numeric', 'float', 'double', 'real']):
                             mentioned_cols.append(col)
                
                # Remove duplicates
                mentioned_cols = list(set(mentioned_cols))
                
                if len(mentioned_cols) >= 2:
                    col1 = mentioned_cols[0]
                    col2 = mentioned_cols[1]
                    logger.info(f"📍 Found correlation columns: {col1}, {col2}")
                    
                    sql = f"""
                    SELECT {col1}, {col2}
                    FROM {active_table}
                    WHERE {col1} IS NOT NULL AND {col2} IS NOT NULL
                    LIMIT 1000;
                    """
                    logger.info(f"✅ Generated Scatter Data SQL: {sql}")
                    return sql, None
                logger.info("🚨 Detected anomaly query - generating monthly aggregation SQL")
                
                # Detect column names and types from schema
                date_col_info = _get_date_column_info()
                date_col = date_col_info['name']
                date_type = date_col_info['type']
                value_col = _get_value_column()
                
                # Get proper date casting expression
                date_cast = _get_date_cast_expr(date_col, date_type)
                
                logger.info(f"📅 Detected date column: {date_col} (type: {date_type})")
                logger.info(f"💰 Detected value column: {value_col}")
                logger.info(f"🔧 Date conversion: {date_cast}")
                
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
                                (DATE_TRUNC('month', MAX({date_cast})) - INTERVAL '2 month') AS m_start,
                                (DATE_TRUNC('month', MAX({date_cast})) + INTERVAL '1 month') AS m_end
                            FROM {active_table}
                            WHERE {date_col} IS NOT NULL
                        ),
                        monthly_sales AS (
                            SELECT 
                                DATE_TRUNC('month', {date_cast}) as {date_col},
                                t.{group_col},
                                SUM(t.{value_col}) as {value_col},
                                COUNT(*) as transaction_count
                            FROM {active_table} t
                            CROSS JOIN bounds b
                            WHERE t.{date_col} IS NOT NULL
                              AND {date_cast} >= b.m_start
                              AND {date_cast} < b.m_end
                            GROUP BY DATE_TRUNC('month', {date_cast}), t.{group_col}
                        )
                        SELECT 
                            {date_col},
                            {group_col},
                            {value_col},
                            transaction_count
                        FROM monthly_sales
                        ORDER BY {group_col}, {date_col};
                        """
                        logger.info("✅ Generated category anomaly SQL for last 3 months")
                        logger.info(f"🔍 SQL Query: {sql}")
                        return sql, None
                    
                    if group_col and ("last 6 months" in uq_lower or "last six months" in uq_lower):
                        logger.info("🗓️ Applying 'last 6 months' timeframe with category grouping")
                        sql = f"""
                        WITH bounds AS (
                            SELECT 
                                (DATE_TRUNC('month', MAX({date_cast})) - INTERVAL '5 month') AS m_start,
                                (DATE_TRUNC('month', MAX({date_cast})) + INTERVAL '1 month') AS m_end
                            FROM {active_table}
                            WHERE {date_col} IS NOT NULL
                        ),
                        monthly_sales AS (
                            SELECT 
                                DATE_TRUNC('month', {date_cast}) as {date_col},
                                t.{group_col},
                                SUM(t.{value_col}) as {value_col},
                                COUNT(*) as transaction_count
                            FROM {active_table} t
                            CROSS JOIN bounds b
                            WHERE t.{date_col} IS NOT NULL
                              AND {date_cast} >= b.m_start
                              AND {date_cast} < b.m_end
                            GROUP BY DATE_TRUNC('month', {date_cast}), t.{group_col}
                        )
                        SELECT 
                            {date_col},
                            {group_col},
                            {value_col},
                            transaction_count
                        FROM monthly_sales
                        ORDER BY {group_col}, {date_col};
                        """
                        logger.info("✅ Generated category anomaly SQL for last 6 months")
                        logger.info(f"🔍 SQL Query: {sql}")
                        return sql, None
                    
                    if group_col and ("last quarter" in uq_lower or "previous quarter" in uq_lower):
                        logger.info("🗓️ Applying 'last quarter' timeframe with category grouping (previous completed quarter)")
                        sql = f"""
                        WITH latest AS (
                            SELECT DATE_TRUNC('quarter', MAX({date_cast})) AS curr_q
                            FROM {active_table}
                            WHERE {date_col} IS NOT NULL
                        ), bounds AS (
                            SELECT (curr_q - INTERVAL '3 month') AS q_start,
                                   curr_q AS q_end
                            FROM latest
                        ),
                        monthly_sales AS (
                            SELECT 
                                DATE_TRUNC('month', {date_cast}) as {date_col},
                                t.{group_col},
                                SUM(t.{value_col}) as {value_col},
                                COUNT(*) as transaction_count
                            FROM {active_table} t
                            CROSS JOIN bounds b
                            WHERE t.{date_col} IS NOT NULL
                              AND {date_cast} >= b.q_start
                              AND {date_cast} < b.q_end
                            GROUP BY DATE_TRUNC('month', {date_cast}), t.{group_col}
                        )
                        SELECT 
                            {date_col},
                            {group_col},
                            {value_col},
                            transaction_count
                        FROM monthly_sales
                        ORDER BY {group_col}, {date_col};
                        """
                        logger.info("✅ Generated category anomaly SQL for last quarter")
                        logger.info(f"🔍 SQL Query: {sql}")
                        return sql, None
                    
                    if group_col:
                        # Full-range grouped monthly aggregation
                        sql = f"""
                        WITH monthly_sales AS (
                            SELECT 
                                DATE_TRUNC('month', {date_cast}) as {date_col},
                                {group_col},
                                SUM({value_col}) as {value_col},
                                COUNT(*) as transaction_count
                            FROM {active_table}
                            WHERE {date_col} IS NOT NULL
                            GROUP BY DATE_TRUNC('month', {date_cast}), {group_col}
                        )
                        SELECT 
                            {date_col},
                            {group_col},
                            {value_col},
                            transaction_count
                        FROM monthly_sales
                        ORDER BY {group_col}, {date_col};
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
                            (DATE_TRUNC('month', MAX({date_cast})) - INTERVAL '2 month') AS m_start,
                            (DATE_TRUNC('month', MAX({date_cast})) + INTERVAL '1 month') AS m_end
                        FROM {active_table}
                        WHERE {date_col} IS NOT NULL
                    ),
                    monthly_sales AS (
                        SELECT 
                            DATE_TRUNC('month', {date_cast}) as {date_col},
                            SUM(t.{value_col}) as {value_col},
                            COUNT(*) as transaction_count
                        FROM {active_table} t
                        CROSS JOIN bounds b
                        WHERE t.{date_col} IS NOT NULL
                          AND {date_cast} >= b.m_start
                          AND {date_cast} < b.m_end
                        GROUP BY DATE_TRUNC('month', {date_cast})
                    )
                    SELECT 
                        {date_col},
                        {value_col},
                        transaction_count,
                        AVG({value_col}) OVER () as avg_amount,
                        STDDEV({value_col}) OVER () as stddev_amount
                    FROM monthly_sales
                    ORDER BY {date_col};
                    """
                    logger.info("✅ Generated anomaly detection SQL for last 3 months")
                    logger.info(f"🔍 SQL Query: {sql}")
                    return sql, None
                
                if "last 6 months" in uq_lower or "last six months" in uq_lower:
                    logger.info("🗓️ Applying 'last 6 months' timeframe filter")
                    sql = f"""
                    WITH bounds AS (
                        SELECT 
                            (DATE_TRUNC('month', MAX({date_cast})) - INTERVAL '5 month') AS m_start,
                            (DATE_TRUNC('month', MAX({date_cast})) + INTERVAL '1 month') AS m_end
                        FROM {active_table}
                        WHERE {date_col} IS NOT NULL
                    ),
                    monthly_sales AS (
                        SELECT 
                            DATE_TRUNC('month', {date_cast}) as {date_col},
                            SUM(t.{value_col}) as {value_col},
                            COUNT(*) as transaction_count
                        FROM {active_table} t
                        CROSS JOIN bounds b
                        WHERE t.{date_col} IS NOT NULL
                          AND {date_cast} >= b.m_start
                          AND {date_cast} < b.m_end
                        GROUP BY DATE_TRUNC('month', {date_cast})
                    )
                    SELECT 
                        {date_col},
                        {value_col},
                        transaction_count,
                        AVG({value_col}) OVER () as avg_amount,
                        STDDEV({value_col}) OVER () as stddev_amount
                    FROM monthly_sales
                    ORDER BY {date_col};
                    """
                    logger.info("✅ Generated anomaly detection SQL for last 6 months")
                    logger.info(f"🔍 SQL Query: {sql}")
                    return sql, None

                if "last quarter" in uq_lower or "previous quarter" in uq_lower:
                    logger.info("🗓️ Applying 'last quarter' timeframe filter (previous completed quarter)")
                    sql = f"""
                    WITH latest AS (
                        SELECT DATE_TRUNC('quarter', MAX({date_cast})) AS curr_q
                        FROM {active_table}
                        WHERE {date_col} IS NOT NULL
                    ), bounds AS (
                        SELECT (curr_q - INTERVAL '3 month') AS q_start,
                               curr_q AS q_end
                        FROM latest
                    ),
                    monthly_sales AS (
                        SELECT 
                            DATE_TRUNC('month', {date_cast}) as {date_col},
                            SUM(t.{value_col}) as {value_col},
                            COUNT(*) as transaction_count
                        FROM {active_table} t
                        CROSS JOIN bounds b
                        WHERE t.{date_col} IS NOT NULL
                          AND {date_cast} >= b.q_start
                          AND {date_cast} < b.q_end
                        GROUP BY DATE_TRUNC('month', {date_cast})
                    )
                    SELECT 
                        {date_col},
                        {value_col},
                        transaction_count,
                        AVG({value_col}) OVER () as avg_amount,
                        STDDEV({value_col}) OVER () as stddev_amount
                    FROM monthly_sales
                    ORDER BY {date_col};
                    """
                    logger.info("✅ Generated anomaly detection SQL for last quarter")
                    logger.info(f"🔍 SQL Query: {sql}")
                    return sql, None
                
                # Default: full-range monthly aggregation
                sql = f"""
                WITH monthly_sales AS (
                    SELECT 
                        DATE_TRUNC('month', {date_cast}) as {date_col},
                        SUM({value_col}) as {value_col},
                        COUNT(*) as transaction_count
                    FROM {active_table}
                    WHERE {date_col} IS NOT NULL
                    GROUP BY DATE_TRUNC('month', {date_cast})
                )
                SELECT 
                    {date_col},
                    {value_col},
                    transaction_count,
                    AVG({value_col}) OVER () as avg_amount,
                    STDDEV({value_col}) OVER () as stddev_amount
                FROM monthly_sales
                ORDER BY {date_col};
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
                        f"🧾 Qwen raw output: "
                        f"{content[:200]}{'...' if len(content) > 200 else ''}"
                    )

                    sql_query = self._clean_sql(content)

                    # If the model added explanation text, grab the first SELECT ... ;
                    if not sql_query.lower().startswith("select"):
                        m = re.search(r"(?is)\bselect\b.*?;", sql_query)
                        if m:
                            sql_query = m.group(0).strip()

                    logger.info(f"🧪 Qwen cleaned candidate: {sql_query}")

                    # --- FIX: Force active table name ---
                    # LLMs often hallucinate the table name timestamp or suffix.
                    # We simply replace whatever follows FROM with the correct active_table.
                    if active_table:
                        # Regex to find FROM table_name (case insensitive)
                        # Avoid replacing FROM inside EXTRACT(YEAR FROM ...) or substring(.. from ..)
                        # We use a callback to check the preceding word
                        def replace_from(match):
                            preceding = match.group(1) or ""
                            # If preceded by a date part (common in EXTRACT), don't replace
                            if preceding.upper() in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'DOY', 'DOW', 'QUARTER', 'WEEK']:
                                return match.group(0) # No change
                            return f"{preceding} FROM {active_table}"

                        # Capture the word before FROM (if any) and FROM
                        sql_query = re.sub(
                            r"(?i)(\w+\s+)?\bFROM\s+([a-zA-Z0-9_]+)", 
                            replace_from, 
                            sql_query
                        )
                        logger.info(f"🔧 Table name post-processed: {sql_query}")

                    validated_sql = self._validate_sql(sql_query)
                    if validated_sql:
                        logger.info("✅ Qwen succeeded (using model output).")
                        return validated_sql, None
                    else:
                        logger.warning(
                            "⚠️ Qwen produced invalid SQL; will fall back to GPT."
                        )

                except Exception as qe:
                    logger.warning(
                        f"⚠️ Qwen exception; falling back to GPT. Details: {qe}"
                    )

            # ---------- GPT FALLBACK ----------
            logger.info("💬 Using GPT fallback...")
            sql_query = self._generate_sql_with_gpt(
                user_query, active_table, schema_info, schema_context
            )
            validated_sql = self._validate_sql(sql_query)
            if validated_sql:
                return validated_sql, None
            else:
                return None, "Generated SQL query is invalid."

        except Exception as e:
            logger.error(f"❌ SQL_AGENT Error: {e}")
            return None, f"Error generating SQL: {str(e)}"

    def _generate_sql_with_gpt(self, user_query, active_table, schema_info, schema_context):
        """GPT-based SQL generation used only as fallback."""
        system_prompt = f"""
        You are an expert SQL query generator. Convert natural language to PostgreSQL SELECT queries.
        Active table: {active_table}
        Schema:
        {schema_info}
        Context:
        {schema_context}
        Rules:
        - Only SELECT queries (read-only).
        - Do NOT modify or delete data.
        - Output ONLY the SQL (no markdown).
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

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _clean_sql(self, sql_query: str) -> str:
        """Strip markdown fences and whitespace from model output."""
        if not sql_query:
            return ""
        s = sql_query.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s.replace("sql\\n", "").replace("SQL\\n", "")
        
        # New safety: remove any potential explaining text after the semicolon
        if ";" in s:
            s = s.split(";")[0] + ";"
            
        return s.strip()

    def _get_table_schema(self, table_name):
        """Get table schema information from information_schema with type hints."""
        try:
            query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            columns_info = db_manager.execute_query_dict(query)
            schema_description = f"Table: {table_name}\\nColumns:\\n"
            
            for row in columns_info:
                col_name = row['column_name']
                data_type = row['data_type']
                hint = ""
                
                # If text column looks like a date, sample it to give a hint
                if 'text' in data_type.lower() or 'char' in data_type.lower():
                    if any(k in col_name.lower() for k in ['date', 'time', 'day', 'month', 'year']):
                        # Sample
                        try:
                            # Use execute_query_dict from global db_manager
                            sample_q = f"SELECT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 1"
                            sample_res = db_manager.execute_query_dict(sample_q)
                            if sample_res:
                                val = str(sample_res[0][col_name])
                                hint = f" -- likely DATE content, sample: '{val}'"
                        except:
                            pass
                            
                schema_description += f"- {col_name} ({data_type}){hint}\\n"
            return schema_description
        except Exception as e:
            logger.error(f"Schema retrieval error: {e}")
            return f"Table: {table_name} (schema unavailable)"

    def _validate_sql(self, sql_query: str):
        """Basic SQL safety/shape validation."""
        if not sql_query:
            return None

        if not sql_query.strip().upper().startswith("SELECT"):
            return None

        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
            return None

        return sql_query

    def _ensure_date_columns_are_typed(self, table_name: str):
        """
        Inspects the table for columns that look like dates but are TEXT,
        and converts them to proper DATE/TIMESTAMP types in the database.
        """
        try:
            # 1. Get text columns
            cols = db_manager.execute_query_dict(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND data_type IN ('text', 'character varying', 'varchar')"
            )
            text_cols = [c['column_name'] for c in cols]
            
            for col in text_cols:
                # heuristic: name contains 'date' or 'time'
                if any(x in col.lower() for x in ['date', 'time', 'day', 'month', 'year']):
                    # Check sample
                    sample = db_manager.execute_query_dict(f"SELECT {col} FROM {table_name} WHERE {col} IS NOT NULL LIMIT 1")
                    if not sample:
                        continue
                    val = str(sample[0][col])
                    
                    # Detect format and alter
                    alter_sql = None
                    if '/' in val and ':' in val: # 12/1/2010 8:26 (MM/DD/YYYY HH:MM)
                        alter_sql = f"ALTER TABLE {table_name} ALTER COLUMN {col} TYPE TIMESTAMP USING TO_TIMESTAMP({col}, 'MM/DD/YYYY HH24:MI')"
                    elif '/' in val and len(val.split('/')) == 3: # 12/1/2010
                         parts = val.split('/')
                         # basic check: if first part > 12, likely DD/MM/YYYY
                         fmt = 'MM/DD/YYYY'
                         if parts[0].isdigit() and int(parts[0]) > 12: 
                             fmt = 'DD/MM/YYYY'
                         alter_sql = f"ALTER TABLE {table_name} ALTER COLUMN {col} TYPE DATE USING TO_DATE({col}, '{fmt}')"
                    elif '-' in val and len(val) >= 10: # 2023-11-24
                        alter_sql = f"ALTER TABLE {table_name} ALTER COLUMN {col} TYPE DATE USING {col}::date"
                    
                    if alter_sql:
                        logger.info(f"🔧 Auto-converting column {col} to proper type...")
                        db_manager.execute_non_query(alter_sql)
                        logger.info(f"✅ Converted {col} successfully.")
                        
        except Exception as e:
            logger.warning(f"Could not checking/fixing types for {table_name}: {e}")
