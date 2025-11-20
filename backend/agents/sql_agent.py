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

            # --- Get schema info and vector context ---
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

    # -------------------------------------------------------------------------
    # GPT fallback path
    # -------------------------------------------------------------------------
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
            s = s.replace("sql\n", "").replace("SQL\n", "")
        return s.strip()

    def _get_table_schema(self, table_name):
        """Get table schema information from information_schema."""
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
