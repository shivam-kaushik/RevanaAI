import logging
import re
import json
from pathlib import Path
from openai import OpenAI
from backend.config import Config
from backend.utils.dataset_manager import DatasetManager
from backend.utils.database import db_manager
from backend.utils.vector_db import vector_db

logger = logging.getLogger(__name__)


class SQLAgent:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.dataset_manager = DatasetManager()

        # Try loading Qwen Text2SQL model
        try:
            from backend.agents.text2sql_qwen.model_loader import get_model
            from backend.agents.text2sql_qwen.prompting import build_messages
            from backend.config import USE_QWEN_MODEL

            self.use_qwen = USE_QWEN_MODEL
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

            # Get schema info and context
            schema_info = self._get_table_schema(active_table)
            schema_context = vector_db.get_schema_context(user_query)

            # Try Qwen model first
            if self.qwen_model and self.use_qwen:
                try:
                    logger.info("🚀 Trying Qwen Text2SQL model...")
                    from backend.agents.text2sql_qwen.prompting import build_messages
                    messages = build_messages(
                        question=user_query,
                        db_schema=f"{schema_info}\n{schema_context}",
                    )
                    response = self.qwen_model.create_chat_completion(
                        messages=messages)
                    if response and "choices" in response:
                        sql_query = response["choices"][0]["message"]["content"].strip(
                        )
                        sql_query = self._clean_sql(sql_query)
                        validated_sql = self._validate_sql(sql_query)
                        if validated_sql:
                            logger.info("✅ Qwen model succeeded.")
                            return validated_sql, None
                except Exception as qe:
                    logger.warning(
                        f"⚠️ Qwen model failed, fallback to GPT: {qe}")

            # GPT fallback
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

    def _clean_sql(self, sql_query):
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace(
                "```sql", "").replace("```", "").strip()
        return sql_query

    # keep all your validation and date-fixing helpers unchanged
    def _get_table_schema(self, table_name):
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
        if not sql_query:
            return None
        if not sql_query.strip().upper().startswith("SELECT"):
            return None
        bad_ops = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
        if any(op in sql_query.upper() for op in bad_ops):
            return None
        return sql_query
