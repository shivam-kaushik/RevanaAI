# backend/tools/nl2sql_tool.py
import re
import logging
logger = logging.getLogger(__name__)

class NL2SQLTool:
    """Converts natural language queries to SQL using an LLM."""

    def __init__(self, llm, schema_text: str):
        self.llm = llm
        self.schema_text = schema_text

    def update_schema(self, schema_text: str):
        self.schema_text = schema_text
        
    def run(self, nl_query: str) -> str:
        prompt = f"""
You are an expert PostgreSQL SQL generator.

TASK: Convert the user's question to a SQL query that returns EXACTLY TWO columns:
  - ds (DATE), y (NUMERIC)

STRICT RULES (do not violate):
- YOU MUST SELECT FROM THE ACTIVE TABLE listed below. Do not invent tables.
- Do NOT use generate_series unless you LEFT JOIN it to the real aggregated data
  (never return constant zeros).
- Use monthly aggregation unless the user asks otherwise:
  date_trunc('month', <date_expr>)::date AS ds
- All date strings in this database are stored as text in format 'YYYY-MM-DD HH24:MI:SS'.
- When parsing text → timestamp, ALWAYS use:
  to_timestamp(<text_col>, 'YYYY-MM-DD HH24:MI:SS')
- If a column is already DATE/TIMESTAMP, do NOT wrap it in to_timestamp().
- If the column is already DATE/TIMESTAMP, do NOT wrap with to_timestamp.
- y must be a real metric (e.g., SUM(line_total), SUM(quantity), COUNT(*), etc.).
- Include a WHERE that excludes future dates (<= CURRENT_DATE) unless the user asks otherwise.
- Return ONLY SQL (no comments, no markdown).

Schema reference:
{self.schema_text}

User Query: {nl_query}
SQL:
"""
        resp = self.llm.invoke(prompt)
        sql = resp.content.strip()

        # 🔍 DEBUG LOG: print the generated SQL
        print("\n================= Generated SQL =================")
        print(sql)
        print("=================================================\n")
        # ---- Guardrails: fix common LLM slips ----

        # 1) Normalize ANY to_timestamp(..., '...') mask to the correct one
        sql = re.sub(
            r"to_timestamp\(([^,]+),\s*'[^']*'\)",
            r"to_timestamp(\1, 'YYYY-MM-DD HH24:MI:SS')",
            sql,
            flags=re.IGNORECASE
        )

        # 2) Ensure monthly truncation alias is 'ds'
        sql = re.sub(
            r"date_trunc\('month'\s*,\s*([^)]+)\)\s*::date\s+AS\s+\w+",
            r"date_trunc('month', \1)::date AS ds",
            sql,
            flags=re.IGNORECASE
        )

        # 3) Trim any garbage after final semicolon
        if ";" in sql:
            sql = sql[: sql.rfind(";") + 1]

        return sql
