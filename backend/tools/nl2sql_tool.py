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

CRITICAL DATE FORMAT RULES:
- The 'invoicedate' column is stored as TEXT in format 'MM/DD/YYYY HH:MI' (e.g., '12/1/2010 8:26')
- When parsing invoicedate, you MUST use: to_timestamp(invoicedate, 'MM/DD/YYYY HH24:MI')
- NEVER use 'YYYY-MM-DD HH24:MI:SS' format for invoicedate
- Example: date_trunc('month', to_timestamp(invoicedate, 'MM/DD/YYYY HH24:MI'))::date AS ds

- y must be a real metric (e.g., SUM(quantity), SUM(unitprice * quantity), COUNT(*), etc.).
- Include a WHERE that excludes future dates (<= CURRENT_DATE) using: 
  WHERE to_timestamp(invoicedate, 'MM/DD/YYYY HH24:MI')::date <= CURRENT_DATE
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
        # Fix any incorrect date format patterns to use the correct MM/DD/YYYY HH24:MI format
        sql = re.sub(
            r"to_timestamp\(invoicedate,\s*'[^']*'\)",
            r"to_timestamp(invoicedate, 'MM/DD/YYYY HH24:MI')",
            sql,
            flags=re.IGNORECASE
        )

        # Ensure monthly truncation alias is 'ds'
        sql = re.sub(
            r"date_trunc\('month'\s*,\s*([^)]+)\)\s*::date\s+AS\s+\w+",
            r"date_trunc('month', \1)::date AS ds",
            sql,
            flags=re.IGNORECASE
        )

        # Trim any garbage after final semicolon
        if ";" in sql:
            sql = sql[: sql.rfind(";") + 1]

        return sql
