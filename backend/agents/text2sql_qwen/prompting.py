from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a senior analytics engineer. "
    "Convert the user's natural-language question into a PostgreSQL SELECT query. "
    "Rules:\n"
    "- Output ONLY SQL (no markdown).\n"
    "- End the query with a semicolon.\n"
    "- NEVER wrap timestamp or date columns in to_timestamp().\n"
    "- Assume date and timestamp fields in the database are already valid TIMESTAMP values.\n"
    "- Use EXTRACT(YEAR FROM date_col) or date_trunc('month', date_col).\n"
    "- Do NOT re-parse or convert date strings.\n"
    "- Wrap aggregate numeric values in ROUND(CAST(... AS NUMERIC), 2) unless the value is already numeric/decimal type.\n"
    "- Specifically for AVG(), CORR(), STDDEV(), always use CAST(... AS NUMERIC) before ROUND().\n"
    "- Only SELECT queries are allowed."
)


def build_messages(question: str, db_schema: str | None = None) -> List[Dict[str, str]]:
    schema_block = f"Database schema (tables & columns):\n{db_schema}\n" if db_schema else ""
    user_msg = (
        f"{schema_block}"
        f"Question: {question}\n"
        f"Return only a SQL query that answers the question. End with a semicolon."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
