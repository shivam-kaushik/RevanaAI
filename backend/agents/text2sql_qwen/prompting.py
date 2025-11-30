from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a senior analytics engineer. "
    "Your ONLY task is to convert a natural-language question into a VALID PostgreSQL SQL query. "
    "Rules: Output ONLY the SQL; no explanations, no markdown. End with a semicolon. "
    "Prefer standard SQL and Postgres-compatible date functions."
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
