from __future__ import annotations
from typing import Optional
from llama_cpp import Llama
from .model_loader import get_model
from .prompting import build_messages
from .validator import is_safe_sql


def generate_sql(question: str, db_schema: Optional[str] = None) -> str:
    llm: Llama = get_model()
    messages = build_messages(question, db_schema)

    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        top_p=0.9,
        max_tokens=256,
    )
    content = out["choices"][0]["message"]["content"].strip()

    # Strip code fences if present
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("sql", "", 1).strip()

    if not content.endswith(";"):
        content = content.rstrip() + ";"

    if not is_safe_sql(content):
        raise ValueError(
            "Generated SQL failed safety checks (must be SELECT and non-destructive).")

    return content
