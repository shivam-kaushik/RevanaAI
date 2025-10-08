import re
from typing import Iterable

DANGEROUS = re.compile(
    r"\b(ALTER|DROP|TRUNCATE|DELETE|UPDATE|INSERT|MERGE|GRANT|REVOKE)\b", re.IGNORECASE)
SELECT = re.compile(r"\bSELECT\b", re.IGNORECASE)


def is_safe_sql(sql: str) -> bool:
    if not SELECT.search(sql):
        return False
    if DANGEROUS.search(sql):
        return False
    return True


def whitelist_columns(sql: str, allowed_columns: Iterable[str]) -> bool:
    lowered = sql.lower()
    for bad_token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", lowered):
        if bad_token in {"select", "from", "where", "and", "or", "group", "by", "order", "limit", "sum", "count",
                         "avg", "min", "max", "as", "on", "join", "left", "right", "inner", "outer", "date_trunc",
                         "extract", "between", "like", "not", "in", "distinct", "having", "case", "when", "then", "end"}:
            continue
    return True
