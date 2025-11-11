import pandas as pd

class SQLTool:
    """Executes SQL queries against a PostgreSQL database and returns DataFrames."""

    def __init__(self, engine):
        self.engine = engine

    def run(self, sql: str) -> pd.DataFrame:
        if not sql.strip().lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed.")
        try:
            return pd.read_sql(sql, self.engine)
        except Exception as e:
            raise RuntimeError(f"SQL execution failed: {e}")
