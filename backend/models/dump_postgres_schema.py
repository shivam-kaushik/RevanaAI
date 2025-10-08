import os
import psycopg2
from collections import defaultdict


def map_type(t: str) -> str:
    t = t.upper()
    if 'TIMESTAMP' in t:
        return 'TIMESTAMP'
    if t == 'DATE':
        return 'DATE'
    if 'CHAR' in t or 'TEXT' in t or 'VARCHAR' in t:
        return 'TEXT'
    if 'INT' in t:
        return 'INT'
    if 'DOUBLE' in t or 'REAL' in t or 'NUMERIC' in t or 'DECIMAL' in t:
        return 'NUMERIC'
    if 'BOOL' in t:
        return 'BOOLEAN'
    return 'TEXT'


def main():
    conn = psycopg2.connect(
        host=os.getenv('PGHOST', 'localhost'),
        port=int(os.getenv('PGPORT', '5432')),
        user=os.getenv('PGUSER', 'postgres'),
        password=os.getenv('PGPASSWORD', ''),
        dbname=os.getenv('PGDATABASE', 'postgres'),
    )
    cur = conn.cursor()
    cur.execute("""
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog','information_schema')
ORDER BY table_schema, table_name, ordinal_position;
""")
    rows = cur.fetchall()
    grouped = defaultdict(list)
    for schema, table, col, dtype in rows:
        grouped[(schema, table)].append((col, map_type(dtype)))
    parts = []
    for (schema, table), cols in grouped.items():
        parts.append('TABLE ' + table + ' (\n' +
                     ',\n'.join([f'  {c} {t}' for c, t in cols]) + '\n);')
    print('\n\n'.join(parts))


if __name__ == '__main__':
    main()
