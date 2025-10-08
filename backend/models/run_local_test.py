from app.generator import generate_sql
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_local_test.py " <
              question > " [schema_file]")
        raise SystemExit(1)

    question = sys.argv[1]
    db_schema = None
    if len(sys.argv) >= 3:
        p = Path(sys.argv[2])
        if p.exists():
            db_schema = p.read_text(encoding="utf-8")

    sql = generate_sql(question, db_schema)
    print(sql)


if __name__ == "__main__":
    main()
