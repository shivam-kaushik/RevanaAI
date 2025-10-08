import argparse
import os
import pandas as pd


def guess_pg_type(dtype_str: str) -> str:
    ds = dtype_str.lower()
    if ds.startswith('datetime64'):
        return 'TIMESTAMP'
    if ds.startswith('int'):
        return 'BIGINT' if '64' in ds else 'INT'
    if ds.startswith('float'):
        return 'NUMERIC'
    if ds == 'bool':
        return 'BOOLEAN'
    return 'TEXT'


def schema_from_csv(path: str, table_name: str) -> str:
    df = pd.read_csv(path, nrows=5000)
    # try to parse likely date columns
    for c in df.columns:
        try:
            parsed = pd.to_datetime(
                df[c], errors='raise', infer_datetime_format=True)
            if parsed.notna().mean() > 0.7:
                df[c] = parsed
        except Exception:
            pass
    lines = [f"TABLE {table_name} ("]
    cols = []
    for c, dt in df.dtypes.items():
        pg = guess_pg_type(str(dt))
        col = c.strip().replace(' ', '_').replace('-', '_')
        cols.append(f"  {col} {pg}")
    lines.append(',\n'.join(cols))
    lines.append(');')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_dir', required=True)
    args = ap.parse_args()
    out = []
    for fname in sorted(os.listdir(args.csv_dir)):
        if not fname.lower().endswith('.csv'):
            continue
        table = os.path.splitext(fname)[0].lower().replace(
            ' ', '_').replace('-', '_')
        out.append(schema_from_csv(os.path.join(args.csv_dir, fname), table))
    print('\n\n'.join(out))


if __name__ == '__main__':
    main()
