"""Quick test to check what's actually in the database"""
import sys
sys.path.insert(0, '.')

from backend.utils.database import db_manager
from backend.utils.dataset_manager import DatasetManager
import pandas as pd

# Get active dataset
dm = DatasetManager()
active = dm.get_active_dataset()
print(f"Active dataset: {active}")

if active and active.get('table_name'):
    table_name = active['table_name']
    
    # Check what columns exist
    schema_query = f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
    """
    columns = db_manager.execute_query_dict(schema_query)
    print(f"\nColumns in {table_name}:")
    for col in columns:
        print(f"  - {col['column_name']} ({col['data_type']})")
    
    # Check total row count
    count_query = f"SELECT COUNT(*) as total FROM {table_name}"
    total = db_manager.execute_query_dict(count_query)
    print(f"\nTotal rows in table: {total[0]['total']}")
    
    # Check date column - count by month
    month_count_query = f"""
        SELECT 
            TO_CHAR(date, 'YYYY-MM') as month,
            COUNT(*) as count
        FROM {table_name}
        WHERE date IS NOT NULL
        GROUP BY TO_CHAR(date, 'YYYY-MM')
        ORDER BY month
    """
    
    print(f"\n\nRow count by month (raw data):")
    try:
        monthly_counts = db_manager.execute_query_dict(month_count_query)
        for row in monthly_counts:
            print(f"  {row['month']}: {row['count']} transactions")
    except Exception as e:
        print(f"Error counting by month: {e}")
    
    # Now run the actual aggregation query
    print(f"\n\nRunning the monthly aggregation query (same as anomaly detection):")
    agg_query = f"""
        WITH monthly_sales AS (
            SELECT 
                DATE_TRUNC('month', CAST(date AS DATE)) as date,
                SUM(total_amount) as total_amount,
                COUNT(*) as transaction_count
            FROM {table_name}
            WHERE date IS NOT NULL
            GROUP BY DATE_TRUNC('month', CAST(date AS DATE))
        )
        SELECT 
            date,
            total_amount,
            transaction_count
        FROM monthly_sales
        ORDER BY date;
    """
    
    try:
        agg_results = db_manager.execute_query(agg_query)
        print(f"\nAggregation returned {len(agg_results)} months:")
        print(agg_results)
    except Exception as e:
        print(f"Error running aggregation: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No active dataset found!")

