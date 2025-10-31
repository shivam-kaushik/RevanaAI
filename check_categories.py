from backend.utils.database import db_manager

result = db_manager.execute_query('SELECT DISTINCT product_category FROM product_embeddings ORDER BY product_category')
print('📋 Available Categories in Database:')
print()
for _, row in result.iterrows():
    print(f'  - {row[0]}')
