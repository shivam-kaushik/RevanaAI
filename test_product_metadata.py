"""Test script to check product metadata"""
from backend.utils.database import db_manager
import json

result = db_manager.execute_query('SELECT product_name, product_category, metadata FROM product_embeddings LIMIT 5')

print("📦 Product Embeddings Metadata:\n")
for _, row in result.iterrows():
    print(f"Product: {row['product_name']}")
    print(f"Category: {row['product_category']}")
    
    # Parse metadata
    metadata = row['metadata']
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    
    if metadata:
        print(f"Metadata Keys: {list(metadata.keys())}")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
    
    print()
