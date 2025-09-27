import psycopg2
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.config import Config

def init_database():
    """Initialize database with sample data"""
    try:
        # Connect to database
        conn = psycopg2.connect(Config.DATABASE_URL)
        cursor = conn.cursor()
        
        # Read and execute schema
        with open('database/schema.sql', 'r') as f:
            schema_sql = f.read()
            cursor.execute(schema_sql)
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM retail_transactions")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Create sample data
            sample_data = generate_sample_data()
            
            # Insert sample data
            insert_query = """
            INSERT INTO retail_transactions 
            (date, customer_name, product_category, product_name, quantity, unit_price, total_cost, payment_method, city, store_type, discount_applied)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for row in sample_data:
                cursor.execute(insert_query, row)
            
            conn.commit()
            print("Sample data inserted successfully")
        
        cursor.close()
        conn.close()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {e}")

def generate_sample_data():
    """Generate realistic sample retail data"""
    import random
    from datetime import datetime, timedelta
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    products = {
        'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Smartwatch'],
        'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Shoes'],
        'Home & Garden': ['Furniture', 'Decor', 'Kitchenware', 'Gardening Tools'],
        'Sports': ['Bicycle', 'Tennis Racket', 'Yoga Mat', 'Running Shoes'],
        'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Children']
    }
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet']
    store_types = ['Physical', 'Online']
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(1000):  # Generate 1000 sample transactions
        date = start_date + timedelta(days=random.randint(0, 365))
        category = random.choice(categories)
        product = random.choice(products[category])
        quantity = random.randint(1, 5)
        unit_price = round(random.uniform(10, 500), 2)
        total_cost = round(quantity * unit_price * (0.9 if random.random() > 0.7 else 1.0), 2)
        
        transaction = (
            date.date(),
            f"Customer_{random.randint(1, 100)}",
            category,
            product,
            quantity,
            unit_price,
            total_cost,
            random.choice(payment_methods),
            random.choice(cities),
            random.choice(store_types),
            random.random() > 0.7
        )
        data.append(transaction)
    
    return data

if __name__ == "__main__":
    init_database()