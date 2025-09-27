-- Create retail transactions table
CREATE TABLE IF NOT EXISTS retail_transactions (
    transaction_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    customer_name VARCHAR(100),
    product_category VARCHAR(50),
    product_name VARCHAR(100),
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_cost DECIMAL(10,2),
    payment_method VARCHAR(50),
    city VARCHAR(50),
    store_type VARCHAR(50),
    discount_applied BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transactions_date ON retail_transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON retail_transactions(product_category);
CREATE INDEX IF NOT EXISTS idx_transactions_city ON retail_transactions(city);