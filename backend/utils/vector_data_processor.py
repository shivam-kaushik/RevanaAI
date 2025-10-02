import logging
import pandas as pd
from typing import List, Dict
from backend.utils.vector_store import PostgresVectorStore

logger = logging.getLogger(__name__)

class VectorDataProcessor:
    def __init__(self):
        self.vector_store = PostgresVectorStore()
    
    def process_uploaded_data(self, df: pd.DataFrame, table_name: str):
        """Extract and store vector embeddings from uploaded data"""
        try:
            print(f"🔍 Vector Processing: Starting for table {table_name}")
            
            # Extract product information
            products_data = self.extract_product_data(df)
            if products_data:
                print(f"📦 Processing {len(products_data)} products for vector embeddings")
                self.vector_store.store_product_embeddings(products_data)
                logger.info(f"✅ Processed {len(products_data)} products for vector search")
            else:
                print("📦 No product data found for vector embeddings")
            
            # Extract customer information
            customers_data = self.extract_customer_data(df)
            if customers_data:
                print(f"👤 Processing {len(customers_data)} customers for vector embeddings")
                self.vector_store.store_customer_embeddings(customers_data)
                logger.info(f"✅ Processed {len(customers_data)} customers for vector search")
            else:
                print("👤 No customer data found for vector embeddings")
            
            print("✅ Vector processing completed successfully")
            # FIX: Don't return anything, just complete the processing
                    
        except Exception as e:
            logger.error(f"Error processing data for vector store: {e}")
            print(f"❌ Vector processing error: {e}")
            # Don't re-raise, just log the error so file upload continues
    
    def extract_product_data(self, df: pd.DataFrame) -> List[Dict]:
        """Extract product information from dataframe"""
        products_data = []
        
        try:
            # Check if we have product-related columns
            if 'product_category' in df.columns:
                # Get unique product categories
                unique_categories = df['product_category'].unique()
                
                for category in unique_categories:
                    # Get sample data from this category
                    category_data = df[df['product_category'] == category]
                    avg_price = category_data['price_per_unit'].mean() if 'price_per_unit' in df.columns else 0
                    total_sales = category_data['total_amount'].sum() if 'total_amount' in df.columns else 0
                    
                    product_info = {
                        'product_name': category,
                        'product_category': category,
                        'description': f"Products in {category} category with average price ${avg_price:.2f}",
                        'price': avg_price,
                        'total_sales': total_sales,
                        'transaction_count': len(category_data)
                    }
                    products_data.append(product_info)
            
            return products_data
            
        except Exception as e:
            logger.error(f"Error extracting product data: {e}")
            return []
    
    def extract_customer_data(self, df: pd.DataFrame) -> List[Dict]:
        """Extract customer information from dataframe"""
        customers_data = []
        
        try:
            # Check if we have customer-related columns
            if 'customer_id' in df.columns:
                # Group by customer to get purchase history
                customer_groups = df.groupby('customer_id')
                
                for customer_id, group in customer_groups:
                    purchase_count = len(group)
                    top_categories = group['product_category'].value_counts().head(3).index.tolist() if 'product_category' in group.columns else []
                    total_spent = group['total_amount'].sum() if 'total_amount' in group.columns else 0
                    avg_transaction = group['total_amount'].mean() if 'total_amount' in group.columns else 0
                    
                    customer_info = {
                        'customer_id': customer_id,
                        'age': group['age'].iloc[0] if 'age' in group.columns else None,
                        'gender': group['gender'].iloc[0] if 'gender' in group.columns else 'Unknown',
                        'purchase_history': f"Purchased {purchase_count} items",
                        'preferences': f"Prefers {', '.join(top_categories)}" if top_categories else "No preference data",
                        'total_spent': total_spent,
                        'avg_transaction': avg_transaction,
                        'total_transactions': purchase_count
                    }
                    customers_data.append(customer_info)
            
            return customers_data
            
        except Exception as e:
            logger.error(f"Error extracting customer data: {e}")
            return []