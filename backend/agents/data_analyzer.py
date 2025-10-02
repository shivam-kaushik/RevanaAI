import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from openai import OpenAI
from backend.config import Config
import json

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.db_manager = None
        
    def set_db_manager(self, db_manager):
        self.db_manager = db_manager

    def analyze_query(self, user_query):
        """Main method to analyze user query and generate response with data"""
        try:
            # Step 1: Generate SQL query using GPT
            sql_query = self.generate_sql_query(user_query)
            if not sql_query:
                return {"error": "Could not generate SQL query for your request"}
            
            # Patch: Cast date column if needed
            if "DATE_TRUNC('month', date)" in sql_query:
                sql_query = sql_query.replace("DATE_TRUNC('month', date)", "DATE_TRUNC('month', date::DATE)")
            if "DATE_TRUNC('week', date)" in sql_query:
                sql_query = sql_query.replace("DATE_TRUNC('week', date)", "DATE_TRUNC('week', date::DATE)")
            if "DATE_TRUNC('day', date)" in sql_query:
                sql_query = sql_query.replace("DATE_TRUNC('day', date)", "DATE_TRUNC('day', date::DATE)")
            
            # Step 2: Execute SQL query
            data = self.execute_sql_query(sql_query)
            if data.empty:
                return {"error": "No data found for your query"}
            
            # # Step 3: Generate insights using GPT
            # insights = self.generate_insights(user_query, data)
            
            # # Step 4: Generate charts if applicable
            # charts = self.generate_charts(user_query, data)
            
            return {
                "success": True,
                "sql_query": sql_query,
                "data": data.to_dict('records'),
                # "insights": insights,
                # "charts": charts,
                "row_count": len(data)
            }
            
        except Exception as e:
            logger.error(f"Data analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def generate_sql_query(self, user_query):
        """Use GPT to generate SQL query from natural language"""
        try:
            # First, get database schema
            schema_info = self.get_database_schema()
            
            system_prompt = f"""
            You are an expert SQL query generator. Convert natural language questions into PostgreSQL SQL queries.
            
            Database Schema:
            {schema_info}
            
            Rules:
            1. Only generate SELECT queries
            2. Use proper PostgreSQL syntax
            3. Include appropriate WHERE clauses for filtering
            4. Use aggregate functions (COUNT, SUM, AVG, MAX, MIN) when needed
            5. Include GROUP BY for categorical analysis
            6. Use ORDER BY for sorting when relevant
            7. Return only the SQL query, no explanations
            8. Use the most recent table (starts with 'revana_')
            
            Examples:
            - "Show me total sales by month" → "SELECT DATE_TRUNC('month', date) as month, SUM(total_amount) as total_sales FROM revana_retail_sales_dataset_20250930_114844 GROUP BY month ORDER BY month"
            - "What are the top 5 products by revenue?" → "SELECT product_category, SUM(total_amount) as revenue FROM revana_retail_sales_dataset_20250930_114844 GROUP BY product_category ORDER BY revenue DESC LIMIT 5"
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate SQL query for: {user_query}"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean the SQL query (remove markdown code blocks if any)
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            logger.info(f"Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return None
    
    def get_database_schema(self):
        """Get information about database tables and columns"""
        try:
            if not self.db_manager:
                return "Database connection not available"
            
            # Get the latest table
            tables = self.db_manager.get_active_tables()
            if not tables:
                return "No tables found in database"
            
            latest_table = tables[0]  # Most recent table
            
            # Get column information
            schema_query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{latest_table}' 
                ORDER BY ordinal_position
            """
            
            columns_info = self.db_manager.execute_query(schema_query)
            schema_description = f"Table: {latest_table}\nColumns:\n"
            
            for _, row in columns_info.iterrows():
                schema_description += f"- {row['column_name']} ({row['data_type']})\n"
            
            # Get sample data to understand content
            sample_query = f"SELECT * FROM {latest_table} LIMIT 3"
            sample_data = self.db_manager.execute_query(sample_query)
            
            schema_description += f"\nSample data structure:\n{sample_data.to_string()}"
            
            return schema_description
            
        except Exception as e:
            logger.error(f"Schema retrieval error: {e}")
            return "Unable to retrieve schema"
    
    def execute_sql_query(self, sql_query):
        """Execute the generated SQL query"""
        try:
            if not self.db_manager:
                raise Exception("Database manager not available")
            
            # Safety check - only allow SELECT queries
            if not sql_query.strip().upper().startswith('SELECT'):
                raise Exception("Only SELECT queries are allowed")
            
            return self.db_manager.execute_query(sql_query)
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise e
    
    def generate_insights(self, user_query, data):
        """Use GPT to generate insights from the data"""
        try:
            # Prepare data summary for GPT
            data_summary = f"""
            User Query: {user_query}
            Data Retrieved: {len(data)} rows
            Columns: {list(data.columns)}
            Sample Data:
            {data.head(10).to_string()}
            
            Data Summary:
            {data.describe().to_string() if len(data.select_dtypes(include=['number']).columns) > 0 else 'No numerical columns to describe'}
            """
            
            system_prompt = """
            You are a data analyst. Provide clear, actionable insights based on the data.
            Focus on:
            1. Key trends and patterns
            2. Notable statistics
            3. Business implications
            4. Recommendations if applicable
            
            Be concise but informative. Use bullet points for clarity.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": data_summary}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Insight generation error: {e}")
            return "Unable to generate insights at this time."
    
    def generate_charts(self, user_query, data):
        """Generate charts based on the data and query type"""
        try:
            charts = {}
            
            # Determine chart type based on query and data
            if self.should_generate_trend_chart(user_query, data):
                charts['trend'] = self.create_trend_chart(data)
            
            if self.should_generate_bar_chart(user_query, data):
                charts['bar'] = self.create_bar_chart(data)
            
            if self.should_generate_pie_chart(user_query, data):
                charts['pie'] = self.create_pie_chart(data)
            
            if self.should_generate_distribution_chart(user_query, data):
                charts['distribution'] = self.create_distribution_chart(data)
            
            return charts
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return {}
    
    def should_generate_trend_chart(self, user_query, data):
        """Check if trend chart is appropriate"""
        time_keywords = ['trend', 'over time', 'monthly', 'weekly', 'daily', 'growth']
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        return any(keyword in user_query.lower() for keyword in time_keywords) and date_columns
    
    def should_generate_bar_chart(self, user_query, data):
        """Check if bar chart is appropriate"""
        bar_keywords = ['compare', 'by category', 'top', 'ranking', 'by']
        categorical_columns = data.select_dtypes(include=['object']).columns
        numerical_columns = data.select_dtypes(include=['number']).columns
        return (any(keyword in user_query.lower() for keyword in bar_keywords) and 
                len(categorical_columns) > 0 and len(numerical_columns) > 0)
    
    def should_generate_pie_chart(self, user_query, data):
        """Check if pie chart is appropriate"""
        pie_keywords = ['percentage', 'share', 'distribution', 'proportion']
        return any(keyword in user_query.lower() for keyword in pie_keywords)
    
    def should_generate_distribution_chart(self, user_query, data):
        """Check if distribution chart is appropriate"""
        dist_keywords = ['distribution', 'histogram', 'spread', 'range']
        numerical_columns = data.select_dtypes(include=['number']).columns
        return any(keyword in user_query.lower() for keyword in dist_keywords) and len(numerical_columns) > 0
    
    def create_trend_chart(self, data):
        """Create a line chart for trends over time"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Find date column and value column
            date_col = next((col for col in data.columns if 'date' in col.lower()), None)
            value_col = next((col for col in data.columns if data[col].dtype in ['int64', 'float64']), None)
            
            if date_col and value_col:
                # Convert date if needed
                if data[date_col].dtype == 'object':
                    data[date_col] = pd.to_datetime(data[date_col])
                
                # Group by date and plot
                trend_data = data.groupby(date_col)[value_col].sum().reset_index()
                plt.plot(trend_data[date_col], trend_data[value_col], marker='o', linewidth=2)
                plt.title(f'Trend of {value_col} over time')
                plt.xlabel(date_col)
                plt.ylabel(value_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                return self.plot_to_base64()
                
        except Exception as e:
            logger.error(f"Trend chart error: {e}")
        return None
    
    def create_bar_chart(self, data):
        """Create a bar chart for categorical comparisons"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Find categorical and numerical columns
            cat_cols = data.select_dtypes(include=['object']).columns
            num_cols = data.select_dtypes(include=['number']).columns
            
            if len(cat_cols) > 0 and len(num_cols) > 0:
                cat_col = cat_cols[0]
                num_col = num_cols[0]
                
                # Group and plot
                bar_data = data.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
                bar_data.plot(kind='bar')
                plt.title(f'{num_col} by {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                return self.plot_to_base64()
                
        except Exception as e:
            logger.error(f"Bar chart error: {e}")
        return None
    
    def create_pie_chart(self, data):
        """Create a pie chart for proportions"""
        try:
            plt.figure(figsize=(8, 8))
            
            cat_cols = data.select_dtypes(include=['object']).columns
            num_cols = data.select_dtypes(include=['number']).columns
            
            if len(cat_cols) > 0 and len(num_cols) > 0:
                cat_col = cat_cols[0]
                num_col = num_cols[0]
                
                pie_data = data.groupby(cat_col)[num_col].sum().head(8)
                plt.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
                plt.title(f'Distribution of {num_col} by {cat_col}')
                
                return self.plot_to_base64()
                
        except Exception as e:
            logger.error(f"Pie chart error: {e}")
        return None
    
    def create_distribution_chart(self, data):
        """Create a histogram for distribution"""
        try:
            plt.figure(figsize=(10, 6))
            
            num_cols = data.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                num_col = num_cols[0]
                plt.hist(data[num_col].dropna(), bins=20, alpha=0.7, edgecolor='black')
                plt.title(f'Distribution of {num_col}')
                plt.xlabel(num_col)
                plt.ylabel('Frequency')
                plt.tight_layout()
                
                return self.plot_to_base64()
                
        except Exception as e:
            logger.error(f"Distribution chart error: {e}")
        return None
    
    def plot_to_base64(self):
        """Convert matplotlib plot to base64 string"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return image_base64