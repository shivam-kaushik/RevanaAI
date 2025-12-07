import logging
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from collections import defaultdict, Counter
from statistics import mean
from openai import OpenAI
from backend.config import Config
from backend.utils.dataset_manager import DatasetManager
import json

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.db_manager = None
        # Use dataset manager to source the active table from DB
        self.dataset_manager = DatasetManager()
        
    def set_db_manager(self, db_manager):
        self.db_manager = db_manager

    def analyze_query(self, user_query):
        """Main method to analyze user query and generate response with data"""
        try:
            # Resolve active dataset from DB
            active_info = self.dataset_manager.get_active_dataset()
            active_table = active_info['table_name'] if isinstance(active_info, dict) and active_info.get('table_name') else None

            # Step 1: Generate SQL query using GPT
            sql_query = self.generate_sql_query(user_query, active_table)
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
            if not data or len(data) == 0:
                return {"error": "No data found for your query"}
            
            # Step 3: Generate insights using GPT
            insights = self.generate_insights(user_query, data)
            
            # Step 4: Generate charts if applicable
            charts = self.generate_charts(user_query, data)
            
            return {
                "success": True,
                "sql_query": sql_query,
                "data": data,
                "insights": insights,
                "charts": charts,
                "row_count": len(data)
            }
            
        except Exception as e:
            logger.error(f"Data analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def generate_sql_query(self, user_query, active_table: str | None = None):
        """Use GPT to generate SQL query from natural language"""
        try:
            # First, get database schema
            schema_info = self.get_database_schema(active_table)
            
            # Build a system prompt that explicitly pins to the active table if available
            active_table_clause = f"\nCurrent Active Table: {active_table}\n\nRules:\n- ALWAYS use the table name exactly as shown above.\n" if active_table else "\nRules:\n- Use the most recent Revana table available.\n"

            system_prompt = f"""
            You are an expert SQL query generator. Convert natural language questions into PostgreSQL SQL queries.
            
            Database Schema:
            {schema_info}
            
            {active_table_clause}
            Additional rules:
            1. Only generate SELECT queries
            2. Use proper PostgreSQL syntax
            3. Include appropriate WHERE clauses for filtering
            4. Use aggregate functions (COUNT, SUM, AVG, MAX, MIN) when needed
            5. Include GROUP BY for categorical analysis
            6. Use ORDER BY for sorting when relevant
            7. Return only the SQL query, no explanations
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
    
    def get_database_schema(self, preferred_table: str | None = None):
        """Get information about database tables and columns"""
        try:
            if not self.db_manager:
                return "Database connection not available"
            
            # Choose the active table if provided; else fall back to the latest revana table
            tables = self.db_manager.get_active_tables()
            if not tables and not preferred_table:
                return "No tables found in database"
            
            target_table = preferred_table if preferred_table else tables[0]
            
            # Get column information
            schema_query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{target_table}' 
                ORDER BY ordinal_position
            """
            
            columns_info = self.db_manager.execute_query_dict(schema_query)
            schema_description = f"Table: {target_table}\nColumns:\n"
            for row in columns_info:
                schema_description += f"- {row['column_name']} ({row['data_type']})\n"

            # Get sample data to understand content
            sample_query = f"SELECT * FROM {target_table} LIMIT 3"
            sample_rows = self.db_manager.execute_query_dict(sample_query)
            schema_description += f"\nSample data structure:\n{self._rows_to_table(sample_rows)}"
            
            return schema_description
            
        except Exception as e:
            logger.error(f"Schema retrieval error: {e}")
            return "Unable to retrieve schema"
    
    def execute_sql_query(self, sql_query):
        """Execute the generated SQL query and return list[dict]"""
        try:
            if not self.db_manager:
                raise Exception("Database manager not available")

            # Safety check - only allow SELECT queries
            if not sql_query.strip().upper().startswith('SELECT'):
                raise Exception("Only SELECT queries are allowed")

            return self.db_manager.execute_query_dict(sql_query)

        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise e
    
    def generate_insights(self, user_query, rows):
        """Use GPT to generate insights from the data (rows: list[dict])"""
        try:
            # Prepare data summary for GPT
            columns = list(rows[0].keys()) if rows else []
            sample_text = self._rows_to_table(rows[:10])
            numeric_summaries = self._numeric_summary(rows)

            data_summary = f"""
            User Query: {user_query}
            Data Retrieved: {len(rows)} rows
            Columns: {columns}
            Sample Data:
            {sample_text}
            
            Numeric Summary (means over detected numeric columns):
            {json.dumps(numeric_summaries)}
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
    
    def generate_charts(self, user_query, rows):
        """Generate charts based on the data and query type (rows: list[dict])"""
        try:
            charts = {}
            
            # Determine chart type based on query and data
            if self.should_generate_trend_chart(user_query, rows):
                charts['trend'] = self.create_trend_chart(rows)
            
            if self.should_generate_bar_chart(user_query, rows):
                charts['bar'] = self.create_bar_chart(rows)
            
            if self.should_generate_pie_chart(user_query, rows):
                charts['pie'] = self.create_pie_chart(rows)
            
            if self.should_generate_distribution_chart(user_query, rows):
                charts['distribution'] = self.create_distribution_chart(rows)
            
            return charts
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return {}
    
    def should_generate_trend_chart(self, user_query, rows):
        """Check if trend chart is appropriate"""
        time_keywords = ['line', 'line chart', 'trend', 'over time', 'monthly', 'weekly', 'daily', 'growth']
        if not rows:
            return False
        date_columns = [col for col in rows[0].keys() if 'date' in col.lower()]
        return any(keyword in user_query.lower() for keyword in time_keywords) and len(date_columns) > 0
    
    def should_generate_bar_chart(self, user_query, rows):
        """Check if bar chart is appropriate"""
        bar_keywords = ['bar', 'bar chart', 'compare', 'by category', 'top', 'ranking', 'by']
        if not rows:
            return False
        categorical_columns = [k for k, v in rows[0].items() if isinstance(v, str)]
        numerical_columns = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        return (any(keyword in user_query.lower() for keyword in bar_keywords) and 
                len(categorical_columns) > 0 and len(numerical_columns) > 0)
    
    def should_generate_pie_chart(self, user_query, rows):
        """Check if pie chart is appropriate"""
        pie_keywords = ['pie', 'pie chart', 'percentage', 'share', 'distribution', 'proportion']
        return any(keyword in user_query.lower() for keyword in pie_keywords)
    
    def should_generate_distribution_chart(self, user_query, rows):
        """Check if distribution chart is appropriate"""
        dist_keywords = ['distribution', 'histogram', 'spread', 'range']
        if not rows:
            return False
        numerical_columns = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        return any(keyword in user_query.lower() for keyword in dist_keywords) and len(numerical_columns) > 0
    
    def create_trend_chart(self, rows):
        """Create a line chart for trends over time (rows: list[dict])"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Find date column and value column
            date_col = next((col for col in rows[0].keys() if 'date' in col.lower()), None)
            value_col = self._first_numeric_column(rows)
            
            logger.info(f"📊 Trend chart - date_col: {date_col}, value_col: {value_col}, rows: {len(rows)}")
            
            if date_col and value_col:
                # Group by date and plot
                series = defaultdict(float)
                parsed_count = 0
                for r in rows:
                    date_val = self._parse_date(r.get(date_col))
                    num_val = self._to_number(r.get(value_col))
                    if date_val is not None and num_val is not None:
                        series[date_val] += num_val
                        parsed_count += 1
                
                logger.info(f"📊 Parsed {parsed_count} data points successfully")
                
                if not series:
                    logger.warning(f"⚠️ No data parsed! Sample row: {rows[0] if rows else 'empty'}")
                    return None
                
                xs = sorted(series.keys())
                ys = [series[x] for x in xs]
                
                logger.info(f"📊 Plotting {len(xs)} points")
                
                plt.plot(xs, ys, marker='o', linewidth=2, markersize=4)
                plt.title(f'Trend of {value_col} over time', fontsize=14, fontweight='bold')
                plt.xlabel(date_col, fontsize=12)
                plt.ylabel(value_col, fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                return self.plot_to_base64()
                
        except Exception as e:
            logger.error(f"❌ Trend chart error: {e}")
            import traceback
            traceback.print_exc()
        return None
    
    def create_bar_chart(self, rows):
        """Create a bar chart for categorical comparisons (rows: list[dict])"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Find categorical and numerical columns
            if not rows:
                return None
            cat_col = next((k for k, v in rows[0].items() if isinstance(v, str)), None)
            num_col = self._first_numeric_column(rows)
            
            if cat_col and num_col:
                agg = defaultdict(float)
                for r in rows:
                    k = r.get(cat_col)
                    v = self._to_number(r.get(num_col))
                    if isinstance(k, str) and v is not None:
                        agg[k] += v
                # Top 10
                items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:10]
                labels = [k for k, _ in items]
                values = [v for _, v in items]
                plt.bar(labels, values)
                plt.title(f'{num_col} by {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                return self.plot_to_base64()
                
        except Exception as e:
            logger.error(f"Bar chart error: {e}")
        return None
    
    def create_pie_chart(self, rows):
        """Create a pie chart for proportions (rows: list[dict])"""
        try:
            plt.figure(figsize=(8, 8))
            
            if not rows:
                return None
            cat_col = next((k for k, v in rows[0].items() if isinstance(v, str)), None)
            num_col = self._first_numeric_column(rows)
            
            if cat_col and num_col:
                agg = defaultdict(float)
                for r in rows:
                    k = r.get(cat_col)
                    v = self._to_number(r.get(num_col))
                    if isinstance(k, str) and v is not None:
                        agg[k] += v
                items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:8]
                labels = [k for k, _ in items]
                values = [v for _, v in items]
                if values and sum(values) > 0:
                    plt.pie(values, labels=labels, autopct='%1.1f%%')
                    plt.title(f'Distribution of {num_col} by {cat_col}')
                    return self.plot_to_base64()
                
        except Exception as e:
            logger.error(f"Pie chart error: {e}")
        return None
    
    def create_distribution_chart(self, rows):
        """Create a histogram for distribution (rows: list[dict])"""
        try:
            plt.figure(figsize=(10, 6))
            
            num_col = self._first_numeric_column(rows)
            if num_col:
                values = [self._to_number(r.get(num_col)) for r in rows]
                values = [v for v in values if v is not None]
                if values:
                    plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
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

    # Helper utilities (no pandas)
    def _rows_to_table(self, rows):
        if not rows:
            return "<empty>"
        # Build a simple aligned table string from list of dicts
        cols = list(rows[0].keys())
        col_widths = {c: max(len(str(c)), max((len(str(r.get(c, ''))) for r in rows), default=0)) for c in cols}
        header = " | ".join(str(c).ljust(col_widths[c]) for c in cols)
        sep = "-+-".join('-' * col_widths[c] for c in cols)
        lines = [header, sep]
        for r in rows:
            lines.append(" | ".join(str(r.get(c, '')).ljust(col_widths[c]) for c in cols))
        return "\n".join(lines)

    def _first_numeric_column(self, rows):
        if not rows:
            return None
        
        # Look for common numeric column names first
        common_numeric = ['sum', 'total', 'amount', 'count', 'value', 'sales', 'revenue', 'quantity']
        for name in common_numeric:
            for k in rows[0].keys():
                if name in k.lower():
                    return k
        
        # Then check by type
        for k, v in rows[0].items():
            if isinstance(v, (int, float)):
                return k
        
        # Try to coerce numeric-looking strings
        for k, v in rows[0].items():
            if isinstance(v, str):
                try:
                    float(v)
                    return k
                except Exception:
                    continue
        return None

    def _to_number(self, v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except Exception:
                return None
        return None

    def _parse_date(self, v):
        from datetime import date
        
        if isinstance(v, datetime):
            return v
        if isinstance(v, date):  # Handle date objects from database
            return datetime.combine(v, datetime.min.time())
        if isinstance(v, str):
            # Try common date formats
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(v, fmt)
                except Exception:
                    continue
        return None

    def _numeric_summary(self, rows):
        """Compute basic numeric statistics for numeric columns (replaces pandas describe)"""
        if not rows:
            return {}
        
        summary = {}
        numeric_cols = []
        
        # Find numeric columns
        for k, v in rows[0].items():
            if isinstance(v, (int, float)) or (isinstance(v, str) and self._to_number(v) is not None):
                numeric_cols.append(k)
        
        for col in numeric_cols:
            values = []
            for row in rows:
                val = self._to_number(row.get(col))
                if val is not None:
                    values.append(val)
            
            if values:
                summary[col] = {
                    "count": len(values),
                    "mean": mean(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary