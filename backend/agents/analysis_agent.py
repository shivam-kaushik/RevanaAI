import logging
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime, date
from collections import defaultdict
from statistics import mean
from openai import OpenAI
from backend.config import Config

logger = logging.getLogger(__name__)

class AnalysisAgent:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    def generate_insights(self, user_query, data_rows, context=""):
        """Generate insights and narratives from data (data_rows: list[dict])"""
        try:
            logger.info(f"🤖 ANALYSIS_AGENT: Generating insights for query: {user_query}")
            
            # Prepare data context
            data_context = self._prepare_data_context(data_rows)
            
            system_prompt = """
            You are a data analyst providing insights about retail sales data.
            Create clear, concise, and actionable insights based on the data provided.
            
            Guidelines:
            - Focus on key trends and patterns
            - Provide business-relevant insights
            - Be specific and data-driven
            - Use simple, clear language
            - Highlight important findings
            - Suggest potential actions when relevant
            - Use bullet points for clarity
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
                    User Question: {user_query}
                    
                    {data_context}
                    
                    Additional Context: {context}
                    
                    Please provide insightful analysis:
                    """}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            insights = response.choices[0].message.content.strip()
            logger.info(f"✅ ANALYSIS_AGENT: Insights generated successfully")
            return insights
            
        except Exception as e:
            logger.error(f"❌ ANALYSIS_AGENT Error: {e}")
            return "I analyzed the data but couldn't generate specific insights at this time."
    
    def create_visualization(self, user_query, data_rows, chart_type=None):
        """Create visualization based on user intent and data"""
        try:
            logger.info(f"🎨 ANALYSIS_AGENT: Creating {chart_type or 'auto-detected'} chart")
            
            if not data_rows:
                return None, "No data available for visualization"
            
            # Auto-detect chart type if not specified
            if not chart_type:
                chart_type = self._detect_chart_type(user_query, data_rows)
            
            if chart_type == "bar":
                return self._create_bar_chart(data_rows, user_query)
            elif chart_type == "pie":
                return self._create_pie_chart(data_rows, user_query)
            elif chart_type == "line":
                return self._create_line_chart(data_rows, user_query)
            elif chart_type == "histogram":
                return self._create_histogram(data_rows, user_query)
            else:
                return None, f"Unsupported chart type: {chart_type}"
                
        except Exception as e:
            logger.error(f"❌ ANALYSIS_AGENT Error: {e}")
            return None, f"Error creating visualization: {str(e)}"
    
    def _detect_chart_type(self, user_query, data_rows):
        """Detect the most appropriate chart type based on query and data"""
        query_lower = user_query.lower()
        
        # Explicit chart type requests
        if any(word in query_lower for word in ['bar chart', 'bar graph', 'column chart']):
            return "bar"
        elif any(word in query_lower for word in ['pie chart', 'pie graph', 'donut chart']):
            return "pie"
        elif any(word in query_lower for word in ['line chart', 'line graph', 'trend', 'over time']):
            return "line"
        elif any(word in query_lower for word in ['histogram', 'distribution', 'frequency']):
            return "histogram"
        
        # Auto-detect based on data characteristics
        if not data_rows:
            return "bar"  # default
        
        # Check for time series data using improved date detection
        date_col = self._find_date_column(data_rows)
        if date_col:
            return "line"
        
        # Check for categorical vs numerical data
        categorical_cols = [k for k, v in data_rows[0].items() if isinstance(v, str)]
        numerical_cols = [k for k, v in data_rows[0].items() if isinstance(v, (int, float))]
        
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            # If few categories, use pie; if many, use bar
            unique_categories = len(set(row.get(categorical_cols[0], '') for row in data_rows))
            return "pie" if unique_categories <= 8 else "bar"
        elif len(numerical_cols) > 0:
            return "histogram"
        
        return "bar"  # default fallback
    
    def _create_bar_chart(self, data_rows, user_query):
        """Create a bar chart"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Find categorical and numerical columns
            cat_col = next((k for k, v in data_rows[0].items() if isinstance(v, str)), None)
            num_col = self._first_numeric_column(data_rows)
            
            if not cat_col or not num_col:
                return None, "Bar chart requires both categorical and numerical data"
            
            # Aggregate data
            agg = defaultdict(float)
            for row in data_rows:
                k = row.get(cat_col)
                v = self._to_number(row.get(num_col))
                if isinstance(k, str) and v is not None:
                    agg[k] += v
            
            # Sort and take top 10
            items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:10]
            labels = [k for k, _ in items]
            values = [v for _, v in items]
            
            plt.bar(labels, values, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title(f'{num_col} by {cat_col}', fontsize=14, fontweight='bold')
            plt.xlabel(cat_col, fontsize=12)
            plt.ylabel(num_col, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return self._plot_to_base64(), None
            
        except Exception as e:
            logger.error(f"Bar chart error: {e}")
            return None, f"Error creating bar chart: {str(e)}"
    
    def _create_pie_chart(self, data_rows, user_query):
        """Create a pie chart"""
        try:
            plt.figure(figsize=(10, 8))
            
            cat_col = next((k for k, v in data_rows[0].items() if isinstance(v, str)), None)
            num_col = self._first_numeric_column(data_rows)
            
            if not cat_col or not num_col:
                return None, "Pie chart requires both categorical and numerical data"
            
            # Aggregate data
            agg = defaultdict(float)
            for row in data_rows:
                k = row.get(cat_col)
                v = self._to_number(row.get(num_col))
                if isinstance(k, str) and v is not None:
                    agg[k] += v
            
            # Take top 8 categories
            items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:8]
            labels = [k for k, _ in items]
            values = [v for _, v in items]
            
            if values and sum(values) > 0:
                colors = plt.cm.Set3(range(len(labels)))
                plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                plt.title(f'Distribution of {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                plt.axis('equal')
                
                return self._plot_to_base64(), None
            else:
                return None, "No valid data for pie chart"
                
        except Exception as e:
            logger.error(f"Pie chart error: {e}")
            return None, f"Error creating pie chart: {str(e)}"
    
    def _create_line_chart(self, data_rows, user_query):
        """Create a line chart for trends"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Find date column and value column
            date_col = self._find_date_column(data_rows)
            value_col = self._first_numeric_column(data_rows)
            
            if not date_col or not value_col:
                return None, "Line chart requires date and numerical data"
            
            # Group by date and plot
            series = defaultdict(float)
            logger.info(f"Creating line chart with date_col: {date_col}, value_col: {value_col}")
            
            for i, row in enumerate(data_rows):
                raw_date = row.get(date_col)
                raw_num = row.get(value_col)
                date_val = self._parse_date(raw_date)
                num_val = self._to_number(raw_num)
                logger.debug(f"Row {i}: raw_date={raw_date} (type: {type(raw_date)}), raw_num={raw_num} (type: {type(raw_num)})")
                logger.debug(f"Row {i}: parsed_date={date_val}, parsed_num={num_val}")
                if date_val is not None and num_val is not None:
                    series[date_val] += num_val
            
            logger.info(f"Series data points: {len(series)}")
            if series:
                xs = sorted(series.keys())
                ys = [series[x] for x in xs]
                
                plt.plot(xs, ys, marker='o', linewidth=2, markersize=6, color='blue')
                plt.title(f'Trend of {value_col} over time', fontsize=14, fontweight='bold')
                plt.xlabel(date_col, fontsize=12)
                plt.ylabel(value_col, fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                return self._plot_to_base64(), None
            else:
                return None, "No valid time series data for line chart"
                
        except Exception as e:
            logger.error(f"Line chart error: {e}")
            return None, f"Error creating line chart: {str(e)}"
    
    def _create_histogram(self, data_rows, user_query):
        """Create a histogram for distribution"""
        try:
            plt.figure(figsize=(10, 6))
            
            num_col = self._first_numeric_column(data_rows)
            if not num_col:
                return None, "Histogram requires numerical data"
            
            values = [self._to_number(row.get(num_col)) for row in data_rows]
            values = [v for v in values if v is not None]
            
            if values:
                plt.hist(values, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
                plt.title(f'Distribution of {num_col}', fontsize=14, fontweight='bold')
                plt.xlabel(num_col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                return self._plot_to_base64(), None
            else:
                return None, "No valid numerical data for histogram"
                
        except Exception as e:
            logger.error(f"Histogram error: {e}")
            return None, f"Error creating histogram: {str(e)}"
    
    def _plot_to_base64(self):
        """Convert matplotlib plot to base64 string"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return image_base64
    
    def _prepare_data_context(self, data_rows):
        """Prepare data context for insights generation"""
        if not data_rows:
            return "No data available for analysis."
        
        # Get basic statistics
        columns = list(data_rows[0].keys())
        row_count = len(data_rows)
        
        # Get sample data (first 5 rows)
        sample_data = self._rows_to_table(data_rows[:5])
        
        # Get numeric summaries
        numeric_summaries = self._numeric_summary(data_rows)
        
        return f"""
        Data Summary:
        - Total Records: {row_count}
        - Columns: {', '.join(columns)}
        
        Sample Data:
        {sample_data}
        
        Numeric Statistics:
        {json.dumps(numeric_summaries, indent=2)}
        """
    
    def _rows_to_table(self, rows):
        """Format list of dicts as a simple table string"""
        if not rows:
            return "<empty>"
        
        cols = list(rows[0].keys())
        col_widths = {c: max(len(str(c)), max((len(str(r.get(c, ''))) for r in rows), default=0)) for c in cols}
        header = " | ".join(str(c).ljust(col_widths[c]) for c in cols)
        sep = "-+-".join('-' * col_widths[c] for c in cols)
        lines = [header, sep]
        for r in rows:
            lines.append(" | ".join(str(r.get(c, '')).ljust(col_widths[c]) for c in cols))
        return "\n".join(lines)
    
    def _numeric_summary(self, rows):
        """Compute basic numeric statistics"""
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
                    "mean": round(sum(values) / len(values), 2),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
    
    # Helper methods
    def _first_numeric_column(self, rows):
        if not rows:
            return None
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
    
    def _find_date_column(self, data_rows):
        """Find the date column in the data"""
        if not data_rows:
            return None
        
        # First, look for columns with 'date' in the name
        for col in data_rows[0].keys():
            if 'date' in col.lower():
                return col
        
        # If no obvious date column, try to detect by content
        for col in data_rows[0].keys():
            sample_value = data_rows[0].get(col)
            if self._parse_date(sample_value) is not None:
                return col
        
        return None
    
    def _parse_date(self, v):
        if isinstance(v, datetime):
            return v
        if isinstance(v, date):
            return datetime(v.year, v.month, v.day)
        if isinstance(v, str):
            # Try common date formats including MM/DD/YYYY HH:MM and YYYY-MM-DD
            for fmt in ("%Y-%m-%d", "%m/%d/%Y %H:%M", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(v, fmt)
                except Exception:
                    continue
        # Handle other date-like objects from database
        if hasattr(v, 'date'):
            return v.date()
        if hasattr(v, 'year') and hasattr(v, 'month') and hasattr(v, 'day'):
            return datetime(v.year, v.month, v.day)
        return None
