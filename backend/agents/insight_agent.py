import logging
from openai import OpenAI
from backend.config import Config
import json

logger = logging.getLogger(__name__)

class InsightAgent:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    def generate_insights(self, user_query, data_rows, context=""):
        """Generate insights and narratives from data (data_rows: list[dict])"""
        try:
            logger.info(f"🤖 INSIGHT_AGENT: Generating insights for query: {user_query}")
            
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
            logger.info(f"✅ INSIGHT_AGENT: Insights generated successfully")
            return insights
            
        except Exception as e:
            logger.error(f"❌ INSIGHT_AGENT Error: {e}")
            return "I analyzed the data but couldn't generate specific insights at this time."
    
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
    
    def _to_number(self, v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except Exception:
                return None
        return None