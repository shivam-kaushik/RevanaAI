import logging
import json
from backend.config import Config

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self):
        self.active_dataset = None
        self.schema_cache = {}  # Simple in-memory cache for schemas
        logger.info("✅ Simple Vector DB Manager initialized")
    
    def update_dataset_schema(self, table_name, schema_docs):
        """Store schema in memory cache"""
        try:
            self.schema_cache[table_name] = schema_docs
            logger.info(f"✅ Schema cached for {table_name}: {len(schema_docs)} columns")
        except Exception as e:
            logger.error(f"Failed to cache schema: {e}")
    
    def set_active_dataset(self, table_name):
        """Set the active dataset for the session"""
        self.active_dataset = table_name
        logger.info(f"Active dataset set to: {table_name}")
    
    def get_active_dataset(self):
        """Get the currently active dataset"""
        return self.active_dataset
    
    def get_schema_context(self, user_query):
        """Get relevant schema information"""
        if not self.active_dataset:
            return "No active dataset. Please upload a CSV file first."
        
        if self.active_dataset not in self.schema_cache:
            return f"Active dataset: {self.active_dataset}. Schema information not available."
        
        schema_info = self.schema_cache[self.active_dataset]
        schema_text = "\n".join(schema_info)
        
        return f"""
        Active Dataset: {self.active_dataset}
        
        Schema Information:
        {schema_text}
        
        You can ask questions about this data using the available columns.
        """

# Singleton instance
vector_db = VectorDBManager()