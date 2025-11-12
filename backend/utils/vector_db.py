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
        # Clear any stale schema cache except the new table
        keys_to_remove = [k for k in self.schema_cache.keys() if k != table_name]
        for key in keys_to_remove:
            try:
                del self.schema_cache[key]
            except Exception:
                pass
        # Preload schema for the active table to ensure immediate correctness
        try:
            self._preload_schema_from_db(table_name)
        except Exception as e:
            logger.warning(f"Could not preload schema for {table_name}: {e}")
        logger.info(f"Active dataset set to: {table_name}")

    def _preload_schema_from_db(self, table_name):
        """Fetch table schema from DB and cache it for prompt context."""
        try:
            # Lazy import to avoid circular deps
            from backend.utils.database import db_manager
            query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """
            df = db_manager.execute_query(query, {"table_name": table_name})
            if df is not None and not df.empty:
                schema_docs = [f"{row['column_name']}: {row['data_type']}" for _, row in df.iterrows()]
                self.update_dataset_schema(table_name, schema_docs)
        except Exception as e:
            raise e
    
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