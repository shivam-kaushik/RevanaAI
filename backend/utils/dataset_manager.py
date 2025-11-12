import logging
import pandas as pd
from backend.utils.database import db_manager
from backend.utils.vector_db import vector_db

logger = logging.getLogger(__name__)

class DatasetManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern - ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super(DatasetManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize only once"""
        if not DatasetManager._initialized:
            self.active_dataset = None
            logger.info("📊 Dataset Manager initialized (Singleton)")
            DatasetManager._initialized = True
    
    def ensure_datasets_table(self):
        """Ensure the datasets table exists"""
        try:
            # Simple check to see if table exists
            db_manager.execute_query("SELECT 1 FROM revana_datasets LIMIT 0")
            logger.info("✅ revana_datasets table exists")
            return True
        except Exception as e:
            logger.warning(f"revana_datasets table may not exist or is inaccessible: {e}")
            return False
    
    def register_dataset(self, table_name, filename, row_count, column_count, description="", is_active=True):
        """Register a new dataset in the datasets table"""
        try:
            # If this should be active, first set all others to inactive
            if is_active:
                try:
                    db_manager.execute_non_query("UPDATE revana_datasets SET is_active = FALSE")
                except Exception as e:
                    logger.warning(f"Could not set other datasets inactive: {e}")

            # Check if dataset already exists
            try:
                check_query = "SELECT * FROM revana_datasets WHERE table_name = :table_name"
                existing = db_manager.execute_query(check_query, {"table_name": table_name})
            except Exception as e:
                logger.warning(f"Could not check existing dataset: {e}")
                existing = pd.DataFrame()

            if not existing.empty:
                # Update existing record
                query = """
                    UPDATE revana_datasets 
                    SET original_filename = :filename, 
                        row_count = :row_count, 
                        column_count = :column_count, 
                        description = :description,
                        is_active = :is_active,
                        created_at = CURRENT_TIMESTAMP
                    WHERE table_name = :table_name
                """
                db_manager.execute_non_query(query, {
                    "filename": filename,
                    "row_count": row_count,
                    "column_count": column_count,
                    "description": description,
                    "is_active": is_active,
                    "table_name": table_name
                })
            else:
                # Insert new record
                query = """
                    INSERT INTO revana_datasets 
                    (table_name, original_filename, row_count, column_count, description, is_active, created_at)
                    VALUES (:table_name, :filename, :row_count, :column_count, :description, :is_active, CURRENT_TIMESTAMP)
                """
                db_manager.execute_non_query(query, {
                    "table_name": table_name,
                    "filename": filename,
                    "row_count": row_count,
                    "column_count": column_count,
                    "description": description,
                    "is_active": is_active
                })

            logger.info(f"✅ Dataset registered: {table_name}")

            # Update active dataset cache
            if is_active:
                # Clear cache and reload the active dataset
                self.active_dataset = None
                active_result = db_manager.execute_query(
                    "SELECT * FROM revana_datasets WHERE table_name = :table_name",
                    {"table_name": table_name}
                )
                if not active_result.empty:
                    self.active_dataset = self._map_dataset_fields(active_result.iloc[0].to_dict())
                    logger.info(f"✅ Active dataset updated to: {table_name}")
                    vector_db.set_active_dataset(table_name)

            return True
        except Exception as e:
            logger.error(f"Failed to register dataset: {e}")
            return False
    
    def get_available_datasets(self):
        """Get all uploaded datasets with metadata"""
        try:
            query = """
                SELECT 
                    id,
                    table_name,
                    original_filename,
                    created_at as upload_timestamp,
                    row_count,
                    column_count,
                    description,
                    is_active
                FROM revana_datasets 
                ORDER BY created_at DESC
            """
            result = db_manager.execute_query(query)
            if result.empty:
                return []

            # Filter out entries whose underlying table no longer exists
            try:
                existing_tables = set(db_manager.get_active_tables())
                filtered = result[result['table_name'].isin(existing_tables)]
                return filtered.to_dict('records') if not filtered.empty else []
            except Exception:
                # Fallback to original list if table existence check fails
                return result.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to get datasets: {e}")
            return []
    
    def set_active_dataset(self, table_name):
        """Set a specific dataset as active"""
        try:
            logger.info(f"🔄 Setting active dataset to: {table_name}")
            
            # First, set all datasets to inactive
            inactive_result = db_manager.execute_non_query("UPDATE revana_datasets SET is_active = FALSE")
            logger.debug(f"Set all datasets inactive: {inactive_result}")

            # Then set the specified one as active
            active_result = db_manager.execute_non_query(
                "UPDATE revana_datasets SET is_active = TRUE WHERE table_name = :table_name",
                {"table_name": table_name}
            )
            logger.debug(f"Set {table_name} active: {active_result}")

            # CRITICAL: Clear the cache to force refresh
            self.active_dataset = None
            logger.info(f"🗑️ Cleared dataset cache")

            # Verify the update worked by querying the dataset
            dataset_info = db_manager.execute_query(
                "SELECT * FROM revana_datasets WHERE table_name = :table_name AND is_active = TRUE",
                {"table_name": table_name}
            )

            if not dataset_info.empty:
                self.active_dataset = self._map_dataset_fields(dataset_info.iloc[0].to_dict())
                logger.info(f"✅ Active dataset set to: {table_name}")
                
                # CRITICAL: Immediately sync vector_db and preload schema
                vector_db.set_active_dataset(table_name)
                logger.info(f"✅ Vector DB synced to: {table_name}")
                
                return True
            else:
                logger.error(f"❌ Dataset not found or not active after update: {table_name}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to set active dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_active_dataset(self, force_refresh=False):
        """Get the currently active dataset
        
        Args:
            force_refresh: If True, bypass cache and query database
        """
        try:
            # Check cache first (unless force refresh)
            if self.active_dataset and not force_refresh:
                return self.active_dataset
            
            # Query the database for active dataset
            query = "SELECT * FROM revana_datasets WHERE is_active = TRUE LIMIT 1"
            result = db_manager.execute_query(query)
            
            if not result.empty:
                self.active_dataset = self._map_dataset_fields(result.iloc[0].to_dict())
                vector_db.set_active_dataset(self.active_dataset['table_name'])
                logger.info(f"📊 Active dataset refreshed: {self.active_dataset['table_name']}")
                return self.active_dataset
            else:
                # Try to auto-set the latest dataset
                logger.info("No active dataset found, attempting auto-set...")
                return self.auto_set_latest_dataset()
                
        except Exception as e:
            logger.error(f"Failed to get active dataset: {e}")
            return None
    
    def _map_dataset_fields(self, dataset_info):
        """Map database fields to consistent field names"""
        return {
            'table_name': dataset_info.get('table_name'),
            'original_filename': dataset_info.get('original_filename'),
            'upload_timestamp': dataset_info.get('created_at'),  # Map created_at to upload_timestamp
            'row_count': dataset_info.get('row_count'),
            'column_count': dataset_info.get('column_count'),
            'description': dataset_info.get('description', ''),
            'is_active': dataset_info.get('is_active', False)
        }
    
    def auto_set_latest_dataset(self):
        """Automatically set the latest dataset as active"""
        try:
            datasets = self.get_available_datasets()
            if datasets:
                latest_dataset = datasets[0]
                success = self.set_active_dataset(latest_dataset['table_name'])
                if success:
                    logger.info(f"✅ Auto-set latest dataset: {latest_dataset['table_name']}")
                    return self.active_dataset
            
            logger.info("No datasets available to auto-set")
            return None
        except Exception as e:
            logger.error(f"Failed to auto-set latest dataset: {e}")
            return None
    
    def has_active_dataset(self):
        """Check if there's an active dataset"""
        return self.get_active_dataset() is not None
    
    def get_dataset_by_name(self, table_name):
        """Get a specific dataset by table name"""
        try:
            query = "SELECT * FROM revana_datasets WHERE table_name = :table_name"
            result = db_manager.execute_query(query, {"table_name": table_name})
            return self._map_dataset_fields(result.iloc[0].to_dict()) if not result.empty else None
        except Exception as e:
            logger.error(f"Failed to get dataset {table_name}: {e}")
            return None
