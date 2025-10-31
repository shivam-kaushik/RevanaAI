import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Password1!@localhost:5432/Revana")
    
    # OpenAI
    OPENAI_API_KEY = "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"
    # Application
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Vector DB
    VECTOR_DB_PATH = "./data/vector_db"
    
    # Data paths
    DATA_PATH = "./data"