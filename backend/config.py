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