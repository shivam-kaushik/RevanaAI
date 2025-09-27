import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Password1!@localhost:5432/Revana")
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("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")
    
    # Application
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Vector DB
    VECTOR_DB_PATH = "./data/vector_db"
    
    # Data paths
    DATA_PATH = "./data"