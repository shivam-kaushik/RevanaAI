import os
from dotenv import load_dotenv
USE_QWEN_MODEL = True  # toggle on/off the embedded Text2SQL model

load_dotenv()


class Config:
    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://postgres:Password1!@localhost:5432/Revana")

    # OpenAI
    OPENAI_API_KEY = "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"  # Default for testing
    # Application
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

    # Vector DB
    VECTOR_DB_PATH = "./data/vector_db"

    # Data paths
    DATA_PATH = "./data"
