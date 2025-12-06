import os
from dotenv import load_dotenv
USE_QWEN_MODEL = True  # toggle on/off the embedded Text2SQL model

load_dotenv()


class Config:
    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://postgres:Password1!@localhost:5432/Revana")

    # OpenAI
    OPENAI_API_KEY = ""  # Default for testing
    # Application
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

    # Vector DB
    VECTOR_DB_PATH = "./data/vector_db"

    # Data paths
    DATA_PATH = "./data"
