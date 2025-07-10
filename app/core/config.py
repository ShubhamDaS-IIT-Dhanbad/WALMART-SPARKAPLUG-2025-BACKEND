from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):

    PROJECT_NAME: str = "FastAPI Application"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "A modern FastAPI application with best practices"
    API_V1_STR: str = "/api/v1"
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        
    ]
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    
    # API Keys
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    PINECONE_INDEX_NAME: str
    PINECONE_INDEX_URI: str
    
    SCRAPY_DOG_API_KEY:str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

settings = Settings()