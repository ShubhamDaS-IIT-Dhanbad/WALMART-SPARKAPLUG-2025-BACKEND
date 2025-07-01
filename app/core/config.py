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
        
        "https://chatbot.iitism.ac.in/",
        "https://chatbot.iitism.ac.in/chat",
        "https://ism-buddy-frontend.onrender.com/",
        "https://ism-buddy-frontend.onrender.com",
        "https://ism-buddy-frontend.onrender.com/chat",
        "https://ism-buddy-frontend.onrender.com/admin",
        
        
        "https://chat-bot-iit-ism-frontend-dbwo.vercel.app/",
        "https://chat-bot-iit-ism-frontend-dbwo.vercel.app",
        "https://chat-bot-iit-ism-frontend-dbwo.vercel.app/chat",
        "https://chat-bot-iit-ism-frontend-dbwo.vercel.app/admin",
        "https://chat-bot-iit-ism-frontend-dbwo.vercel.app",
        "https://chat-bot-iit-ism-frontend-dbwo.vercel.app/"
    ]
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    
    # API Keys
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    PINECONE_INDEX_NAME: str
    PINECONE_INDEX_URI: str
    
    #appwrite env variable
    APPWRITE_PROJECT_ID: str
    APPWRITE_DATABASE_ID: str
    APPWRITE_API_KEY: str
    APPWRITE_ENDPOINT:str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

settings = Settings()