from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API key")
    NLP_CLOUD_API_KEY: str = Field(..., description="NLP Cloud API key")
    
    NLP_CLOUD_URL: str = 'https://api.nlpcloud.io/v1'

    class Config:
        env_file = ".env"

settings = Settings()
