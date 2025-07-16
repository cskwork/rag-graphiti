"""
Configuration management for RAG chatbot.
환경 변수와 설정 파일을 통한 설정 관리
"""

import os
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # FalkorDB 설정
    falkor_host: str = Field(default="localhost", description="FalkorDB host")
    falkor_port: str = Field(default="6379", description="FalkorDB port")
    falkor_username: Optional[str] = Field(default=None, description="FalkorDB username")
    falkor_password: Optional[SecretStr] = Field(default=None, description="FalkorDB password")
    
    # LLM 설정 (선택사항)
    openai_api_key: Optional[SecretStr] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    
    anthropic_api_key: Optional[SecretStr] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-3-haiku-20240307", description="Anthropic model")
    
    google_api_key: Optional[SecretStr] = Field(default=None, description="Google API key")
    google_model: str = Field(default="gemini-1.5-flash", description="Google model name")
    
    # 로깅 설정
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # 채팅 설정
    default_max_results: int = Field(default=5, description="Default max search results")
    default_chat_history_size: int = Field(default=10, description="Chat history size")
    
    # 웹 서버 설정
    web_host: str = Field(default="0.0.0.0", description="Web server host")
    web_port: int = Field(default=8000, description="Web server port")
    web_reload: bool = Field(default=False, description="Enable auto-reload")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


def setup_logging(settings: Settings) -> None:
    """Set up logging configuration."""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
    )