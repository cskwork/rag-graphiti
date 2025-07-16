"""
Configuration management for RAG chatbot.
환경 변수와 설정 파일을 통한 설정 관리
"""

import os
import logging
from typing import Optional
from pathlib import Path

from pydantic import Field, SecretStr, validator, root_validator
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
    
    # 고급 설정
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=100, description="Maximum cache entries")
    circuit_breaker_threshold: int = Field(default=5, description="Circuit breaker failure threshold")
    circuit_breaker_timeout: int = Field(default=60, description="Circuit breaker timeout in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('falkor_port')
    def validate_port(cls, v):
        """FalkorDB 포트 유효성 검증"""
        if isinstance(v, str):
            try:
                port_int = int(v)
            except ValueError:
                raise ValueError(f"Invalid port format: {v}")
        else:
            port_int = v
            
        if not 1 <= port_int <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {port_int}")
        return str(port_int)  # FalkorDB 포트는 문자열로 저장
    
    @validator('web_port')
    def validate_web_port(cls, v):
        """웹 서버 포트 유효성 검증"""
        if not 1 <= v <= 65535:
            raise ValueError(f"Web port must be between 1 and 65535, got {v}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """로그 레벨 유효성 검증"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}, got {v}")
        return v.upper()
    
    @validator('default_max_results')
    def validate_max_results(cls, v):
        """최대 결과 수 유효성 검증"""
        if v < 1:
            raise ValueError(f"Max results must be positive, got {v}")
        if v > 100:
            raise ValueError(f"Max results cannot exceed 100, got {v}")
        return v
    
    @validator('default_chat_history_size')
    def validate_chat_history_size(cls, v):
        """채팅 기록 크기 유효성 검증"""
        if v < 0:
            raise ValueError(f"Chat history size cannot be negative, got {v}")
        if v > 1000:
            raise ValueError(f"Chat history size cannot exceed 1000, got {v}")
        return v
    
    @validator('cache_ttl')
    def validate_cache_ttl(cls, v):
        """캐시 TTL 유효성 검증"""
        if v < 0:
            raise ValueError(f"Cache TTL cannot be negative, got {v}")
        if v > 3600:  # 1시간 제한
            raise ValueError(f"Cache TTL cannot exceed 3600 seconds, got {v}")
        return v
    
    @validator('cache_max_size')
    def validate_cache_max_size(cls, v):
        """캐시 최대 크기 유효성 검증"""
        if v < 1:
            raise ValueError(f"Cache max size must be positive, got {v}")
        if v > 10000:
            raise ValueError(f"Cache max size cannot exceed 10000, got {v}")
        return v
    
    @validator('web_host')
    def validate_web_host(cls, v):
        """웹 호스트 유효성 검증"""
        if not v or not isinstance(v, str):
            raise ValueError("Web host cannot be empty")
        # 간단한 IP/hostname 형식 검증
        if not (v == "0.0.0.0" or v == "localhost" or v.replace('.', '').replace('-', '').replace('_', '').isalnum()):
            raise ValueError(f"Invalid web host format: {v}")
        return v
    
    @root_validator
    def validate_settings_combination(cls, values):
        """설정 조합 유효성 검증"""
        # 웹 포트와 FalkorDB 포트가 같으면 안됨
        web_port = values.get('web_port')
        falkor_port = values.get('falkor_port')
        
        if web_port and falkor_port:
            try:
                falkor_port_int = int(falkor_port)
                if web_port == falkor_port_int:
                    raise ValueError("Web port and FalkorDB port cannot be the same")
            except ValueError:
                pass  # falkor_port가 숫자가 아닌 경우는 다른 validator에서 처리
        
        return values


def get_settings() -> Settings:
    """Get application settings instance with validation."""
    try:
        return Settings()
    except Exception as e:
        # 설정 로드 실패 시 기본 설정으로 폴백
        print(f"Warning: Failed to load settings: {e}")
        print("Using default settings")
        return Settings(
            falkor_host="localhost",
            falkor_port="6379",
            log_level="INFO",
            web_host="0.0.0.0",
            web_port=8000
        )


def validate_settings(settings: Settings) -> None:
    """Validate settings after loading."""
    # 중요한 설정 검증
    if not settings.falkor_host:
        raise ValueError("FalkorDB host cannot be empty")
    
    # 설정 조합 검증
    try:
        falkor_port_int = int(settings.falkor_port)
        if settings.web_port == falkor_port_int:
            raise ValueError("Web port and FalkorDB port cannot be the same")
    except ValueError as e:
        if "Web port and FalkorDB port" in str(e):
            raise
        # 포트 형식 오류는 이미 validator에서 처리됨
    
    # 로그 레벨 실제 존재 여부 확인
    if not hasattr(logging, settings.log_level):
        raise ValueError(f"Invalid log level: {settings.log_level}")


def setup_logging(settings: Settings) -> None:
    """Set up logging configuration with enhanced validation."""
    import logging
    
    # 로그 레벨 검증
    try:
        log_level = getattr(logging, settings.log_level.upper())
    except AttributeError:
        print(f"Warning: Invalid log level '{settings.log_level}', using INFO")
        log_level = logging.INFO
    
    # 로그 포맷 검증
    try:
        # 포맷 문자열이 유효한지 테스트
        test_record = logging.LogRecord(
            name="test", level=log_level, pathname="", lineno=0,
            msg="test", args=(), exc_info=None
        )
        settings.log_format % test_record.__dict__
    except (KeyError, ValueError, TypeError):
        print(f"Warning: Invalid log format, using default")
        settings.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 로깅 설정
    logging.basicConfig(
        level=log_level,
        format=settings.log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # 기존 핸들러 재설정
    )
    
    # 로깅 설정 완료 메시지
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {settings.log_level}")
    logger.debug(f"Settings validation passed")


def print_settings_summary(settings: Settings) -> None:
    """Print a summary of current settings (for debugging)."""
    print("\n=== RAG Chatbot Settings Summary ===")
    print(f"FalkorDB: {settings.falkor_host}:{settings.falkor_port}")
    print(f"Web Server: {settings.web_host}:{settings.web_port}")
    print(f"Log Level: {settings.log_level}")
    print(f"Max Results: {settings.default_max_results}")
    print(f"Chat History Size: {settings.default_chat_history_size}")
    print(f"Cache TTL: {settings.cache_ttl}s")
    print(f"Cache Max Size: {settings.cache_max_size}")
    
    # API 키 상태 (보안상 값은 표시하지 않음)
    api_keys = []
    if settings.openai_api_key:
        api_keys.append("OpenAI")
    if settings.anthropic_api_key:
        api_keys.append("Anthropic") 
    if settings.google_api_key:
        api_keys.append("Google")
    
    print(f"API Keys configured: {', '.join(api_keys) if api_keys else 'None'}")
    print("=" * 37)