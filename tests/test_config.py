"""
Minimal configuration module tests.
최소한의 Configuration 모듈 테스트
"""
import pytest
from unittest.mock import patch
from pydantic import ValidationError

from rag_chatbot.config import Settings, get_settings, setup_logging


def test_default_settings():
    """Test default settings."""
    settings = Settings()
    
    assert settings.falkor_host == "localhost"
    assert settings.falkor_port == "6379"
    assert settings.log_level == "INFO"
    assert settings.web_port == 8000


def test_settings_validation():
    """Test settings validation."""
    # Valid settings
    settings = Settings(falkor_port="6379", web_port=8000)
    assert settings.falkor_port == "6379"
    
    # Invalid port range
    with pytest.raises(ValidationError):
        Settings(web_port=99999)
    
    # Invalid log level
    with pytest.raises(ValidationError):
        Settings(log_level="INVALID")


def test_get_settings():
    """Test get_settings function."""
    settings = get_settings()
    assert isinstance(settings, Settings)


@patch('logging.basicConfig')
def test_setup_logging(mock_basic_config):
    """Test logging setup."""
    settings = Settings(log_level="DEBUG")
    setup_logging(settings)
    
    # 로깅이 호출되었는지만 확인 (구체적인 파라미터는 검증하지 않음)
    mock_basic_config.assert_called_once()