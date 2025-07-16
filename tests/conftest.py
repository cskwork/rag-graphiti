"""
Minimal test configuration and common fixtures.
최소한의 테스트 설정 및 공통 fixture 정의
"""
import pytest
from unittest.mock import AsyncMock

from rag_chatbot.config import Settings


@pytest.fixture
def settings():
    """Minimal test settings fixture."""
    return Settings(
        falkor_host="localhost",
        falkor_port="6379",
        log_level="DEBUG",
        web_host="127.0.0.1",
        web_port=8001,  # 다른 포트 사용하여 충돌 방지
        default_max_results=3,  # 테스트용으로 작은 값
        default_chat_history_size=5
    )


@pytest.fixture
def mock_graphiti_service():
    """Minimal GraphitiService mock."""
    service = AsyncMock()
    service.initialize = AsyncMock()
    service.close = AsyncMock()
    service.search = AsyncMock(return_value=[])
    service.add_text_episode = AsyncMock()
    service.add_json_episode = AsyncMock()
    service.get_health_status = AsyncMock(return_value={
        "status": "healthy",
        "connection_ready": True
    })
    return service


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "테스트용 샘플 텍스트입니다."


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    from unittest.mock import MagicMock
    
    result = MagicMock()
    result.fact = "테스트 검색 결과"
    result.valid_at = None
    result.invalid_at = None
    return [result]


@pytest.fixture(autouse=True)  
def clean_test_env(monkeypatch):
    """Minimal test environment setup."""
    # 테스트용 환경 변수만 설정
    monkeypatch.setenv("FALKOR_HOST", "localhost")
    monkeypatch.setenv("FALKOR_PORT", "6380")  # 다른 포트로 충돌 방지
    monkeypatch.setenv("LOG_LEVEL", "ERROR")    # 테스트 시 로그 최소화