"""
GraphitiService 통합 테스트
Integration tests for GraphitiService
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from rag_chatbot.graphiti_service import (
    GraphitiService, 
    get_graphiti_service, 
    close_graphiti_service,
    _service_instance
)
from rag_chatbot.config import Settings


class TestGraphitiService:
    """GraphitiService 클래스 테스트"""
    
    @pytest.fixture
    def service(self, mock_settings):
        """GraphitiService fixture"""
        return GraphitiService(mock_settings)
    
    @pytest.fixture
    def mock_graphiti(self):
        """Mock Graphiti instance"""
        graphiti = AsyncMock()
        graphiti.build_indices_and_constraints = AsyncMock()
        graphiti.add_episode = AsyncMock()
        graphiti.search = AsyncMock(return_value=[])
        graphiti._search = AsyncMock()
        graphiti.close = AsyncMock()
        return graphiti
    
    @pytest.fixture
    def mock_falkor_driver(self):
        """Mock FalkorDriver"""
        return MagicMock()
    
    async def test_initialize_success(self, service, mock_graphiti, mock_falkor_driver):
        """정상 초기화 테스트"""
        with patch('rag_chatbot.graphiti_service.FalkorDriver', return_value=mock_falkor_driver), \
             patch('rag_chatbot.graphiti_service.Graphiti', return_value=mock_graphiti):
            
            await service.initialize()
            
            assert service._connection_pool_ready is True
            assert service._graphiti is mock_graphiti
            mock_graphiti.build_indices_and_constraints.assert_called_once()
    
    async def test_initialize_failure(self, service):
        """초기화 실패 테스트"""
        with patch('rag_chatbot.graphiti_service.FalkorDriver', side_effect=Exception("연결 실패")):
            
            with pytest.raises(Exception, match="연결 실패"):
                await service.initialize()
            
            assert service._connection_pool_ready is False
            assert service._graphiti is None
    
    async def test_close(self, service, mock_graphiti):
        """서비스 종료 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        await service.close()
        
        mock_graphiti.close.assert_called_once()
        assert service._connection_pool_ready is False
    
    def test_ensure_connected_not_initialized(self, service):
        """초기화되지 않은 상태에서 연결 확인 테스트"""
        with pytest.raises(RuntimeError, match="Graphiti service not initialized"):
            service._ensure_connected()
    
    def test_ensure_connected_initialized(self, service, mock_graphiti):
        """초기화된 상태에서 연결 확인 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        # 예외가 발생하지 않아야 함
        service._ensure_connected()
    
    async def test_add_text_episode(self, service, mock_graphiti):
        """텍스트 에피소드 추가 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        await service.add_text_episode(
            name="테스트 에피소드",
            content="테스트 내용",
            source_description="test"
        )
        
        mock_graphiti.add_episode.assert_called_once()
        call_args = mock_graphiti.add_episode.call_args[1]
        assert call_args['name'] == "테스트 에피소드"
        assert call_args['episode_body'] == "테스트 내용"
        assert call_args['source_description'] == "test"
    
    async def test_add_text_episode_with_reference_time(self, service, mock_graphiti):
        """참조 시간이 있는 텍스트 에피소드 추가 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        ref_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        await service.add_text_episode(
            name="시간 에피소드",
            content="시간 테스트",
            reference_time=ref_time
        )
        
        call_args = mock_graphiti.add_episode.call_args[1]
        assert call_args['reference_time'] == ref_time
    
    async def test_add_json_episode(self, service, mock_graphiti):
        """JSON 에피소드 추가 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        test_data = {"name": "테스트", "value": 123}
        
        await service.add_json_episode(
            name="JSON 테스트",
            data=test_data
        )
        
        mock_graphiti.add_episode.assert_called_once()
        call_args = mock_graphiti.add_episode.call_args[1]
        assert call_args['name'] == "JSON 테스트"
        assert '"name": "테스트"' in call_args['episode_body']
        assert '"value": 123' in call_args['episode_body']
    
    async def test_search_basic(self, service, mock_graphiti):
        """기본 검색 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        mock_results = [MagicMock(), MagicMock()]
        mock_graphiti.search.return_value = mock_results
        
        results = await service.search("테스트 쿼리")
        
        assert results == mock_results
        mock_graphiti.search.assert_called_once_with(
            query="테스트 쿼리",
            num_results=5  # default_max_results
        )
    
    async def test_search_with_center_node(self, service, mock_graphiti):
        """중심 노드가 있는 검색 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        mock_results = [MagicMock()]
        mock_graphiti.search.return_value = mock_results
        
        results = await service.search(
            "개인화 쿼리", 
            center_node_uuid="user-123",
            max_results=10
        )
        
        assert results == mock_results
        mock_graphiti.search.assert_called_once_with(
            query="개인화 쿼리",
            center_node_uuid="user-123",
            num_results=10
        )
    
    async def test_search_failure(self, service, mock_graphiti):
        """검색 실패 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        mock_graphiti.search.side_effect = Exception("검색 실패")
        
        with pytest.raises(Exception, match="검색 실패"):
            await service.search("실패 쿼리")
    
    async def test_node_search(self, service, mock_graphiti):
        """노드 검색 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        mock_search_result = MagicMock()
        mock_search_result.nodes = [MagicMock(), MagicMock()]
        mock_graphiti._search.return_value = mock_search_result
        
        results = await service.node_search("노드 쿼리", max_results=3)
        
        assert results == mock_search_result.nodes
        mock_graphiti._search.assert_called_once()
        
        # 검색 설정 확인
        call_args = mock_graphiti._search.call_args
        assert call_args[1]['query'] == "노드 쿼리"
        assert call_args[1]['config'].limit == 3
    
    async def test_get_health_status_healthy(self, service, mock_graphiti):
        """정상 상태 확인 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        mock_graphiti.search.return_value = []
        
        status = await service.get_health_status()
        
        assert status['service'] == 'graphiti'
        assert status['status'] == 'healthy'
        assert status['connection_ready'] is True
        assert 'timestamp' in status
    
    async def test_get_health_status_not_initialized(self, service):
        """초기화되지 않은 상태 확인 테스트"""
        status = await service.get_health_status()
        
        assert status['status'] == 'not_initialized'
        assert status['connection_ready'] is False
    
    async def test_get_health_status_unhealthy(self, service, mock_graphiti):
        """비정상 상태 확인 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        mock_graphiti.search.side_effect = Exception("연결 오류")
        
        status = await service.get_health_status()
        
        assert status['status'] == 'unhealthy'
        assert 'error' in status
        assert status['error'] == '연결 오류'


class TestGraphitiServiceFormatting:
    """검색 결과 포맷팅 테스트"""
    
    @pytest.fixture
    def service(self, mock_settings):
        """GraphitiService fixture"""
        return GraphitiService(mock_settings)
    
    def test_format_search_results_empty(self, service):
        """빈 검색 결과 포맷팅 테스트"""
        formatted = service.format_search_results([])
        assert formatted == "검색 결과가 없습니다."
    
    def test_format_search_results_with_data(self, service):
        """데이터가 있는 검색 결과 포맷팅 테스트"""
        # Mock EntityEdge 객체들 생성
        result1 = MagicMock()
        result1.fact = "첫 번째 사실"
        result1.valid_at = None
        result1.invalid_at = None
        
        result2 = MagicMock()
        result2.fact = "두 번째 사실"
        result2.valid_at = datetime(2024, 1, 1)
        result2.invalid_at = None
        
        results = [result1, result2]
        formatted = service.format_search_results(results)
        
        assert "1. 첫 번째 사실" in formatted
        assert "2. 두 번째 사실" in formatted
        assert "시작: 2024-01-01" in formatted
    
    def test_format_search_results_with_dates(self, service):
        """날짜 정보가 있는 검색 결과 포맷팅 테스트"""
        result = MagicMock()
        result.fact = "날짜 테스트"
        result.valid_at = datetime(2024, 1, 1)
        result.invalid_at = datetime(2024, 12, 31)
        
        formatted = service.format_search_results([result])
        
        assert "날짜 테스트" in formatted
        assert "시작: 2024-01-01" in formatted
        assert "종료: 2024-12-31" in formatted


class TestGlobalServiceManagement:
    """전역 서비스 관리 테스트"""
    
    @pytest.fixture(autouse=True)
    async def cleanup_global_service(self):
        """각 테스트 후 전역 서비스 정리"""
        yield
        await close_graphiti_service()
    
    async def test_get_graphiti_service_singleton(self, mock_settings):
        """싱글톤 패턴 테스트"""
        with patch.object(GraphitiService, 'initialize') as mock_init:
            service1 = await get_graphiti_service(mock_settings)
            service2 = await get_graphiti_service(mock_settings)
            
            # 같은 인스턴스여야 함
            assert service1 is service2
            
            # 초기화는 한 번만 호출되어야 함
            mock_init.assert_called_once()
    
    async def test_close_graphiti_service(self, mock_settings):
        """전역 서비스 종료 테스트"""
        with patch.object(GraphitiService, 'initialize'), \
             patch.object(GraphitiService, 'close') as mock_close:
            
            # 서비스 생성
            service = await get_graphiti_service(mock_settings)
            assert service is not None
            
            # 서비스 종료
            await close_graphiti_service()
            
            # close 메서드가 호출되어야 함
            mock_close.assert_called_once()
            
            # 전역 인스턴스가 None이 되어야 함
            from rag_chatbot.graphiti_service import _service_instance
            assert _service_instance is None
    
    async def test_get_service_after_close(self, mock_settings):
        """서비스 종료 후 재생성 테스트"""
        with patch.object(GraphitiService, 'initialize') as mock_init:
            # 첫 번째 서비스 생성 및 종료
            service1 = await get_graphiti_service(mock_settings)
            await close_graphiti_service()
            
            # 두 번째 서비스 생성
            service2 = await get_graphiti_service(mock_settings)
            
            # 다른 인스턴스여야 함
            assert service1 is not service2
            
            # 초기화가 두 번 호출되어야 함
            assert mock_init.call_count == 2


class TestGraphitiServiceErrorHandling:
    """에러 처리 테스트"""
    
    @pytest.fixture
    def service(self, mock_settings):
        """GraphitiService fixture"""
        return GraphitiService(mock_settings)
    
    async def test_add_episode_not_connected(self, service):
        """연결되지 않은 상태에서 에피소드 추가 테스트"""
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.add_text_episode("테스트", "내용")
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.add_json_episode("테스트", {})
    
    async def test_search_not_connected(self, service):
        """연결되지 않은 상태에서 검색 테스트"""
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.search("쿼리")
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.node_search("쿼리")
    
    async def test_add_text_episode_failure(self, service, mock_graphiti):
        """텍스트 에피소드 추가 실패 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        mock_graphiti.add_episode.side_effect = Exception("추가 실패")
        
        with pytest.raises(Exception, match="추가 실패"):
            await service.add_text_episode("실패 테스트", "내용")
    
    async def test_add_json_episode_failure(self, service, mock_graphiti):
        """JSON 에피소드 추가 실패 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        mock_graphiti.add_episode.side_effect = Exception("JSON 추가 실패")
        
        with pytest.raises(Exception, match="JSON 추가 실패"):
            await service.add_json_episode("실패 테스트", {})
    
    async def test_node_search_failure(self, service, mock_graphiti):
        """노드 검색 실패 테스트"""
        service._graphiti = mock_graphiti
        service._connection_pool_ready = True
        
        mock_graphiti._search.side_effect = Exception("노드 검색 실패")
        
        with pytest.raises(Exception, match="노드 검색 실패"):
            await service.node_search("실패 쿼리")