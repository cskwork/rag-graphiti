"""
Core Graphiti service layer for RAG chatbot.
Graphiti 0.17.4를 사용한 지식 그래프 관리 서비스
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

from .config import Settings

logger = logging.getLogger(__name__)


class GraphitiService:
    """
    Production-ready Graphiti service wrapper.
    FalkorDB와 함께 지식 그래프 관리를 위한 서비스 클래스
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._graphiti: Optional[Graphiti] = None
        self._connection_pool_ready = False
        
    async def initialize(self) -> None:
        """Initialize Graphiti connection and build indices."""
        try:
            # FalkorDB 드라이버 초기화
            falkor_driver = FalkorDriver(
                host=self.settings.falkordb_host,
                port=self.settings.falkordb_port,
                username=self.settings.falkordb_username,
                password=self.settings.falkordb_password.get_secret_value() 
                if self.settings.falkordb_password else None
            )
            
            # Graphiti 인스턴스 생성
            self._graphiti = Graphiti(graph_driver=falkor_driver)
            
            # 인덱스와 제약조건 생성 (처음 실행시에만 필요)
            await self._graphiti.build_indices_and_constraints()
            self._connection_pool_ready = True
            
            logger.info("Graphiti service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti service: {e}")
            raise
    
    async def close(self) -> None:
        """Close Graphiti connection gracefully."""
        if self._graphiti:
            await self._graphiti.close()
            self._connection_pool_ready = False
            logger.info("Graphiti service closed")
    
    def _ensure_connected(self) -> None:
        """Ensure Graphiti is initialized."""
        if not self._connection_pool_ready or not self._graphiti:
            raise RuntimeError("Graphiti service not initialized. Call initialize() first.")
    
    async def add_text_episode(
        self,
        name: str,
        content: str,
        source_description: str = "user_input",
        reference_time: Optional[datetime] = None
    ) -> None:
        """
        Add text episode to knowledge graph.
        텍스트 에피소드를 지식 그래프에 추가
        """
        self._ensure_connected()
        
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        
        try:
            await self._graphiti.add_episode(
                name=name,
                episode_body=content,
                source=EpisodeType.text,
                source_description=source_description,
                reference_time=reference_time,
            )
            logger.info(f"Added text episode: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add text episode {name}: {e}")
            raise
    
    async def add_json_episode(
        self,
        name: str,
        data: Dict[str, Any],
        source_description: str = "structured_data",
        reference_time: Optional[datetime] = None
    ) -> None:
        """
        Add JSON episode to knowledge graph.
        JSON 에피소드를 지식 그래프에 추가
        """
        self._ensure_connected()
        
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        
        try:
            await self._graphiti.add_episode(
                name=name,
                episode_body=json.dumps(data, ensure_ascii=False),
                source=EpisodeType.json,
                source_description=source_description,
                reference_time=reference_time,
            )
            logger.info(f"Added JSON episode: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add JSON episode {name}: {e}")
            raise
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        center_node_uuid: Optional[str] = None
    ) -> List[EntityEdge]:
        """
        Search knowledge graph using hybrid search.
        하이브리드 검색을 사용한 지식 그래프 검색
        """
        self._ensure_connected()
        
        if max_results is None:
            max_results = self.settings.default_max_results
        
        try:
            if center_node_uuid:
                # 중심 노드 기반 검색 (개인화된 결과)
                results = await self._graphiti.search(
                    query=query,
                    center_node_uuid=center_node_uuid,
                    num_results=max_results
                )
            else:
                # 일반 하이브리드 검색
                results = await self._graphiti.search(
                    query=query,
                    num_results=max_results
                )
            
            logger.info(f"Search query '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise
    
    async def node_search(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[Any]:
        """
        Search for nodes directly using search recipes.
        검색 레시피를 사용한 노드 직접 검색
        """
        self._ensure_connected()
        
        if max_results is None:
            max_results = self.settings.default_max_results
        
        try:
            # 노드 검색을 위한 설정
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            search_config.limit = max_results
            
            # 노드 검색 실행
            search_results = await self._graphiti._search(
                query=query,
                config=search_config,
            )
            
            logger.info(f"Node search for '{query}' returned {len(search_results.nodes)} nodes")
            return search_results.nodes
            
        except Exception as e:
            logger.error(f"Node search failed for query '{query}': {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get service health status.
        서비스 상태 확인
        """
        status = {
            "service": "graphiti",
            "status": "unknown",
            "connection_ready": self._connection_pool_ready,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            if self._graphiti and self._connection_pool_ready:
                # 간단한 검색으로 연결 테스트
                await self._graphiti.search("health_check", num_results=1)
                status["status"] = "healthy"
            else:
                status["status"] = "not_initialized"
                
        except Exception as e:
            status["status"] = "unhealthy"
            status["error"] = str(e)
            logger.warning(f"Health check failed: {e}")
        
        return status
    
    def format_search_results(self, results: List[EntityEdge]) -> str:
        """
        Format search results for display.
        검색 결과를 표시용으로 포맷팅
        """
        if not results:
            return "검색 결과가 없습니다."
        
        formatted = []
        for i, result in enumerate(results, 1):
            fact = result.fact
            # 날짜 정보가 있으면 포함
            date_info = ""
            if hasattr(result, 'valid_at') and result.valid_at:
                date_info = f" (시작: {result.valid_at})"
            if hasattr(result, 'invalid_at') and result.invalid_at:
                date_info += f" (종료: {result.invalid_at})"
            
            formatted.append(f"{i}. {fact}{date_info}")
        
        return "\n".join(formatted)


# 전역 서비스 인스턴스 (싱글톤 패턴)
_service_instance: Optional[GraphitiService] = None


async def get_graphiti_service(settings: Settings) -> GraphitiService:
    """
    Get or create Graphiti service instance.
    Graphiti 서비스 인스턴스 가져오기 또는 생성
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = GraphitiService(settings)
        await _service_instance.initialize()
    
    return _service_instance


async def close_graphiti_service() -> None:
    """Close global Graphiti service instance."""
    global _service_instance
    
    if _service_instance:
        await _service_instance.close()
        _service_instance = None