"""
Core Graphiti service layer for RAG chatbot.
Graphiti 0.17.4를 사용한 지식 그래프 관리 서비스
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from functools import lru_cache, wraps
from collections import OrderedDict
import time
from enum import Enum

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

from .config import Settings

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """회로 차단기 상태"""
    CLOSED = "closed"      # 정상 상태
    OPEN = "open"          # 차단됨 (빠른 실패)
    HALF_OPEN = "half_open"  # 반열림 (복구 시도)


class CircuitBreaker:
    """
    Simple circuit breaker for database connections.
    데이터베이스 연결을 위한 간단한 회로 차단기
    """
    
    def __init__(
        self, 
        failure_threshold: int = 5,     # 실패 임계값
        recovery_timeout: int = 60,     # 복구 시도 시간 (초)
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """회로 차단기를 통한 함수 호출"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker: attempting recovery (HALF_OPEN)")
                else:
                    raise Exception("Circuit breaker is OPEN - failing fast")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except self.expected_exception as e:
            await self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """복구 시도 여부 판단"""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    async def _on_success(self) -> None:
        """성공 시 처리"""
        async with self._lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
            if self.state == CircuitBreakerState.HALF_OPEN:
                logger.info("Circuit breaker: recovered (CLOSED)")
    
    async def _on_failure(self) -> None:
        """실패 시 처리"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker: opened after {self.failure_count} failures")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
):
    """
    Exponential backoff retry decorator.
    지수 백오프를 사용한 재시도 데코레이터
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Final retry failed for {func.__name__}: {e}")
                        raise
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
            
        return wrapper
    return decorator


class TTLCache:
    """
    Simple TTL (Time To Live) cache for search results.
    검색 결과를 위한 간단한 TTL 캐시
    """
    
    def __init__(self, maxsize: int = 100, ttl: int = 300):  # 5분 TTL
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()
    
    def _generate_key(self, query: str, max_results: int, center_node_uuid: Optional[str]) -> str:
        """캐시 키 생성"""
        key_data = f"{query}:{max_results}:{center_node_uuid or ''}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, query: str, max_results: int, center_node_uuid: Optional[str]) -> Optional[List[EntityEdge]]:
        """캐시에서 검색 결과 조회"""
        async with self._lock:
            key = self._generate_key(query, max_results, center_node_uuid)
            
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    # 최근에 사용된 항목을 맨 뒤로 이동 (LRU)
                    self.cache.move_to_end(key)
                    logger.debug(f"Cache hit for query: {query}")
                    return value
                else:
                    # TTL 만료된 항목 제거
                    del self.cache[key]
                    logger.debug(f"Cache expired for query: {query}")
            
            return None
    
    async def put(self, query: str, max_results: int, center_node_uuid: Optional[str], results: List[EntityEdge]) -> None:
        """검색 결과를 캐시에 저장"""
        async with self._lock:
            key = self._generate_key(query, max_results, center_node_uuid)
            
            # 캐시 크기 제한
            while len(self.cache) >= self.maxsize:
                # 가장 오래된 항목 제거 (FIFO)
                self.cache.popitem(last=False)
            
            self.cache[key] = (results, time.time())
            logger.debug(f"Cached results for query: {query}")
    
    async def clear(self) -> None:
        """캐시 전체 삭제"""
        async with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")


class GraphitiService:
    """
    Production-ready Graphiti service wrapper.
    FalkorDB와 함께 지식 그래프 관리를 위한 서비스 클래스
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._graphiti: Optional[Graphiti] = None
        self._connection_pool_ready = False
        self._cache = TTLCache(maxsize=100, ttl=300)  # 5분 TTL 캐시
        self._initialization_lock = asyncio.Lock()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
    @retry_with_backoff(max_retries=3, base_delay=2.0)
    async def initialize(self) -> None:
        """Initialize Graphiti connection and build indices."""
        async with self._initialization_lock:
            if self._connection_pool_ready:
                return  # 이미 초기화됨
                
            try:
                await self._circuit_breaker.call(self._do_initialize)
                
            except Exception as e:
                logger.error(f"Failed to initialize Graphiti service: {e}")
                raise
    
    async def _do_initialize(self) -> None:
        """실제 초기화 로직"""
        # FalkorDB 드라이버 초기화
        falkor_driver = FalkorDriver(
            host=self.settings.falkor_host,
            port=self.settings.falkor_port,
            username=self.settings.falkor_username,
            password=self.settings.falkor_password.get_secret_value() 
            if self.settings.falkor_password else None
        )
        
        # Graphiti 인스턴스 생성
        self._graphiti = Graphiti(graph_driver=falkor_driver)
        
        # 인덱스와 제약조건 생성 (처음 실행시에만 필요)
        await self._graphiti.build_indices_and_constraints()
        self._connection_pool_ready = True
        
        logger.info("Graphiti service initialized successfully")
    
    async def close(self) -> None:
        """Close Graphiti connection gracefully."""
        if self._graphiti:
            await self._graphiti.close()
            self._connection_pool_ready = False
            await self._cache.clear()
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
        Search knowledge graph using hybrid search with caching.
        캐싱을 포함한 하이브리드 검색을 사용한 지식 그래프 검색
        """
        self._ensure_connected()
        
        if max_results is None:
            max_results = self.settings.default_max_results
        
        # 캐시에서 먼저 검색 시도
        cached_results = await self._cache.get(query, max_results, center_node_uuid)
        if cached_results is not None:
            logger.info(f"Search query '{query}' returned {len(cached_results)} cached results")
            return cached_results
        
        try:
            # 회로 차단기를 통한 검색 실행
            results = await self._circuit_breaker.call(
                self._do_search, query, max_results, center_node_uuid
            )
            
            # 결과를 캐시에 저장
            await self._cache.put(query, max_results, center_node_uuid, results)
            
            logger.info(f"Search query '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise
    
    @retry_with_backoff(max_retries=2, base_delay=0.5, max_delay=5.0)
    async def _do_search(
        self, 
        query: str, 
        max_results: int, 
        center_node_uuid: Optional[str]
    ) -> List[EntityEdge]:
        """실제 검색 로직"""
        if center_node_uuid:
            # 중심 노드 기반 검색 (개인화된 결과)
            return await self._graphiti.search(
                query=query,
                center_node_uuid=center_node_uuid,
                num_results=max_results
            )
        else:
            # 일반 하이브리드 검색
            return await self._graphiti.search(
                query=query,
                num_results=max_results
            )
    
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
    
    async def clear_cache(self) -> None:
        """수동으로 캐시 삭제"""
        await self._cache.clear()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 조회"""
        async with self._cache._lock:
            return {
                "cache_size": len(self._cache.cache),
                "max_size": self._cache.maxsize,
                "ttl_seconds": self._cache.ttl
            }
    
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