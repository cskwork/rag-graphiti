"""
Chat handling logic for RAG chatbot.
RAG 채팅봇의 대화 처리 로직
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .config import Settings
from .graphiti_service import GraphitiService

logger = logging.getLogger(__name__)


class ChatError(Exception):
    """채팅 처리 관련 기본 예외"""
    pass


class SearchError(ChatError):
    """검색 관련 예외"""
    pass


class ContextError(ChatError):
    """컨텍스트 처리 관련 예외"""
    pass


class ConversationSaveError(ChatError):
    """대화 저장 관련 예외"""
    pass


class ChatHandler:
    """
    Handle chat interactions using RAG pattern.
    RAG 패턴을 사용한 채팅 상호작용 처리
    """
    
    def __init__(self, graphiti_service: GraphitiService, settings: Settings):
        self.graphiti_service = graphiti_service
        self.settings = settings
        self.chat_history: List[Dict[str, Any]] = []
        self.recent_topics: List[str] = []  # 최근 3개 토픽만 유지
        
    async def process_query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        max_context_results: Optional[int] = None
    ) -> str:
        """
        Process user query using RAG pattern with enhanced error handling.
        향상된 오류 처리를 포함한 RAG 패턴을 사용한 사용자 질의 처리
        """
        # 입력 검증
        if not user_query or not user_query.strip():
            return "질문을 입력해 주세요."
        
        if len(user_query.strip()) > 1000:  # 길이 제한
            return "질문이 너무 길습니다. 1000자 이내로 입력해 주세요."
        
        try:
            # 1. 간단한 컨텍스트 업데이트
            self._update_recent_topics(user_query)
            
            # 2. 컨텍스트 기반 검색
            search_results = await self._search_with_context(
                user_query, user_id, max_context_results
            )
            
            # 3. 검색 결과를 컨텍스트로 포맷팅
            try:
                context = self._format_context(search_results)
            except Exception as e:
                logger.error(f"Context formatting error: {e}")
                raise ContextError(f"검색 결과 처리 중 오류가 발생했습니다: {str(e)}")
            
            # 4. 응답 생성
            response = self._generate_response(user_query, context, search_results)
            
            # 5. 채팅 기록에 저장
            try:
                self._add_to_history(user_query, response)
            except Exception as e:
                logger.warning(f"Failed to add to chat history: {e}")
            
            # 6. 대화 저장
            try:
                await self._save_conversation_to_graph(user_query, response, user_id)
            except ConversationSaveError as e:
                logger.warning(f"Failed to save conversation: {e}")
            
            return response
            
        except SearchError as e:
            logger.error(f"Search error for query '{user_query}': {e}")
            return "죄송합니다. 검색 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
            
        except ContextError as e:
            logger.error(f"Context error for query '{user_query}': {e}")
            return "죄송합니다. 정보 처리 중 문제가 발생했습니다. 다시 시도해 주세요."
            
        except Exception as e:
            logger.error(f"Unexpected error processing query '{user_query}': {e}", exc_info=True)
            return "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    
    def _update_recent_topics(self, user_query: str) -> None:
        """
        Update recent topics from user query.
        사용자 쿼리에서 최근 토픽 업데이트
        """
        # 간단한 키워드 기반 토픽 추출
        topics = []
        text_lower = user_query.lower()
        
        # 주요 키워드만 체크
        if any(word in text_lower for word in ['python', 'javascript', 'code', 'programming']):
            topics.append("programming")
        if any(word in text_lower for word in ['database', 'sql', 'data']):
            topics.append("database")
        if any(word in text_lower for word in ['web', 'html', 'css', 'frontend']):
            topics.append("web")
        if any(word in text_lower for word in ['server', 'backend', 'api']):
            topics.append("backend")
        
        # 최근 토픽 업데이트 (최대 3개)
        for topic in topics:
            if topic not in self.recent_topics:
                self.recent_topics.append(topic)
        
        # 최근 3개만 유지
        if len(self.recent_topics) > 3:
            self.recent_topics = self.recent_topics[-3:]
    
    async def _search_with_context(
        self,
        user_query: str,
        user_id: Optional[str],
        max_context_results: Optional[int]
    ) -> List[Any]:
        """
        Search with minimal context enhancement.
        최소한의 컨텍스트 향상을 통한 검색
        """
        if max_context_results is None:
            max_context_results = self.settings.default_max_results
        
        try:
            # 최근 토픽으로 쿼리 확장 (간단하게)
            enhanced_query = user_query
            if self.recent_topics:
                enhanced_query = f"{user_query} {' '.join(self.recent_topics)}"
            
            # 사용자 중심 검색
            center_node_uuid = None
            if user_id:
                try:
                    user_nodes = await self.graphiti_service.node_search(f"user:{user_id}")
                    if user_nodes:
                        center_node_uuid = user_nodes[0].uuid
                except Exception as e:
                    logger.warning(f"Failed to find user node for {user_id}: {e}")
            
            # 검색 실행
            search_results = await self.graphiti_service.search(
                query=enhanced_query,
                max_results=max_context_results,
                center_node_uuid=center_node_uuid
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed for query '{user_query}': {e}")
            raise SearchError(f"검색 중 오류가 발생했습니다: {str(e)}")
    
    def _format_context(self, search_results: List[Any]) -> str:
        """
        Format search results as context.
        검색 결과를 컨텍스트로 포맷팅
        """
        if not search_results:
            return "관련 정보를 찾을 수 없습니다."
        
        context_lines = ["관련 정보:"]
        for i, result in enumerate(search_results, 1):
            fact = result.fact
            context_lines.append(f"- {fact}")
        
        return "\n".join(context_lines)
    
    def _generate_response(
        self,
        user_query: str,
        context: str,
        search_results: List[Any]
    ) -> str:
        """
        Generate response based on query and context.
        질의와 컨텍스트를 기반으로 응답 생성
        
        Note: 이 기본 구현은 LLM 없이 단순한 응답을 생성합니다.
        프로덕션에서는 OpenAI, Anthropic, 또는 로컬 LLM을 통합할 수 있습니다.
        """
        if not search_results:
            return (
                f"'{user_query}'에 대한 정보를 지식 베이스에서 찾을 수 없습니다.\n"
                "더 구체적인 질문을 해보시거나, 관련 문서를 먼저 추가해 주세요."
            )
        
        # 기본적인 패턴 매칭 응답
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ["안녕", "hello", "hi"]):
            response = "안녕하세요! 무엇을 도와드릴까요?"
        elif any(word in query_lower for word in ["도움", "help", "사용법"]):
            response = self._get_help_response()
        else:
            # 검색 결과 기반 응답
            response = f"'{user_query}'에 대한 정보를 찾았습니다:\n\n{context}"
            
            if len(search_results) >= self.settings.default_max_results:
                response += f"\n\n더 많은 결과가 있을 수 있습니다. 더 구체적인 질문을 해보세요."
        
        return response
    
    def _get_help_response(self) -> str:
        """Get help response."""
        return """
사용 가능한 명령어:
- 일반 질문: 지식 베이스에서 정보를 검색합니다
- 'exit' 또는 'quit': 채팅을 종료합니다
- 'clear': 채팅 기록을 지웁니다
- 'help': 이 도움말을 표시합니다

예시 질문:
- "Python에 대해 알려주세요"
- "머신러닝이란 무엇인가요?"
- "프로젝트 상태는 어떻게 되나요?"
"""
    
    def _add_to_history(self, user_query: str, response: str) -> None:
        """
        Add conversation to chat history.
        대화를 채팅 기록에 추가
        """
        timestamp = datetime.now(timezone.utc)
        
        self.chat_history.extend([
            {
                "role": "user",
                "content": user_query,
                "timestamp": timestamp.isoformat()
            },
            {
                "role": "assistant", 
                "content": response,
                "timestamp": timestamp.isoformat()
            }
        ])
        
        # 기록 크기 제한
        max_history = self.settings.default_chat_history_size * 2  # user + assistant
        if len(self.chat_history) > max_history:
            self.chat_history = self.chat_history[-max_history:]
    
    async def _save_conversation_to_graph(
        self,
        user_query: str,
        response: str,
        user_id: Optional[str] = None
    ) -> None:
        """
        Save conversation to knowledge graph with enhanced error handling.
        향상된 오류 처리를 포함한 대화를 지식 그래프에 저장
        """
        try:
            # 입력 검증
            if not user_query.strip() or not response.strip():
                raise ConversationSaveError("대화 내용이 비어있습니다")
            
            conversation_text = f"User: {user_query}\nAssistant: {response}"
            
            episode_name = f"chat_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            if user_id:
                # 사용자 ID 검증 (간단한 알파뉴메릭 체크)
                if user_id and not user_id.replace('_', '').replace('-', '').isalnum():
                    logger.warning(f"Invalid user_id format: {user_id}")
                    user_id = None
                else:
                    episode_name += f"_{user_id}"
            
            await self.graphiti_service.add_text_episode(
                name=episode_name,
                content=conversation_text,
                source_description="chat_conversation"
            )
            
            logger.debug(f"Saved conversation to graph: {episode_name}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation to graph: {e}")
            raise ConversationSaveError(f"대화 저장 실패: {str(e)}")
    
    def clear_history(self) -> None:
        """Clear chat history and recent topics."""
        self.chat_history.clear()
        self.recent_topics.clear()
        logger.info("Chat history and context cleared")
    
    def get_history_summary(self) -> str:
        """Get chat history summary."""
        if not self.chat_history:
            return "채팅 기록이 없습니다."
        
        user_messages = [msg for msg in self.chat_history if msg["role"] == "user"]
        return f"총 {len(user_messages)}개의 대화가 있습니다."