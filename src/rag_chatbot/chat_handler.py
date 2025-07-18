"""
Chat handling logic for RAG chatbot.
RAG 채팅봇의 대화 처리 로직
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import Settings
from .graphiti_service import GraphitiService

logger = logging.getLogger(__name__)


class ChatHandler:
    """
    Handle chat interactions using RAG pattern.
    RAG 패턴을 사용한 채팅 상호작용 처리
    """
    
    def __init__(self, graphiti_service: GraphitiService, settings: Settings):
        self.graphiti_service = graphiti_service
        self.settings = settings
        self.chat_history: List[Dict[str, Any]] = []
        
    async def process_query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        max_context_results: Optional[int] = None
    ) -> str:
        """
        Process user query using RAG pattern.
        RAG 패턴을 사용한 사용자 질의 처리
        """
        try:
            # 1. 지식 그래프에서 관련 정보 검색
            if max_context_results is None:
                max_context_results = self.settings.default_max_results
            
            # 사용자별 개인화된 검색 (user_id가 있는 경우)
            center_node_uuid = None
            if user_id:
                # 사용자 노드 찾기
                user_nodes = await self.graphiti_service.node_search(f"user:{user_id}")
                if user_nodes:
                    center_node_uuid = user_nodes[0].uuid
            
            # 관련 정보 검색
            search_results = await self.graphiti_service.search(
                query=user_query,
                max_results=max_context_results,
                center_node_uuid=center_node_uuid
            )
            
            # 2. 검색 결과를 컨텍스트로 포맷팅
            context = self._format_context(search_results)
            
            # 3. 간단한 응답 생성 (LLM 없이 기본 응답)
            response = self._generate_response(user_query, context, search_results)
            
            # 4. 채팅 기록에 저장
            self._add_to_history(user_query, response)
            
            # 5. 현재 대화를 지식 그래프에 추가 (비동기로)
            await self._save_conversation_to_graph(user_query, response, user_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{user_query}': {e}")
            return f"죄송합니다. 쿼리 처리 중 오류가 발생했습니다: {str(e)}"
    
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
        Save conversation to knowledge graph.
        대화를 지식 그래프에 저장
        """
        try:
            conversation_text = f"User: {user_query}\nAssistant: {response}"
            
            episode_name = f"chat_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            if user_id:
                episode_name += f"_{user_id}"
            
            await self.graphiti_service.add_text_episode(
                name=episode_name,
                content=conversation_text,
                source_description="chat_conversation"
            )
            
            logger.debug(f"Saved conversation to graph: {episode_name}")
            
        except Exception as e:
            logger.warning(f"Failed to save conversation to graph: {e}")
    
    def clear_history(self) -> None:
        """Clear chat history."""
        self.chat_history.clear()
        logger.info("Chat history cleared")
    
    def get_history_summary(self) -> str:
        """Get chat history summary."""
        if not self.chat_history:
            return "채팅 기록이 없습니다."
        
        user_messages = [msg for msg in self.chat_history if msg["role"] == "user"]
        return f"총 {len(user_messages)}개의 대화가 있습니다."