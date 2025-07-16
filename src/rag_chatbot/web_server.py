"""
Simple web interface for RAG chatbot.
RAG 채팅봇을 위한 간단한 웹 인터페이스
"""

import logging
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .chat_handler import ChatHandler
from .config import Settings
from .graphiti_service import get_graphiti_service

logger = logging.getLogger(__name__)

# 템플릿 디렉토리 설정 및 헬퍼 함수
def get_templates_dir() -> Path:
    """Get templates directory path."""
    return Path(__file__).parent / "templates"


def create_templates_instance() -> Jinja2Templates:
    """Create Jinja2Templates instance with proper directory."""
    templates_dir = get_templates_dir()
    if not templates_dir.exists():
        raise RuntimeError(f"Templates directory not found: {templates_dir}")
    return Jinja2Templates(directory=str(templates_dir))


def create_app(settings: Settings) -> FastAPI:
    """
    Create FastAPI application.
    FastAPI 애플리케이션 생성
    """
    app = FastAPI(
        title="RAG Chatbot",
        description="Production-ready RAG chatbot using Graphiti knowledge graph",
        version="0.1.0"
    )
    
    # 템플릿 설정
    templates = create_templates_instance()
    
    # 전역 변수로 서비스 관리
    app.state.graphiti_service = None
    app.state.chat_handler = None
    app.state.conversation_history = []
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        try:
            app.state.graphiti_service = await get_graphiti_service(settings)
            app.state.chat_handler = ChatHandler(app.state.graphiti_service, settings)
            logger.info("Web server services initialized")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup services on shutdown."""
        from .graphiti_service import close_graphiti_service
        await close_graphiti_service()
        logger.info("Web server services closed")
    
    @app.get("/", response_class=HTMLResponse)
    async def chat_interface(request: Request):
        """Main chat interface."""
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "conversation": app.state.conversation_history,
                "user_id": "web_user",
                "status_message": None,
                "status_type": None
            }
        )
    
    @app.post("/", response_class=HTMLResponse)
    async def process_chat(request: Request, user_input: str = Form(...), user_id: str = Form("web_user")):
        """Process chat message."""
        try:
            if not app.state.chat_handler:
                raise HTTPException(status_code=503, detail="Chat service not available")
            
            # 사용자 메시지 추가
            app.state.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # 응답 생성
            response = await app.state.chat_handler.process_query(user_input, user_id)
            
            # 어시스턴트 응답 추가
            app.state.conversation_history.append({
                "role": "assistant", 
                "content": response
            })
            
            # 기록 크기 제한 (최근 20개 메시지)
            if len(app.state.conversation_history) > 20:
                app.state.conversation_history = app.state.conversation_history[-20:]
            
            return templates.TemplateResponse(
                "chat.html",
                {
                    "request": request,
                    "conversation": app.state.conversation_history,
                    "user_id": user_id,
                    "status_message": None,
                    "status_type": None
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            
            return templates.TemplateResponse(
                "chat.html",
                {
                    "request": request,
                    "conversation": app.state.conversation_history,
                    "user_id": user_id,
                    "status_message": f"오류가 발생했습니다: {str(e)}",
                    "status_type": "error"
                }
            )
    
    @app.get("/status", response_class=HTMLResponse)
    async def system_status(request: Request):
        """System status page."""
        try:
            status_items = []
            error_message = None
            
            if app.state.graphiti_service:
                health_status = await app.state.graphiti_service.get_health_status()
                
                # Graphiti 상태
                if health_status["status"] == "healthy":
                    graphiti_status = "healthy"
                    graphiti_color = "#4caf50"
                    graphiti_text = "정상"
                else:
                    graphiti_status = "error"
                    graphiti_color = "#f44336"
                    graphiti_text = "오류"
                    if "error" in health_status:
                        error_message = health_status["error"]
                
                status_items.append({
                    "component": "Graphiti Service",
                    "details": f"연결 상태: {health_status['connection_ready']}",
                    "status": graphiti_status,
                    "status_text": graphiti_text,
                    "color": graphiti_color
                })
            else:
                status_items.append({
                    "component": "Graphiti Service", 
                    "details": "서비스가 초기화되지 않음",
                    "status": "error",
                    "status_text": "오류",
                    "color": "#f44336"
                })
            
            # FalkorDB 설정
            status_items.append({
                "component": "FalkorDB Connection",
                "details": f"{settings.falkor_host}:{settings.falkor_port}",
                "status": "healthy",
                "status_text": "설정됨",
                "color": "#2196f3"
            })
            
            # LLM 제공자
            llm_providers = []
            if settings.openai_api_key:
                llm_providers.append("OpenAI")
            if settings.anthropic_api_key:
                llm_providers.append("Anthropic")
            if settings.google_api_key:
                llm_providers.append("Google")
            
            llm_text = ", ".join(llm_providers) if llm_providers else "없음"
            llm_status = "warning" if not llm_providers else "healthy"
            llm_color = "#ff9800" if not llm_providers else "#4caf50"
            
            status_items.append({
                "component": "LLM Providers",
                "details": f"사용 가능: {llm_text}",
                "status": llm_status,
                "status_text": "사용 가능" if llm_providers else "설정 필요",
                "color": llm_color
            })
            
            return templates.TemplateResponse(
                "status.html",
                {
                    "request": request,
                    "status_items": status_items,
                    "error_message": error_message
                }
            )
            
        except Exception as e:
            logger.error(f"Status page error: {e}")
            return templates.TemplateResponse(
                "status.html",
                {
                    "request": request,
                    "status_items": [],
                    "error_message": f"Failed to get status: {e}"
                },
                status_code=500
            )
    
    @app.get("/health")
    async def health_check():
        """Simple health check endpoint."""
        if app.state.graphiti_service:
            try:
                health_status = await app.state.graphiti_service.get_health_status()
                return {"status": "ok", "graphiti": health_status}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": "Services not initialized"}
    
    return app


if __name__ == "__main__":
    # 개발 서버 실행
    from .config import get_settings
    
    settings = get_settings()
    app = create_app(settings)
    
    uvicorn.run(
        app,
        host=settings.web_host,
        port=settings.web_port,
        reload=settings.web_reload
    )