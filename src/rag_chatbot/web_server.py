"""
Simple web interface for RAG chatbot.
RAG 채팅봇을 위한 간단한 웹 인터페이스
"""

import logging
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from .chat_handler import ChatHandler
from .config import Settings
from .graphiti_service import get_graphiti_service

logger = logging.getLogger(__name__)

# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Chatbot</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            min-height: 400px;
            margin-bottom: 20px;
        }
        .chat-messages {
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 15px;
            background-color: #fafafa;
        }
        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 8px;
            max-width: 80%;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: auto;
            text-align: right;
        }
        .assistant-message {
            background-color: #f1f8e9;
            margin-right: auto;
        }
        .input-form {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .input-field {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        .input-field:focus {
            border-color: #667eea;
        }
        .send-button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        .send-button:hover {
            transform: translateY(-1px);
        }
        .status {
            text-align: center;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status.error {
            background-color: #ffebee;
            color: #c62828;
            border: 1px solid #e57373;
        }
        .status.success {
            background-color: #e8f5e8;
            color: #2e7d32;
            border: 1px solid #81c784;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 RAG Chatbot</h1>
        <p>Graphiti 지식 그래프 기반 챗봇</p>
    </div>

    {% if status_message %}
    <div class="status {{ status_type }}">
        {{ status_message }}
    </div>
    {% endif %}

    <div class="chat-container">
        <div class="chat-messages" id="chatMessages">
            {% if conversation %}
                {% for message in conversation %}
                <div class="message {{ 'user-message' if message.role == 'user' else 'assistant-message' }}">
                    <strong>{{ 'You' if message.role == 'user' else 'Assistant' }}:</strong>
                    {{ message.content }}
                </div>
                {% endfor %}
            {% else %}
            <div class="message assistant-message">
                <strong>Assistant:</strong> 안녕하세요! 무엇을 도와드릴까요? 지식 베이스에서 정보를 검색해드립니다.
            </div>
            {% endif %}
        </div>

        <form method="post" class="input-form">
            <input type="text" name="user_input" class="input-field" 
                   placeholder="질문을 입력하세요..." required autocomplete="off">
            <input type="hidden" name="user_id" value="{{ user_id }}">
            <button type="submit" class="send-button">전송</button>
        </form>
    </div>

    <div class="footer">
        <p>Powered by Graphiti 0.17.4 Knowledge Graph</p>
        <p><a href="/status" style="color: #667eea;">시스템 상태 확인</a></p>
    </div>

    <script>
        // 자동 스크롤
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // 폼 제출시 입력창 클리어
        document.querySelector('form').addEventListener('submit', function() {
            setTimeout(() => {
                document.querySelector('input[name="user_input"]').value = '';
            }, 100);
        });
    </script>
</body>
</html>
"""

STATUS_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Status - RAG Chatbot</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #ddd;
        }
        .status-healthy {
            background-color: #e8f5e8;
            border-left-color: #4caf50;
        }
        .status-warning {
            background-color: #fff3e0;
            border-left-color: #ff9800;
        }
        .status-error {
            background-color: #ffebee;
            border-left-color: #f44336;
        }
        .back-link {
            text-align: center;
            margin-top: 20px;
        }
        .back-link a {
            color: #667eea;
            text-decoration: none;
            padding: 10px 20px;
            border: 2px solid #667eea;
            border-radius: 5px;
            transition: all 0.3s;
        }
        .back-link a:hover {
            background-color: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 System Status</h1>
        <p>RAG Chatbot 시스템 상태</p>
    </div>

    <div class="status-container">
        {% for item in status_items %}
        <div class="status-item status-{{ item.status }}">
            <div>
                <strong>{{ item.component }}</strong><br>
                <small>{{ item.details }}</small>
            </div>
            <div>
                <span style="font-weight: bold; color: {{ item.color }};">{{ item.status_text }}</span>
            </div>
        </div>
        {% endfor %}
        
        {% if error_message %}
        <div class="status-item status-error">
            <div>
                <strong>Error Details</strong><br>
                <small>{{ error_message }}</small>
            </div>
        </div>
        {% endif %}
    </div>

    <div class="back-link">
        <a href="/">← 채팅으로 돌아가기</a>
    </div>
</body>
</html>
"""


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
        # 간단한 템플릿 변수 치환을 위한 문자열 렌더링
        html_content = HTML_TEMPLATE
        
        # 상태 메시지 처리
        status_section = ""
        
        # 대화 내용 렌더링
        conversation_html = ""
        if app.state.conversation_history:
            for message in app.state.conversation_history:
                role_class = "user-message" if message.get("role") == "user" else "assistant-message"
                role_name = "You" if message.get("role") == "user" else "Assistant"
                conversation_html += f'''
                <div class="message {role_class}">
                    <strong>{role_name}:</strong>
                    {message.get("content", "")}
                </div>'''
        else:
            conversation_html = '''
            <div class="message assistant-message">
                <strong>Assistant:</strong> 안녕하세요! 무엇을 도와드릴까요? 지식 베이스에서 정보를 검색해드립니다.
            </div>'''
        
        # 템플릿 변수 치환
        html_content = html_content.replace("{% if status_message %}", "")
        html_content = html_content.replace("{% endif %}", "")
        html_content = html_content.replace("{{ status_type }}", "")
        html_content = html_content.replace("{{ status_message }}", "")
        html_content = html_content.replace("{% if conversation %}", "")
        html_content = html_content.replace("{% for message in conversation %}", "")
        html_content = html_content.replace("{{ 'user-message' if message.role == 'user' else 'assistant-message' }}", "")
        html_content = html_content.replace("{{ 'You' if message.role == 'user' else 'Assistant' }}", "")
        html_content = html_content.replace("{{ message.content }}", "")
        html_content = html_content.replace("{% endfor %}", "")
        html_content = html_content.replace("{% else %}", "")
        html_content = html_content.replace("{{ user_id }}", "web_user")
        
        # 대화 내용 삽입
        html_content = html_content.replace(
            '''            {% if conversation %}
                {% for message in conversation %}
                <div class="message {{ 'user-message' if message.role == 'user' else 'assistant-message' }}">
                    <strong>{{ 'You' if message.role == 'user' else 'Assistant' }}:</strong>
                    {{ message.content }}
                </div>
                {% endfor %}
            {% else %}
            <div class="message assistant-message">
                <strong>Assistant:</strong> 안녕하세요! 무엇을 도와드릴까요? 지식 베이스에서 정보를 검색해드립니다.
            </div>
            {% endif %}''',
            conversation_html
        )
        
        return HTMLResponse(content=html_content)
    
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
            
            return HTMLResponse(
                content=HTML_TEMPLATE.replace(
                    "{% if conversation %}", ""
                ).replace(
                    "{% for message in conversation %}", 
                    "".join([
                        f'<div class="message {"user-message" if msg["role"] == "user" else "assistant-message"}">'
                        f'<strong>{"You" if msg["role"] == "user" else "Assistant"}:</strong> {msg["content"]}</div>'
                        for msg in app.state.conversation_history
                    ])
                ).replace(
                    "{% endfor %}", ""
                ).replace(
                    "{% else %}", "<!--"
                ).replace(
                    "{% endif %}", "-->"
                ).replace(
                    "{{ user_id }}", user_id
                ).replace(
                    "{% if status_message %}", "<!--"
                ).replace(
                    "{% endif %}", "-->"
                )
            )
            
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            
            return HTMLResponse(
                content=HTML_TEMPLATE.replace(
                    "{% if status_message %}", ""
                ).replace(
                    "{{ status_message }}", f"오류가 발생했습니다: {str(e)}"
                ).replace(
                    "{{ status_type }}", "error"
                ).replace(
                    "{% endif %}", ""
                ).replace(
                    "{{ user_id }}", user_id
                ).replace(
                    "{% if conversation %}", "<!--"
                ).replace(
                    "{% endfor %}", "-->"
                )
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
                "details": f"{settings.falkordb_host}:{settings.falkordb_port}",
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
            
            return HTMLResponse(
                content=STATUS_TEMPLATE.replace(
                    "{% for item in status_items %}", ""
                ).replace(
                    "{% endfor %}", ""
                ).replace(
                    "{% if error_message %}", "" if error_message else "<!--"
                ).replace(
                    "{% endif %}", "" if error_message else "-->"
                ).replace(
                    "{{ error_message }}", error_message or ""
                ) + "".join([
                    f'<div class="status-item status-{item["status"]}">'
                    f'<div><strong>{item["component"]}</strong><br><small>{item["details"]}</small></div>'
                    f'<div><span style="font-weight: bold; color: {item["color"]};">{item["status_text"]}</span></div></div>'
                    for item in status_items
                ])
            )
            
        except Exception as e:
            return HTMLResponse(
                content=f"<h1>Status Error</h1><p>Failed to get status: {e}</p>",
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