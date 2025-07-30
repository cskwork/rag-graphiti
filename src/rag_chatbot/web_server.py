"""
Simple web interface for RAG chatbot.
RAG 채팅봇을 위한 간단한 웹 인터페이스
"""

import logging
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse

from .chat_handler import ChatHandler
from .config import Settings
from .document_processor import DocumentProcessor
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
        .nav-links {
            text-align: center;
            margin: 20px 0;
        }
        .nav-links a {
            color: #667eea;
            text-decoration: none;
            margin: 0 15px;
            padding: 10px 20px;
            border: 2px solid #667eea;
            border-radius: 25px;
            transition: all 0.3s;
            display: inline-block;
        }
        .nav-links a:hover {
            background-color: #667eea;
            color: white;
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

    <div class="nav-links">
        <a href="/upload">📤 데이터 업로드</a>
        <a href="/data">📊 데이터 관리</a>
        <a href="/status">⚙️ 시스템 상태</a>
        <a href="{falkordb_url}" target="_blank">🌐 그래프 뷰어</a>
    </div>

    <div class="footer">
        <p>Powered by Graphiti 0.17.4 Knowledge Graph</p>
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

UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>데이터 업로드 - RAG Chatbot</title>
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
        .nav-links {
            text-align: center;
            margin-bottom: 20px;
        }
        .nav-links a {
            color: #667eea;
            text-decoration: none;
            margin: 0 15px;
            padding: 10px 20px;
            border: 2px solid #667eea;
            border-radius: 25px;
            transition: all 0.3s;
        }
        .nav-links a:hover {
            background-color: #667eea;
            color: white;
        }
        .upload-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .upload-section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .upload-section h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        .form-control {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        .form-control:focus {
            border-color: #667eea;
            outline: none;
        }
        textarea.form-control {
            min-height: 100px;
            resize: vertical;
        }
        .upload-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        .upload-button:hover {
            transform: translateY(-1px);
        }
        .status {
            text-align: center;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .status.success {
            background-color: #e8f5e8;
            color: #2e7d32;
            border: 1px solid #81c784;
        }
        .status.error {
            background-color: #ffebee;
            color: #c62828;
            border: 1px solid #e57373;
        }
        .file-info {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📤 데이터 업로드</h1>
        <p>지식 그래프에 새로운 데이터를 추가하세요</p>
    </div>

    <div class="nav-links">
        <a href="/">💬 채팅</a>
        <a href="/data">📊 데이터 관리</a>
        <a href="/status">⚙️ 시스템 상태</a>
        <a href="{falkordb_url}" target="_blank">🌐 그래프 뷰어</a>
    </div>

    {% if status_message %}
    <div class="status {{ status_type }}">
        {{ status_message }}
    </div>
    {% endif %}

    <div class="upload-container">
        <!-- 텍스트 직접 입력 -->
        <div class="upload-section">
            <h3>📝 텍스트 직접 입력</h3>
            <form method="post" enctype="application/x-www-form-urlencoded">
                <input type="hidden" name="upload_type" value="text">
                <div class="form-group">
                    <label for="title">제목</label>
                    <input type="text" id="title" name="title" class="form-control" placeholder="문서 제목을 입력하세요">
                </div>
                <div class="form-group">
                    <label for="content">내용</label>
                    <textarea id="content" name="content" class="form-control" placeholder="문서 내용을 입력하세요" required></textarea>
                </div>
                <div class="form-group">
                    <label for="source">소스 설명</label>
                    <input type="text" id="source" name="source" class="form-control" value="web_upload" placeholder="데이터 소스 설명">
                </div>
                <button type="submit" class="upload-button">텍스트 추가</button>
            </form>
        </div>

        <!-- 파일 업로드 -->
        <div class="upload-section">
            <h3>📁 파일 업로드</h3>
            <form method="post" enctype="multipart/form-data">
                <input type="hidden" name="upload_type" value="file">
                <div class="form-group">
                    <label for="file">파일 선택</label>
                    <input type="file" id="file" name="file" class="form-control" accept=".txt,.md,.json" required>
                    <div class="file-info">지원 형식: .txt, .md, .json</div>
                </div>
                <div class="form-group">
                    <label for="file_source">소스 설명</label>
                    <input type="text" id="file_source" name="source" class="form-control" value="file_upload" placeholder="파일 소스 설명">
                </div>
                <button type="submit" class="upload-button">파일 업로드</button>
            </form>
        </div>

        <!-- URL 입력 -->
        <div class="upload-section">
            <h3>🌐 URL에서 가져오기</h3>
            <form method="post" enctype="application/x-www-form-urlencoded">
                <input type="hidden" name="upload_type" value="url">
                <div class="form-group">
                    <label for="url">URL</label>
                    <input type="url" id="url" name="url" class="form-control" placeholder="https://example.com/document" required>
                </div>
                <div class="form-group">
                    <label for="url_title">제목 (선택사항)</label>
                    <input type="text" id="url_title" name="title" class="form-control" placeholder="문서 제목 (자동 생성됨)">
                </div>
                <div class="form-group">
                    <label for="url_source">소스 설명</label>
                    <input type="text" id="url_source" name="source" class="form-control" value="web_url" placeholder="URL 소스 설명">
                </div>
                <button type="submit" class="upload-button">URL 가져오기</button>
            </form>
        </div>
    </div>

    <script>
        // 파일 선택 시 파일명 표시
        document.getElementById('file').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                console.log('Selected file:', file.name, file.type, file.size);
            }
        });
    </script>
</body>
</html>
"""

DATA_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>데이터 관리 - RAG Chatbot</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
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
        .nav-links {
            text-align: center;
            margin-bottom: 20px;
        }
        .nav-links a {
            color: #667eea;
            text-decoration: none;
            margin: 0 15px;
            padding: 10px 20px;
            border: 2px solid #667eea;
            border-radius: 25px;
            transition: all 0.3s;
        }
        .nav-links a:hover {
            background-color: #667eea;
            color: white;
        }
        .stats-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .search-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .search-form {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .search-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
        }
        .search-input:focus {
            border-color: #667eea;
            outline: none;
        }
        .search-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
        }
        .results-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .results-table th {
            background-color: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
            font-weight: bold;
        }
        .results-table td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }
        .results-table tr:hover {
            background-color: #f8f9fa;
        }
        .fact-content {
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .date-cell {
            color: #666;
            font-size: 0.9em;
        }
        .no-results {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 데이터 관리</h1>
        <p>지식 그래프의 데이터를 조회하고 관리하세요</p>
    </div>

    <div class="nav-links">
        <a href="/">💬 채팅</a>
        <a href="/upload">📤 데이터 업로드</a>
        <a href="/status">⚙️ 시스템 상태</a>
    </div>

    <div class="stats-container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ total_facts }}</div>
                <div class="stat-label">총 Facts</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_nodes }}</div>
                <div class="stat-label">총 Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_edges }}</div>
                <div class="stat-label">총 Edges</div>
            </div>
        </div>
    </div>

    <div class="search-container">
        <h3>💡 지식 검색</h3>
        <form method="get" class="search-form">
            <input type="text" name="search_query" class="search-input" 
                   placeholder="검색어를 입력하세요..." 
                   value="{{ search_query }}">
            <button type="submit" class="search-button">검색</button>
        </form>
    </div>

    <div class="results-container">
        <h3>📋 검색 결과</h3>
        {% if results %}
        <table class="results-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Fact</th>
                    <th>생성일</th>
                </tr>
            </thead>
            <tbody>
                {% for result in results %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td class="fact-content" title="{{ result.fact }}">{{ result.fact }}</td>
                    <td class="date-cell">{{ result.created_at }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <div class="no-results">
            {% if search_query %}
                '{{ search_query }}'에 대한 검색 결과가 없습니다.
            {% else %}
                검색어를 입력하여 지식 그래프의 데이터를 조회하세요.
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def get_falkordb_url(settings: Settings) -> str:
    """
    Construct FalkorDB web interface URL.
    FalkorDB 웹 인터페이스 URL 생성
    """
    return f"https://browser.falkordb.com/graph"#"http://{settings.falkordb_host}:{settings.falkordb_port}"


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
    app.state.document_processor = None
    app.state.conversation_history = []
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        try:
            app.state.graphiti_service = await get_graphiti_service(settings)
            app.state.chat_handler = ChatHandler(app.state.graphiti_service, settings)
            app.state.document_processor = DocumentProcessor(app.state.graphiti_service, settings)
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
        html_content = html_content.replace("{falkordb_url}", get_falkordb_url(settings))
        
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
                ).replace(
                    "{falkordb_url}", get_falkordb_url(settings)
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
                ).replace(
                    "{falkordb_url}", get_falkordb_url(settings)
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
    
    @app.get("/upload", response_class=HTMLResponse)
    async def upload_page(request: Request, status_message: str = None, status_type: str = None):
        """데이터 업로드 페이지"""
        html_content = UPLOAD_TEMPLATE
        
        # 상태 메시지 처리
        if status_message:
            html_content = html_content.replace("{% if status_message %}", "")
            html_content = html_content.replace("{{ status_message }}", status_message)
            html_content = html_content.replace("{{ status_type }}", status_type or "")
            html_content = html_content.replace("{% endif %}", "")
        else:
            # 상태 메시지 섹션 제거
            html_content = html_content.replace("{% if status_message %}", "<!--")
            html_content = html_content.replace("{% endif %}", "-->")
            html_content = html_content.replace("{{ status_message }}", "")
            html_content = html_content.replace("{{ status_type }}", "")
        
        return HTMLResponse(content=html_content)
    
    @app.post("/upload", response_class=HTMLResponse)
    async def handle_upload(
        request: Request,
        upload_type: str = Form(...),
        # 텍스트 업로드용
        title: str = Form(None),
        content: str = Form(None),
        source: str = Form("web_upload"),
        # 파일 업로드용
        file: UploadFile = File(None),
        # URL 업로드용
        url: str = Form(None)
    ):
        """업로드 요청 처리"""
        try:
            if not app.state.document_processor:
                raise HTTPException(status_code=503, detail="Document processor not available")
            
            chunks = 0
            
            if upload_type == "text":
                # 텍스트 직접 입력 처리
                if not content or not content.strip():
                    raise ValueError("텍스트 내용이 필요합니다")
                
                chunks = await app.state.document_processor.add_text_document(
                    content=content,
                    title=title,
                    source_description=source
                )
                message = f"텍스트 문서가 성공적으로 추가되었습니다 ({chunks}개 청크)"
                
            elif upload_type == "file":
                # 파일 업로드 처리
                if not file:
                    raise ValueError("업로드할 파일이 필요합니다")
                
                # 파일 내용 읽기
                file_content = await file.read()
                
                # 임시 파일 저장 후 처리
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=f".{file.filename.split('.')[-1]}"
                ) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                try:
                    chunks = await app.state.document_processor.add_file_document(
                        file_path=temp_file_path,
                        source_description=source
                    )
                    message = f"파일 '{file.filename}'이 성공적으로 추가되었습니다 ({chunks}개 청크)"
                finally:
                    # 임시 파일 삭제
                    os.unlink(temp_file_path)
                
            elif upload_type == "url":
                # URL 처리  
                if not url:
                    raise ValueError("URL이 필요합니다")
                
                chunks = await app.state.document_processor.add_url_document(
                    url=url,
                    title=title,
                    source_description=source
                )
                message = f"URL 문서가 성공적으로 추가되었습니다 ({chunks}개 청크)"
            
            else:
                raise ValueError("지원하지 않는 업로드 타입입니다")
            
            # 성공 메시지와 함께 페이지 재렌더링
            return await upload_page(request, status_message=message, status_type="success")
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            error_message = f"업로드 실패: {str(e)}"
            return await upload_page(request, status_message=error_message, status_type="error")
    
    @app.get("/data", response_class=HTMLResponse)
    async def data_management_page(request: Request, search_query: str = None):
        """데이터 관리 페이지"""
        try:
            if not app.state.graphiti_service:
                raise HTTPException(status_code=503, detail="Graphiti service not available")
            
            # 기본 통계 정보 (임시로 더미 데이터 사용)
            total_facts = 0
            total_nodes = 0
            total_edges = 0
            results = []
            
            # 검색이 있는 경우 결과 조회
            if search_query and search_query.strip():
                try:
                    search_results = await app.state.graphiti_service.search(
                        query=search_query.strip(),
                        max_results=20
                    )
                    # 검색 결과를 템플릿에 맞게 변환
                    results = []
                    for result in search_results:
                        results.append({
                            'fact': result.fact,
                            'created_at': str(result.valid_at)[:19] if hasattr(result, 'valid_at') and result.valid_at else 'N/A'
                        })
                    total_facts = len(results)
                except Exception as e:
                    logger.error(f"Search error: {e}")
                    results = []
            
            # HTML 템플릿 렌더링
            html_content = DATA_TEMPLATE
            
            # 템플릿 변수 치환
            html_content = html_content.replace("{{ total_facts }}", str(total_facts))
            html_content = html_content.replace("{{ total_nodes }}", str(total_nodes))  
            html_content = html_content.replace("{{ total_edges }}", str(total_edges))
            html_content = html_content.replace("{{ search_query }}", search_query or "")
            
            # 검색 결과 처리
            if results:
                # 결과가 있는 경우
                html_content = html_content.replace("{% if results %}", "")
                html_content = html_content.replace("{% else %}", "<!--")
                html_content = html_content.replace("{% endif %}", "-->")
                
                # 결과 테이블 생성
                results_html = ""
                for i, result in enumerate(results, 1):
                    results_html += f"""
                    <tr>
                        <td>{i}</td>
                        <td class="fact-content" title="{result['fact']}">{result['fact']}</td>
                        <td class="date-cell">{result['created_at']}</td>
                    </tr>"""
                
                html_content = html_content.replace("{% for result in results %}", "")
                html_content = html_content.replace("{{ loop.index }}", "")
                html_content = html_content.replace("{{ result.fact }}", "")  
                html_content = html_content.replace("{{ result.created_at }}", "")
                html_content = html_content.replace("{% endfor %}", results_html)
                
            else:
                # 결과가 없는 경우
                html_content = html_content.replace("{% if results %}", "<!--")
                html_content = html_content.replace("{% else %}", "")
                html_content = html_content.replace("{% endif %}", "")
                html_content = html_content.replace("{% for result in results %}", "<!--")
                html_content = html_content.replace("{% endfor %}", "-->")
                
                # 검색 메시지 처리
                if search_query:
                    html_content = html_content.replace("{% if search_query %}", "")
                    html_content = html_content.replace("{% else %}", "<!--")
                else:
                    html_content = html_content.replace("{% if search_query %}", "<!--")
                    html_content = html_content.replace("{% else %}", "")
                
                # 남은 중괄호 제거
                html_content = html_content.replace("{{ loop.index }}", "")
                html_content = html_content.replace("{{ result.fact }}", "")
                html_content = html_content.replace("{{ result.created_at }}", "")
            
            return HTMLResponse(content=html_content)
                
        except Exception as e:
            logger.error(f"Data management page error: {e}")
            error_html = f"""
            <html><body>
            <h1>오류 발생</h1>
            <p>데이터 관리 페이지를 로드하는 중 오류가 발생했습니다: {str(e)}</p>
            <a href="/">홈으로 돌아가기</a>
            </body></html>
            """
            return HTMLResponse(content=error_html, status_code=500)
    
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