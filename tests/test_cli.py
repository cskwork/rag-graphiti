"""
CLI 명령어 End-to-End 테스트
End-to-end tests for CLI commands
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from click.testing import CliRunner

from rag_chatbot.cli import main
from rag_chatbot.config import Settings


class TestCliInit:
    """init 명령어 테스트"""
    
    def test_init_success(self):
        """정상 초기화 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.close_graphiti_service') as mock_close:
            
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service
            
            result = runner.invoke(main, ['init'])
            
            assert result.exit_code == 0
            assert "✓ Graphiti database initialized successfully" in result.output
    
    def test_init_failure(self):
        """초기화 실패 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service', side_effect=Exception("연결 실패")), \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            result = runner.invoke(main, ['init'])
            
            assert result.exit_code == 1
            assert "✗ Initialization failed" in result.output
            assert "연결 실패" in result.output
    
    def test_init_reset_cancelled(self):
        """리셋 취소 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.close_graphiti_service'), \
             patch('rag_chatbot.cli.Prompt.ask', return_value="no"):
            
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service
            
            result = runner.invoke(main, ['init', '--reset'])
            
            assert result.exit_code == 0
            assert "Operation cancelled" in result.output


class TestCliAddDocument:
    """add-doc 명령어 테스트"""
    
    def test_add_text_document(self):
        """텍스트 문서 추가 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.DocumentProcessor') as mock_processor_class, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service
            
            mock_processor = AsyncMock()
            mock_processor.add_text_document.return_value = 2
            mock_processor_class.return_value = mock_processor
            
            result = runner.invoke(main, [
                'add-doc',
                '--text', '테스트 문서 내용',
                '--title', '테스트 제목'
            ])
            
            assert result.exit_code == 0
            assert "✓ Added text document (2 chunks)" in result.output
            mock_processor.add_text_document.assert_called_once()
    
    def test_add_file_document(self):
        """파일 문서 추가 테스트"""
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("테스트 파일 내용")
            temp_file = f.name
        
        try:
            with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
                 patch('rag_chatbot.cli.DocumentProcessor') as mock_processor_class, \
                 patch('rag_chatbot.cli.close_graphiti_service'):
                
                mock_service = AsyncMock()
                mock_get_service.return_value = mock_service
                
                mock_processor = AsyncMock()
                mock_processor.add_file_document.return_value = 1
                mock_processor_class.return_value = mock_processor
                
                result = runner.invoke(main, [
                    'add-doc',
                    '--file', temp_file
                ])
                
                assert result.exit_code == 0
                assert f"✓ Added file '{temp_file}' (1 chunks)" in result.output
                mock_processor.add_file_document.assert_called_once()
        finally:
            Path(temp_file).unlink()
    
    def test_add_document_no_input(self):
        """입력이 없는 경우 테스트"""
        runner = CliRunner()
        
        result = runner.invoke(main, ['add-doc'])
        
        assert result.exit_code == 1
        assert "Error: Must provide either --file or --text" in result.output
    
    def test_add_document_failure(self):
        """문서 추가 실패 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.DocumentProcessor') as mock_processor_class, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service
            
            mock_processor = AsyncMock()
            mock_processor.add_text_document.side_effect = Exception("추가 실패")
            mock_processor_class.return_value = mock_processor
            
            result = runner.invoke(main, [
                'add-doc',
                '--text', '실패할 내용'
            ])
            
            assert result.exit_code == 1
            assert "✗ Failed to add document" in result.output
            assert "추가 실패" in result.output


class TestCliAddJson:
    """add-json 명령어 테스트"""
    
    def test_add_json_data_string(self):
        """JSON 문자열 데이터 추가 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.DocumentProcessor') as mock_processor_class, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service
            
            mock_processor = AsyncMock()
            mock_processor.add_json_data.return_value = 1
            mock_processor_class.return_value = mock_processor
            
            json_data = '{"name": "테스트", "value": 123}'
            
            result = runner.invoke(main, [
                'add-json',
                '--data', json_data,
                '--title', 'JSON 테스트'
            ])
            
            assert result.exit_code == 0
            assert "✓ Added JSON data (1 items)" in result.output
            mock_processor.add_json_data.assert_called_once()
    
    def test_add_json_file(self):
        """JSON 파일 추가 테스트"""
        runner = CliRunner()
        
        test_data = {"name": "파일 테스트", "items": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_file = f.name
        
        try:
            with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
                 patch('rag_chatbot.cli.DocumentProcessor') as mock_processor_class, \
                 patch('rag_chatbot.cli.close_graphiti_service'):
                
                mock_service = AsyncMock()
                mock_get_service.return_value = mock_service
                
                mock_processor = AsyncMock()
                mock_processor.add_file_document.return_value = 1
                mock_processor_class.return_value = mock_processor
                
                result = runner.invoke(main, [
                    'add-json',
                    '--file', temp_file
                ])
                
                assert result.exit_code == 0
                assert f"✓ Added JSON file '{temp_file}' (1 items)" in result.output
                mock_processor.add_file_document.assert_called_once()
        finally:
            Path(temp_file).unlink()
    
    def test_add_json_invalid_data(self):
        """잘못된 JSON 데이터 테스트"""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            'add-json',
            '--data', '{ invalid json'
        ])
        
        assert result.exit_code == 1
        assert "✗ Invalid JSON data" in result.output
    
    def test_add_json_no_input(self):
        """입력이 없는 경우 테스트"""
        runner = CliRunner()
        
        result = runner.invoke(main, ['add-json'])
        
        assert result.exit_code == 1
        assert "Error: Must provide either --file or --data" in result.output


class TestCliSearch:
    """search 명령어 테스트"""
    
    def test_search_basic(self):
        """기본 검색 테스트"""
        runner = CliRunner()
        
        mock_results = [
            MagicMock(fact="첫 번째 결과", valid_at=None),
            MagicMock(fact="두 번째 결과", valid_at=None)
        ]
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_service.search.return_value = mock_results
            mock_get_service.return_value = mock_service
            
            result = runner.invoke(main, [
                'search', '테스트 쿼리'
            ])
            
            assert result.exit_code == 0
            assert "첫 번째 결과" in result.output
            assert "두 번째 결과" in result.output
            mock_service.search.assert_called_once_with(
                query="테스트 쿼리",
                max_results=5,
                center_node_uuid=None
            )
    
    def test_search_with_user_id(self):
        """사용자 ID가 있는 검색 테스트"""
        runner = CliRunner()
        
        mock_user_node = MagicMock(uuid="user-node-123")
        mock_results = [MagicMock(fact="개인화된 결과", valid_at=None)]
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_service.node_search.return_value = [mock_user_node]
            mock_service.search.return_value = mock_results
            mock_get_service.return_value = mock_service
            
            result = runner.invoke(main, [
                'search', '개인화 쿼리',
                '--user-id', 'test-user',
                '--max-results', '10'
            ])
            
            assert result.exit_code == 0
            assert "개인화된 결과" in result.output
            mock_service.search.assert_called_once_with(
                query="개인화 쿼리",
                max_results=10,
                center_node_uuid="user-node-123"
            )
    
    def test_search_no_results(self):
        """검색 결과가 없는 경우 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_service.search.return_value = []
            mock_get_service.return_value = mock_service
            
            result = runner.invoke(main, [
                'search', '없는 쿼리'
            ])
            
            assert result.exit_code == 0
            assert "No results found for: '없는 쿼리'" in result.output
    
    def test_search_failure(self):
        """검색 실패 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_service.search.side_effect = Exception("검색 실패")
            mock_get_service.return_value = mock_service
            
            result = runner.invoke(main, [
                'search', '실패 쿼리'
            ])
            
            assert result.exit_code == 1
            assert "✗ Search failed" in result.output
            assert "검색 실패" in result.output


class TestCliChat:
    """chat 명령어 테스트"""
    
    def test_chat_single_query(self):
        """단일 질의 모드 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.ChatHandler') as mock_chat_handler_class, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service
            
            mock_chat_handler = AsyncMock()
            mock_chat_handler.process_query.return_value = "테스트 응답"
            mock_chat_handler_class.return_value = mock_chat_handler
            
            result = runner.invoke(main, [
                'chat',
                '--query', '테스트 질문'
            ])
            
            assert result.exit_code == 0
            assert "테스트 응답" in result.output
            mock_chat_handler.process_query.assert_called_once_with("테스트 질문", None)
    
    def test_chat_with_user_id(self):
        """사용자 ID가 있는 채팅 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.ChatHandler') as mock_chat_handler_class, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service
            
            mock_chat_handler = AsyncMock()
            mock_chat_handler.process_query.return_value = "개인화된 응답"
            mock_chat_handler_class.return_value = mock_chat_handler
            
            result = runner.invoke(main, [
                'chat',
                '--user-id', 'test-user',
                '--query', '개인화 질문'
            ])
            
            assert result.exit_code == 0
            assert "개인화된 응답" in result.output
            mock_chat_handler.process_query.assert_called_once_with("개인화 질문", "test-user")
    
    def test_chat_failure(self):
        """채팅 실패 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service', side_effect=Exception("채팅 서비스 실패")), \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            result = runner.invoke(main, [
                'chat',
                '--query', '실패 질문'
            ])
            
            assert result.exit_code == 1
            assert "✗ Chat failed" in result.output
            assert "채팅 서비스 실패" in result.output


class TestCliStatus:
    """status 명령어 테스트"""
    
    def test_status_healthy(self):
        """정상 상태 확인 테스트"""
        runner = CliRunner()
        
        mock_health_status = {
            "service": "graphiti",
            "status": "healthy",
            "connection_ready": True,
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_service.get_health_status.return_value = mock_health_status
            mock_get_service.return_value = mock_service
            
            result = runner.invoke(main, ['status'])
            
            assert result.exit_code == 0
            assert "healthy" in result.output
            assert "System Status" in result.output
    
    def test_status_unhealthy(self):
        """비정상 상태 확인 테스트"""
        runner = CliRunner()
        
        mock_health_status = {
            "service": "graphiti",
            "status": "unhealthy",
            "connection_ready": False,
            "error": "연결 오류"
        }
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            mock_service = AsyncMock()
            mock_service.get_health_status.return_value = mock_health_status
            mock_get_service.return_value = mock_service
            
            result = runner.invoke(main, ['status'])
            
            assert result.exit_code == 0
            assert "unhealthy" in result.output
            assert "연결 오류" in result.output
    
    def test_status_failure(self):
        """상태 확인 실패 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service', side_effect=Exception("상태 확인 실패")), \
             patch('rag_chatbot.cli.close_graphiti_service'):
            
            result = runner.invoke(main, ['status'])
            
            assert result.exit_code == 1
            assert "✗ Status check failed" in result.output
            assert "상태 확인 실패" in result.output


class TestCliServe:
    """serve 명령어 테스트"""
    
    def test_serve_success(self):
        """웹 서버 시작 성공 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.uvicorn') as mock_uvicorn, \
             patch('rag_chatbot.cli.create_app') as mock_create_app:
            
            mock_app = MagicMock()
            mock_create_app.return_value = mock_app
            
            result = runner.invoke(main, [
                'serve',
                '--host', '127.0.0.1',
                '--port', '9000'
            ])
            
            # uvicorn.run이 호출되었는지 확인
            mock_uvicorn.run.assert_called_once()
            call_args = mock_uvicorn.run.call_args
            assert call_args[0][0] is mock_app  # app
            assert call_args[1]['host'] == '127.0.0.1'
            assert call_args[1]['port'] == 9000
    
    def test_serve_import_error(self):
        """웹 서버 의존성 누락 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.uvicorn', side_effect=ImportError()):
            
            result = runner.invoke(main, ['serve'])
            
            assert result.exit_code == 1
            assert "Web server dependencies not installed" in result.output
    
    def test_serve_failure(self):
        """웹 서버 시작 실패 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.uvicorn') as mock_uvicorn, \
             patch('rag_chatbot.cli.create_app'):
            
            mock_uvicorn.run.side_effect = Exception("서버 시작 실패")
            
            result = runner.invoke(main, ['serve'])
            
            assert result.exit_code == 1
            assert "✗ Failed to start web server" in result.output
            assert "서버 시작 실패" in result.output


class TestCliVerboseMode:
    """verbose 모드 테스트"""
    
    def test_verbose_flag(self):
        """verbose 플래그 테스트"""
        runner = CliRunner()
        
        with patch('rag_chatbot.cli.get_graphiti_service') as mock_get_service, \
             patch('rag_chatbot.cli.close_graphiti_service'), \
             patch('rag_chatbot.cli.setup_logging') as mock_setup_logging:
            
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service
            
            result = runner.invoke(main, ['-v', 'init'])
            
            # setup_logging이 호출되었는지 확인
            mock_setup_logging.assert_called_once()
            
            # 설정에서 log_level이 DEBUG로 변경되었는지 확인
            call_args = mock_setup_logging.call_args[0][0]
            assert call_args.log_level == "DEBUG"