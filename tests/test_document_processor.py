"""
DocumentProcessor 모듈 테스트
Tests for document processor module
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from rag_chatbot.document_processor import DocumentProcessor
from rag_chatbot.config import Settings


class TestDocumentProcessor:
    """DocumentProcessor 클래스 테스트"""
    
    @pytest.fixture
    def mock_graphiti_service(self):
        """Mock GraphitiService fixture"""
        service = AsyncMock()
        service.add_text_episode = AsyncMock()
        service.add_json_episode = AsyncMock()
        return service
    
    @pytest.fixture
    def processor(self, mock_graphiti_service, mock_settings):
        """DocumentProcessor fixture"""
        return DocumentProcessor(mock_graphiti_service, mock_settings)
    
    async def test_add_text_document_simple(self, processor, mock_graphiti_service):
        """간단한 텍스트 문서 추가 테스트"""
        content = "이것은 테스트 문서입니다."
        title = "테스트 문서"
        
        chunks_added = await processor.add_text_document(content, title)
        
        assert chunks_added == 1
        mock_graphiti_service.add_text_episode.assert_called_once_with(
            name=title,
            content=content,
            source_description="document"
        )
    
    async def test_add_text_document_auto_title(self, processor, mock_graphiti_service):
        """자동 제목 생성 테스트"""
        content = "제목이 없는 문서"
        
        with patch('rag_chatbot.document_processor.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            
            chunks_added = await processor.add_text_document(content)
            
            assert chunks_added == 1
            mock_graphiti_service.add_text_episode.assert_called_once_with(
                name="document_20240101_120000",
                content=content,
                source_description="document"
            )
    
    async def test_add_text_document_empty_content(self, processor):
        """빈 내용 문서 처리 테스트"""
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            await processor.add_text_document("")
        
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            await processor.add_text_document("   ")
    
    async def test_add_text_document_chunked(self, processor, mock_graphiti_service):
        """긴 문서 청킹 테스트"""
        # 청크 크기보다 긴 문서 생성
        content = "문장1. " * 100  # 약 500자
        title = "긴 문서"
        
        chunks_added = await processor.add_text_document(content, title, chunk_size=100)
        
        assert chunks_added > 1
        assert mock_graphiti_service.add_text_episode.call_count == chunks_added
        
        # 첫 번째 청크 확인
        first_call = mock_graphiti_service.add_text_episode.call_args_list[0]
        assert first_call[1]['name'] == f"{title}_chunk_1"
    
    async def test_add_json_data_dict(self, processor, mock_graphiti_service):
        """JSON 딕셔너리 데이터 추가 테스트"""
        data = {"name": "테스트", "value": 123}
        title = "JSON 테스트"
        
        chunks_added = await processor.add_json_data(data, title)
        
        assert chunks_added == 1
        mock_graphiti_service.add_json_episode.assert_called_once_with(
            name=title,
            data=data,
            source_description="json_data"
        )
    
    async def test_add_json_data_list(self, processor, mock_graphiti_service):
        """JSON 리스트 데이터 추가 테스트"""
        data = [
            {"name": "항목1", "value": 1},
            {"name": "항목2", "value": 2}
        ]
        title = "JSON 리스트"
        
        chunks_added = await processor.add_json_data(data, title)
        
        assert chunks_added == 2
        assert mock_graphiti_service.add_json_episode.call_count == 2
        
        # 각 호출 확인
        calls = mock_graphiti_service.add_json_episode.call_args_list
        assert calls[0][1]['name'] == f"{title}_item_1"
        assert calls[0][1]['data'] == data[0]
        assert calls[1][1]['name'] == f"{title}_item_2"
        assert calls[1][1]['data'] == data[1]
    
    async def test_add_json_data_auto_title(self, processor, mock_graphiti_service):
        """JSON 데이터 자동 제목 생성 테스트"""
        data = {"test": "data"}
        
        with patch('rag_chatbot.document_processor.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            chunks_added = await processor.add_json_data(data)
            
            assert chunks_added == 1
            mock_graphiti_service.add_json_episode.assert_called_once_with(
                name="json_data_20240101_120000",
                data=data,
                source_description="json_data"
            )


class TestDocumentProcessorFiles:
    """파일 처리 관련 테스트"""
    
    @pytest.fixture
    def processor(self, mock_graphiti_service, mock_settings):
        """DocumentProcessor fixture"""
        return DocumentProcessor(mock_graphiti_service, mock_settings)
    
    async def test_add_file_document_txt(self, processor, temp_dir):
        """텍스트 파일 추가 테스트"""
        # 임시 텍스트 파일 생성
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("테스트 파일 내용", encoding='utf-8')
        
        with patch.object(processor, 'add_text_document', return_value=1) as mock_add_text:
            chunks_added = await processor.add_file_document(txt_file)
            
            assert chunks_added == 1
            mock_add_text.assert_called_once_with(
                content="테스트 파일 내용",
                title="test",
                source_description="file_txt",
                chunk_size=1000
            )
    
    async def test_add_file_document_json(self, processor, temp_dir):
        """JSON 파일 추가 테스트"""
        # 임시 JSON 파일 생성
        json_file = temp_dir / "test.json"
        test_data = {"name": "테스트", "value": 123}
        json_file.write_text(json.dumps(test_data, ensure_ascii=False), encoding='utf-8')
        
        with patch.object(processor, 'add_json_data', return_value=1) as mock_add_json:
            chunks_added = await processor.add_file_document(json_file)
            
            assert chunks_added == 1
            mock_add_json.assert_called_once_with(
                data=test_data,
                title="test",
                source_description="file_json"
            )
    
    async def test_add_file_document_not_found(self, processor):
        """존재하지 않는 파일 처리 테스트"""
        with pytest.raises(FileNotFoundError, match="File not found"):
            await processor.add_file_document("/nonexistent/file.txt")
    
    async def test_add_file_document_unsupported(self, processor, temp_dir):
        """지원하지 않는 파일 형식 테스트"""
        # 지원하지 않는 확장자 파일 생성
        unsupported_file = temp_dir / "test.pdf"
        unsupported_file.write_text("PDF content")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            await processor.add_file_document(unsupported_file)
    
    async def test_add_json_file_invalid(self, processor, temp_dir):
        """잘못된 JSON 파일 처리 테스트"""
        # 잘못된 JSON 파일 생성
        invalid_json_file = temp_dir / "invalid.json"
        invalid_json_file.write_text("{ invalid json", encoding='utf-8')
        
        with pytest.raises(ValueError, match="Invalid JSON file"):
            await processor.add_file_document(invalid_json_file)


class TestTextChunking:
    """텍스트 청킹 테스트"""
    
    @pytest.fixture
    def processor(self, mock_graphiti_service, mock_settings):
        """DocumentProcessor fixture"""
        return DocumentProcessor(mock_graphiti_service, mock_settings)
    
    def test_split_text_short(self, processor):
        """짧은 텍스트 분할 테스트"""
        text = "짧은 텍스트"
        chunks = processor._split_text_into_chunks(text, 100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_text_sentences(self, processor):
        """문장 단위 분할 테스트"""
        text = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
        chunks = processor._split_text_into_chunks(text, 30)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50  # 여유분 고려
    
    def test_split_text_long_sentence(self, processor):
        """긴 문장 강제 분할 테스트"""
        # 청크 크기보다 긴 단일 문장
        long_sentence = "이것은매우긴문장으로청크크기를초과합니다" * 10
        chunks = processor._split_text_into_chunks(long_sentence, 50)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50
    
    def test_split_text_empty(self, processor):
        """빈 텍스트 분할 테스트"""
        chunks = processor._split_text_into_chunks("", 100)
        assert len(chunks) == 1
        assert chunks[0] == ""
    
    def test_split_text_preserve_content(self, processor):
        """내용 보존 테스트"""
        text = "문장1. 문장2. 문장3. 문장4."
        chunks = processor._split_text_into_chunks(text, 20)
        
        # 모든 청크를 합쳤을 때 원본과 동일해야 함 (공백 정리 후)
        combined = " ".join(chunks)
        # 공백 정규화 후 비교
        assert "".join(text.split()) in "".join(combined.split())


class TestBulkProcessing:
    """대량 처리 테스트"""
    
    @pytest.fixture
    def processor(self, mock_graphiti_service, mock_settings):
        """DocumentProcessor fixture"""
        return DocumentProcessor(mock_graphiti_service, mock_settings)
    
    async def test_bulk_process_directory(self, processor, temp_dir):
        """디렉토리 대량 처리 테스트"""
        # 테스트 파일들 생성
        (temp_dir / "file1.txt").write_text("내용1")
        (temp_dir / "file2.txt").write_text("내용2")
        (temp_dir / "data.json").write_text('{"key": "value"}')
        (temp_dir / "ignore.pdf").write_text("무시될 파일")
        
        with patch.object(processor, 'add_file_document', return_value=1) as mock_add_file:
            results = await processor.bulk_process_directory(temp_dir)
            
            # PDF 파일은 제외하고 3개 파일 처리되어야 함
            assert len(results) == 3
            assert mock_add_file.call_count == 3
            
            # 모든 결과가 1이어야 함 (mock return value)
            assert all(count == 1 for count in results.values())
    
    async def test_bulk_process_directory_not_found(self, processor):
        """존재하지 않는 디렉토리 처리 테스트"""
        with pytest.raises(ValueError, match="Directory not found"):
            await processor.bulk_process_directory("/nonexistent/directory")
    
    async def test_bulk_process_directory_error_handling(self, processor, temp_dir):
        """파일 처리 오류 처리 테스트"""
        # 테스트 파일 생성
        (temp_dir / "good.txt").write_text("정상 파일")
        (temp_dir / "bad.txt").write_text("문제가 될 파일")
        
        # add_file_document가 특정 파일에서 에러를 발생시키도록 설정
        async def mock_add_file(file_path, source_description):
            if "bad.txt" in str(file_path):
                raise Exception("처리 실패")
            return 1
        
        with patch.object(processor, 'add_file_document', side_effect=mock_add_file):
            results = await processor.bulk_process_directory(temp_dir)
            
            # 2개 파일 모두 결과에 포함되어야 함
            assert len(results) == 2
            
            # 성공한 파일은 1, 실패한 파일은 0
            good_file = [k for k in results.keys() if "good.txt" in k][0]
            bad_file = [k for k in results.keys() if "bad.txt" in k][0]
            
            assert results[good_file] == 1
            assert results[bad_file] == 0
    
    async def test_bulk_process_custom_patterns(self, processor, temp_dir):
        """커스텀 파일 패턴 테스트"""
        # 다양한 확장자 파일 생성
        (temp_dir / "file.txt").write_text("텍스트")
        (temp_dir / "data.json").write_text('{}')
        (temp_dir / "config.yaml").write_text("yaml: content")
        
        with patch.object(processor, 'add_file_document', return_value=1) as mock_add_file:
            # txt 파일만 처리
            results = await processor.bulk_process_directory(
                temp_dir, 
                file_patterns=["*.txt"]
            )
            
            assert len(results) == 1
            assert mock_add_file.call_count == 1
            assert any("file.txt" in path for path in results.keys())