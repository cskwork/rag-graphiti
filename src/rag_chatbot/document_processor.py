"""
Document processing and ingestion for RAG chatbot.
RAG 채팅봇을 위한 문서 처리 및 수집
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import Settings
from .graphiti_service import GraphitiService

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process and ingest documents into knowledge graph.
    문서를 지식 그래프에 처리하고 수집
    """
    
    def __init__(self, graphiti_service: GraphitiService, settings: Settings):
        self.graphiti_service = graphiti_service
        self.settings = settings
        
    async def add_text_document(
        self,
        content: str,
        title: Optional[str] = None,
        source_description: str = "document",
        chunk_size: int = 1000
    ) -> int:
        """
        Add text document to knowledge graph.
        텍스트 문서를 지식 그래프에 추가
        """
        if not content.strip():
            raise ValueError("Document content cannot be empty")
        
        # 제목이 없으면 자동 생성
        if not title:
            title = f"document_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # 긴 문서는 청크로 분할
        chunks = self._split_text_into_chunks(content, chunk_size)
        
        logger.info(f"Processing document '{title}' into {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            chunk_title = f"{title}_chunk_{i+1}" if len(chunks) > 1 else title
            
            await self.graphiti_service.add_text_episode(
                name=chunk_title,
                content=chunk,
                source_description=source_description
            )
        
        logger.info(f"Successfully added document '{title}' with {len(chunks)} chunks")
        return len(chunks)
    
    async def add_file_document(
        self,
        file_path: Union[str, Path],
        source_description: Optional[str] = None,
        chunk_size: int = 1000
    ) -> int:
        """
        Add file document to knowledge graph.
        파일 문서를 지식 그래프에 추가
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if source_description is None:
            source_description = f"file_{file_path.suffix[1:]}"
        
        # 파일 확장자에 따른 처리
        if file_path.suffix.lower() == '.txt':
            content = file_path.read_text(encoding='utf-8')
            return await self.add_text_document(
                content=content,
                title=file_path.stem,
                source_description=source_description,
                chunk_size=chunk_size
            )
        
        elif file_path.suffix.lower() == '.json':
            return await self._add_json_file(file_path, source_description)
        
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    async def add_json_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        title: Optional[str] = None,
        source_description: str = "json_data"
    ) -> int:
        """
        Add JSON data to knowledge graph.
        JSON 데이터를 지식 그래프에 추가
        """
        if not title:
            title = f"json_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # 리스트인 경우 각 항목을 별도 에피소드로 처리
        if isinstance(data, list):
            for i, item in enumerate(data):
                item_title = f"{title}_item_{i+1}"
                await self.graphiti_service.add_json_episode(
                    name=item_title,
                    data=item,
                    source_description=source_description
                )
            
            logger.info(f"Added {len(data)} JSON items from '{title}'")
            return len(data)
        
        else:
            # 단일 딕셔너리인 경우
            await self.graphiti_service.add_json_episode(
                name=title,
                data=data,
                source_description=source_description
            )
            
            logger.info(f"Added JSON data '{title}'")
            return 1
    
    async def _add_json_file(
        self,
        file_path: Path,
        source_description: str
    ) -> int:
        """Add JSON file to knowledge graph."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return await self.add_json_data(
                data=data,
                title=file_path.stem,
                source_description=source_description
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file {file_path}: {e}")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks for processing.
        텍스트를 처리용 청크로 분할
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            # 문장이 너무 긴 경우 강제로 분할
            if len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 긴 문장을 청크 크기로 분할
                for i in range(0, len(sentence), chunk_size):
                    chunks.append(sentence[i:i + chunk_size])
                continue
            
            # 현재 청크에 문장을 추가했을 때 크기 확인
            potential_chunk = current_chunk + sentence + ". "
            
            if len(potential_chunk) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                current_chunk = potential_chunk
        
        # 마지막 청크 추가
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def bulk_process_directory(
        self,
        directory_path: Union[str, Path],
        file_patterns: List[str] = ["*.txt", "*.json"],
        source_description: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Process all matching files in a directory.
        디렉토리의 모든 일치하는 파일 처리
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        if source_description is None:
            source_description = f"bulk_import_{directory_path.name}"
        
        results = {}
        
        for pattern in file_patterns:
            for file_path in directory_path.glob(pattern):
                try:
                    chunks_added = await self.add_file_document(
                        file_path=file_path,
                        source_description=source_description
                    )
                    results[str(file_path)] = chunks_added
                    
                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
                    results[str(file_path)] = 0
        
        total_chunks = sum(results.values())
        logger.info(f"Bulk processing completed: {len(results)} files, {total_chunks} chunks")
        
        return results