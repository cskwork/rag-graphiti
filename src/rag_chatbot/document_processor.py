"""
Document processing and ingestion for RAG chatbot.
RAG 채팅봇을 위한 문서 처리 및 수집
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import aiofiles
import httpx
import markdown
from bs4 import BeautifulSoup

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
        
        elif file_path.suffix.lower() == '.md':
            return await self._add_markdown_file(file_path, source_description, chunk_size)
        
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
    
    async def _add_markdown_file(
        self,
        file_path: Path,
        source_description: str,
        chunk_size: int = 1000
    ) -> int:
        """Add markdown file to knowledge graph."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = await f.read()
            
            # 마크다운을 HTML로 변환 후 텍스트 추출
            html_content = markdown.markdown(
                markdown_content,
                extensions=['codehilite', 'fenced_code', 'tables', 'toc']
            )
            
            # HTML에서 텍스트 추출
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text(separator='\n', strip=True)
            
            return await self.add_text_document(
                content=text_content,
                title=file_path.stem,
                source_description=source_description,
                chunk_size=chunk_size
            )
            
        except Exception as e:
            raise ValueError(f"Failed to process markdown file {file_path}: {e}")
    
    async def _add_json_file(
        self,
        file_path: Path,
        source_description: str
    ) -> int:
        """Add JSON file to knowledge graph."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
            
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
        file_patterns: List[str] = ["*.txt", "*.md", "*.json"],
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
    
    async def add_url_document(
        self,
        url: str,
        title: Optional[str] = None,
        source_description: str = "web_url",
        chunk_size: int = 1000,
        timeout: int = 30
    ) -> int:
        """
        Add document from URL to knowledge graph.
        URL에서 문서를 가져와 지식 그래프에 추가
        """
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(f"Fetching content from URL: {url}")
                response = await client.get(
                    url,
                    headers={
                        'User-Agent': 'RAG-Chatbot/1.0 (Document Processor)',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                    }
                )
                response.raise_for_status()
                
                # 컨텐츠 타입에 따른 처리
                content_type = response.headers.get('content-type', '').lower()
                
                if 'text/html' in content_type:
                    content = self._extract_text_from_html(response.text)
                elif 'text/plain' in content_type:
                    content = response.text
                elif 'application/json' in content_type:
                    # JSON 컨텐츠는 별도 처리
                    json_data = response.json()
                    return await self.add_json_data(
                        data=json_data,
                        title=title or self._extract_title_from_url(url),
                        source_description=f"{source_description}_json"
                    )
                else:
                    content = response.text
                
                if not title:
                    title = self._extract_title_from_url(url)
                
                return await self.add_text_document(
                    content=content,
                    title=title,
                    source_description=f"{source_description}_{url}",
                    chunk_size=chunk_size
                )
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise ValueError(f"Failed to fetch URL {url}: {e}")
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            raise ValueError(f"Failed to process URL {url}: {e}")
    
    async def process_urls_file(
        self,
        urls_file_path: Union[str, Path],
        source_description: str = "urls_file",
        chunk_size: int = 1000,
        timeout: int = 30
    ) -> Dict[str, int]:
        """
        Process URLs from a text file.
        텍스트 파일에서 URL들을 처리
        """
        urls_file_path = Path(urls_file_path)
        
        if not urls_file_path.exists():
            raise FileNotFoundError(f"URLs file not found: {urls_file_path}")
        
        results = {}
        
        try:
            async with aiofiles.open(urls_file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            urls = self._parse_urls_from_content(content)
            logger.info(f"Found {len(urls)} URLs to process from {urls_file_path}")
            
            for url in urls:
                try:
                    chunks_added = await self.add_url_document(
                        url=url,
                        source_description=source_description,
                        chunk_size=chunk_size,
                        timeout=timeout
                    )
                    results[url] = chunks_added
                    logger.info(f"Successfully processed URL: {url} ({chunks_added} chunks)")
                    
                except Exception as e:
                    logger.error(f"Failed to process URL {url}: {e}")
                    results[url] = 0
            
            total_chunks = sum(results.values())
            logger.info(f"URL processing completed: {len(results)} URLs, {total_chunks} chunks")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing URLs file {urls_file_path}: {e}")
            raise ValueError(f"Failed to process URLs file: {e}")
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate URL format.
        URL 형식 검증
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract text content from HTML.
        HTML에서 텍스트 컨텐츠 추출
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 마크업에서 제거할 요소들
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # 텍스트 추출
        text = soup.get_text(separator='\n', strip=True)
        
        # 비어있는 줄 정리
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _extract_title_from_url(self, url: str) -> str:
        """
        Extract title from URL.
        URL에서 제목 추출
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace('www.', '')
        path = parsed_url.path.strip('/')
        
        if path:
            # 마지막 경로 세그먼트를 제목으로 사용
            title_part = path.split('/')[-1]
            return f"{domain}_{title_part}"
        else:
            return f"{domain}_homepage"
    
    def _parse_urls_from_content(self, content: str) -> List[str]:
        """
        Parse URLs from text content, ignoring comments.
        텍스트 컨텐츠에서 URL 파싱 (주석 제외)
        """
        urls = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # 빈 줄이나 주석 제외
            if not line or line.startswith('#'):
                continue
            
            # URL 검증
            if self._is_valid_url(line):
                urls.append(line)
            else:
                logger.warning(f"Skipping invalid URL: {line}")
        
        return urls