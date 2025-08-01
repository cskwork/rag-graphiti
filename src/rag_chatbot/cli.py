"""
Command Line Interface for RAG chatbot.
RAG 채팅봇을 위한 명령줄 인터페이스
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .chat_handler import ChatHandler
from .config import Settings, get_settings, setup_logging
from .document_processor import DocumentProcessor
from .graphiti_service import close_graphiti_service, get_graphiti_service

console = Console()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """
    Production-ready RAG chatbot using Graphiti knowledge graph.
    
    Graphiti 지식 그래프를 사용한 프로덕션 RAG 챗봇
    """
    # Context 객체에 설정 저장
    ctx.ensure_object(dict)
    settings = get_settings()
    
    if verbose:
        settings.log_level = "DEBUG"
    
    setup_logging(settings)
    ctx.obj['settings'] = settings


@main.command()
@click.option('--reset', is_flag=True, help='Reset database (clear all data)')
@click.pass_context
def init(ctx: click.Context, reset: bool) -> None:
    """
    Initialize Graphiti database and indices.
    Graphiti 데이터베이스와 인덱스 초기화
    """
    settings = ctx.obj['settings']
    
    async def _init():
        try:
            service = await get_graphiti_service(settings)
            
            if reset:
                console.print("[yellow]Warning: This will delete all existing data![/yellow]")
                confirm = Prompt.ask("Are you sure? (yes/no)", default="no")
                if confirm.lower() != 'yes':
                    console.print("[green]Operation cancelled.[/green]")
                    return
                
                # 데이터 초기화 로직은 여기에 추가
                console.print("[red]Database reset not implemented yet[/red]")
            
            console.print("[green]✓ Graphiti database initialized successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Initialization failed: {e}[/red]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_init())


@main.command('add-doc')
@click.option('--file', 'file_path', type=click.Path(exists=True), help='File to add')
@click.option('--text', help='Text content to add directly')
@click.option('--title', help='Document title')
@click.option('--source', default='user_input', help='Source description')
@click.option('--chunk-size', default=1000, help='Chunk size for long documents')
@click.pass_context
def add_document(
    ctx: click.Context,
    file_path: Optional[str],
    text: Optional[str],
    title: Optional[str],
    source: str,
    chunk_size: int
) -> None:
    """
    Add document to knowledge graph.
    지식 그래프에 문서 추가
    """
    settings = ctx.obj['settings']
    
    if not file_path and not text:
        console.print("[red]Error: Must provide either --file or --text[/red]")
        sys.exit(1)
    
    async def _add_doc():
        try:
            service = await get_graphiti_service(settings)
            processor = DocumentProcessor(service, settings)
            
            if file_path:
                chunks = await processor.add_file_document(
                    file_path=file_path,
                    source_description=source,
                    chunk_size=chunk_size
                )
                console.print(f"[green]✓ Added file '{file_path}' ({chunks} chunks)[/green]")
                
            elif text:
                chunks = await processor.add_text_document(
                    content=text,
                    title=title,
                    source_description=source,
                    chunk_size=chunk_size
                )
                console.print(f"[green]✓ Added text document ({chunks} chunks)[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to add document: {e}[/red]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_add_doc())


@main.command('add-json')
@click.option('--file', 'file_path', type=click.Path(exists=True), help='JSON file to add')
@click.option('--data', help='JSON data as string')
@click.option('--title', help='Data title')
@click.option('--source', default='json_data', help='Source description')
@click.pass_context
def add_json_data(
    ctx: click.Context,
    file_path: Optional[str],
    data: Optional[str],
    title: Optional[str],
    source: str
) -> None:
    """
    Add JSON data to knowledge graph.
    지식 그래프에 JSON 데이터 추가
    """
    settings = ctx.obj['settings']
    
    if not file_path and not data:
        console.print("[red]Error: Must provide either --file or --data[/red]")
        sys.exit(1)
    
    async def _add_json():
        try:
            service = await get_graphiti_service(settings)
            processor = DocumentProcessor(service, settings)
            
            if file_path:
                items = await processor.add_file_document(
                    file_path=file_path,
                    source_description=source
                )
                console.print(f"[green]✓ Added JSON file '{file_path}' ({items} items)[/green]")
                
            elif data:
                try:
                    json_data = json.loads(data)
                    items = await processor.add_json_data(
                        data=json_data,
                        title=title,
                        source_description=source
                    )
                    console.print(f"[green]✓ Added JSON data ({items} items)[/green]")
                except json.JSONDecodeError as e:
                    console.print(f"[red]✗ Invalid JSON data: {e}[/red]")
                    sys.exit(1)
            
        except Exception as e:
            console.print(f"[red]✗ Failed to add JSON data: {e}[/red]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_add_json())


@main.command()
@click.argument('query')
@click.option('--max-results', default=5, help='Maximum number of results')
@click.option('--user-id', help='User ID for personalized search')
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    max_results: int,
    user_id: Optional[str]
) -> None:
    """
    Search knowledge graph.
    지식 그래프 검색
    """
    settings = ctx.obj['settings']
    
    async def _search():
        try:
            service = await get_graphiti_service(settings)
            
            # 사용자 중심 검색
            center_node_uuid = None
            if user_id:
                user_nodes = await service.node_search(f"user:{user_id}")
                if user_nodes:
                    center_node_uuid = user_nodes[0].uuid
            
            results = await service.search(
                query=query,
                max_results=max_results,
                center_node_uuid=center_node_uuid
            )
            
            if not results:
                console.print(f"[yellow]No results found for: '{query}'[/yellow]")
                return
            
            # 결과 표시
            table = Table(title=f"Search Results for: '{query}'")
            table.add_column("No.", style="dim")
            table.add_column("Fact", style="cyan")
            table.add_column("Valid From", style="green")
            
            for i, result in enumerate(results, 1):
                valid_from = ""
                if hasattr(result, 'valid_at') and result.valid_at:
                    valid_from = str(result.valid_at)[:19]  # 날짜만 표시
                
                table.add_row(str(i), result.fact, valid_from)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]✗ Search failed: {e}[/red]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_search())


@main.command()
@click.option('--user-id', help='User ID for personalized chat')
@click.option('--query', help='Single query (non-interactive mode)')
@click.pass_context
def chat(
    ctx: click.Context,
    user_id: Optional[str],
    query: Optional[str]
) -> None:
    """
    Interactive chat interface.
    대화형 채팅 인터페이스
    """
    settings = ctx.obj['settings']
    
    async def _chat():
        try:
            service = await get_graphiti_service(settings)
            chat_handler = ChatHandler(service, settings)
            
            if query:
                # 단일 질의 모드
                response = await chat_handler.process_query(query, user_id)
                console.print(Panel(response, title="Response", border_style="blue"))
                return
            
            # 대화형 모드
            console.print(Panel(
                "RAG Chatbot - Interactive Mode\n"
                "Commands: 'exit', 'quit', 'clear', 'help'",
                title="Welcome",
                border_style="green"
            ))
            
            while True:
                try:
                    user_input = Prompt.ask("\n[bold blue]You[/bold blue]", default="").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['exit', 'quit']:
                        break
                    elif user_input.lower() == 'clear':
                        chat_handler.clear_history()
                        console.print("[green]Chat history cleared[/green]")
                        continue
                    elif user_input.lower() == 'help':
                        console.print(Panel(
                            "Available commands:\n"
                            "• exit/quit - Exit chat\n"
                            "• clear - Clear chat history\n"
                            "• help - Show this help\n"
                            "• Any other text - Ask a question",
                            title="Help",
                            border_style="yellow"
                        ))
                        continue
                    
                    # 응답 처리
                    response = await chat_handler.process_query(user_input, user_id)
                    console.print(f"\n[bold green]Assistant[/bold green]: {response}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
            
            console.print("\n[green]Goodbye![/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Chat failed: {e}[/red]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_chat())


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """
    Check system health status.
    시스템 상태 확인
    """
    settings = ctx.obj['settings']
    
    async def _status():
        try:
            service = await get_graphiti_service(settings)
            health_status = await service.get_health_status()
            
            # 상태 테이블 생성
            table = Table(title="System Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="dim")
            
            # Graphiti 상태
            status_color = "green" if health_status["status"] == "healthy" else "red"
            table.add_row(
                "Graphiti Service",
                f"[{status_color}]{health_status['status']}[/{status_color}]",
                f"Connection: {health_status['connection_ready']}"
            )
            
            # 설정 정보
            table.add_row(
                "FalkorDB",
                "[blue]configured[/blue]",
                f"{settings.falkordb_host}:{settings.falkordb_port}"
            )
            
            # LLM 설정 확인
            llm_status = []
            if settings.openai_api_key:
                llm_status.append("OpenAI")
            if settings.anthropic_api_key:
                llm_status.append("Anthropic")
            if settings.google_api_key:
                llm_status.append("Google")
            
            llm_text = ", ".join(llm_status) if llm_status else "None configured"
            table.add_row("LLM Providers", "[yellow]available[/yellow]", llm_text)
            
            console.print(table)
            
            # 오류가 있으면 표시
            if "error" in health_status:
                console.print(f"\n[red]Error: {health_status['error']}[/red]")
            
        except Exception as e:
            console.print(f"[red]✗ Status check failed: {e}[/red]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_status())


@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool) -> None:
    """
    Start web interface.
    웹 인터페이스 시작
    """
    try:
        import uvicorn
        from .web_server import create_app
        
        settings = ctx.obj['settings']
        app = create_app(settings)
        
        console.print(f"[green]Starting web server at http://{host}:{port}[/green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=settings.log_level.lower()
        )
        
    except ImportError:
        console.print("[red]Web server dependencies not installed[/red]")
        console.print("Install with: pip install 'rag-chatbot[web]'")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ Failed to start web server: {e}[/red]")
        sys.exit(1)


@main.command('add-url')
@click.argument('url')
@click.option('--title', help='Document title')
@click.option('--source', default='web_url', help='Source description')
@click.option('--chunk-size', default=1000, help='Chunk size for long documents')
@click.option('--timeout', default=30, help='Request timeout in seconds')
@click.pass_context
def add_url(
    ctx: click.Context,
    url: str,
    title: Optional[str],
    source: str,
    chunk_size: int,
    timeout: int
) -> None:
    """
    Add document from URL to knowledge graph.
    URL에서 문서를 가져와 지식 그래프에 추가
    """
    settings = ctx.obj['settings']
    
    async def _add_url():
        try:
            service = await get_graphiti_service(settings)
            processor = DocumentProcessor(service, settings)
            
            chunks = await processor.add_url_document(
                url=url,
                title=title,
                source_description=source,
                chunk_size=chunk_size,
                timeout=timeout
            )
            console.print(f"[green]✓ Added URL document '{url}' ({chunks} chunks)[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to add URL: {e}[/red]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_add_url())


@main.command('import-urls')
@click.argument('urls_file', type=click.Path(exists=True))
@click.option('--source', default='urls_file', help='Source description')
@click.option('--chunk-size', default=1000, help='Chunk size for long documents')
@click.option('--timeout', default=30, help='Request timeout in seconds')
@click.pass_context
def import_urls(
    ctx: click.Context,
    urls_file: str,
    source: str,
    chunk_size: int,
    timeout: int
) -> None:
    """
    Import documents from URLs listed in a file.
    파일에 나열된 URL들에서 문서를 가져와 추가
    """
    settings = ctx.obj['settings']
    
    async def _import_urls():
        try:
            service = await get_graphiti_service(settings)
            processor = DocumentProcessor(service, settings)
            
            console.print(f"[blue]Processing URLs from: {urls_file}[/blue]")
            
            results = await processor.process_urls_file(
                urls_file_path=urls_file,
                source_description=source,
                chunk_size=chunk_size,
                timeout=timeout
            )
            
            # 결과 표시
            table = Table(title=f"URL Import Results")
            table.add_column("URL", style="cyan")
            table.add_column("Chunks", style="green")
            table.add_column("Status", style="yellow")
            
            for url, chunks in results.items():
                status = "✓ Success" if chunks > 0 else "✗ Failed"
                table.add_row(url, str(chunks), status)
            
            console.print(table)
            
            total_chunks = sum(results.values())
            successful = sum(1 for c in results.values() if c > 0)
            console.print(f"[green]Processed {successful}/{len(results)} URLs successfully, {total_chunks} total chunks[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to import URLs: {e}[/red]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_import_urls())


@main.command('bulk-import')
@click.argument('directory', type=click.Path(exists=True))
@click.option('--patterns', default='*.txt,*.md,*.json', help='File patterns to process (comma-separated)')
@click.option('--recursive', is_flag=True, help='Process directories recursively')
@click.option('--source', default='bulk_import', help='Source description')
@click.option('--chunk-size', default=1000, help='Chunk size for long documents')
@click.pass_context
def bulk_import(
    ctx: click.Context,
    directory: str,
    patterns: str,
    recursive: bool,
    source: str,
    chunk_size: int
) -> None:
    """
    Bulk import documents from a directory.
    디렉토리에서 문서를 대량으로 가져오기
    """
    settings = ctx.obj['settings']
    
    async def _bulk_import():
        try:
            service = await get_graphiti_service(settings)
            processor = DocumentProcessor(service, settings)
            
            console.print(f"[blue]Processing directory: {directory}[/blue]")
            
            # 파일 패턴 파싱
            file_patterns = [p.strip() for p in patterns.split(',')]
            
            from pathlib import Path
            
            results = {}
            directory_path = Path(directory)
            
            # 재귀적 처리 지원
            if recursive:
                for pattern in file_patterns:
                    for file_path in directory_path.rglob(pattern):
                        if file_path.is_file():
                            try:
                                chunks = await processor.add_file_document(
                                    file_path=file_path,
                                    source_description=source,
                                    chunk_size=chunk_size
                                )
                                results[str(file_path)] = chunks
                            except Exception as e:
                                logger.error(f"Failed to process file {file_path}: {e}")
                                results[str(file_path)] = 0
            else:
                # 기존 bulk_process_directory 사용
                results = await processor.bulk_process_directory(
                    directory_path=directory_path,
                    file_patterns=file_patterns,
                    source_description=source
                )
            
            # 결과 표시
            table = Table(title=f"Bulk Import Results")
            table.add_column("File", style="cyan")
            table.add_column("Chunks", style="green")
            table.add_column("Status", style="yellow")
            
            for file_path, chunks in results.items():
                status = "✓ Success" if chunks > 0 else "✗ Failed"
                # 파일 이름만 표시 (전체 경로 대신)
                file_name = Path(file_path).name
                table.add_row(file_name, str(chunks), status)
            
            console.print(table)
            
            total_chunks = sum(results.values())
            successful = sum(1 for c in results.values() if c > 0)
            console.print(f"[green]Processed {successful}/{len(results)} files successfully, {total_chunks} total chunks[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to bulk import: {e}[/red]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_bulk_import())


if __name__ == '__main__':
    main()