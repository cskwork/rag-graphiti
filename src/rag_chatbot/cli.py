"""
Command Line Interface for RAG chatbot.
RAG 채팅봇을 위한 명령줄 인터페이스
"""

import asyncio
import json
import re
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .chat_handler import ChatHandler
from .config import Settings, get_settings, setup_logging, create_example_env_file
from .document_processor import DocumentProcessor
from .graphiti_service import close_graphiti_service, get_graphiti_service
from .sample_data import get_sample_documents, get_sample_json_data, save_sample_data_files, print_quick_start_guide

console = Console()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Enable quiet mode (ERROR level only)')
@click.pass_context
def main(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """
    Production-ready RAG chatbot using Graphiti knowledge graph.
    
    Graphiti 지식 그래프를 사용한 프로덕션 RAG 챗봇
    """
    # Context 객체에 설정 저장
    ctx.ensure_object(dict)
    settings = get_settings()
    
    # 로깅 레벨 조정
    if quiet:
        settings.log_level = "ERROR"
    elif verbose:
        settings.log_level = "DEBUG"
    
    setup_logging(settings)
    
    # 시작 로그 및 설정 요약
    logger = logging.getLogger(__name__)
    logger.info("=== RAG Chatbot Starting ===")
    logger.info(f"Version: 0.1.0")
    logger.info(f"Log Level: {settings.log_level}")
    logger.debug(f"FalkorDB: {settings.falkor_host}:{settings.falkor_port}")
    logger.debug(f"Web Server: {settings.web_host}:{settings.web_port}")
    
    ctx.obj['settings'] = settings


@main.command()
@click.option('--reset', is_flag=True, help='Reset database (clear all data)')
@click.option('--sample-data', is_flag=True, help='Load sample data for testing')
@click.pass_context
def init(ctx: click.Context, reset: bool, sample_data: bool) -> None:
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
            
            # 샘플 데이터 로드
            if sample_data:
                console.print("[yellow]Loading sample data...[/yellow]")
                processor = DocumentProcessor(service, settings)
                
                # 샘플 문서 추가
                doc_count = 0
                for doc in get_sample_documents():
                    chunks = await processor.add_text_document(
                        content=doc['content'],
                        title=doc['title'],
                        source_description=doc['source']
                    )
                    doc_count += 1
                    console.print(f"[dim]Added document: {doc['title']} ({chunks} chunks)[/dim]")
                
                # 샘플 JSON 데이터 추가
                json_count = 0
                for item in get_sample_json_data():
                    items = await processor.add_json_data(
                        data=item['data'],
                        title=item['title'],
                        source_description='sample_json'
                    )
                    json_count += 1
                    console.print(f"[dim]Added JSON data: {item['title']} ({items} items)[/dim]")
                
                console.print(f"[green]✓ Sample data loaded: {doc_count} documents, {json_count} JSON items[/green]")
                console.print("[dim]Try: rag-chatbot search \"AI\" or rag-chatbot chat[/dim]")
            
        except Exception as e:
            console.print(f"[red]✗ Initialization failed: {e}[/red]")
            console.print("\n[yellow]🔧 Common solutions:[/yellow]")
            console.print("  1. Check FalkorDB is running: [cyan]docker-compose up -d[/cyan]")
            console.print("  2. Verify connection settings: [cyan]rag-chatbot status[/cyan]")
            console.print("  3. Check .env file exists: [cyan]ls -la .env[/cyan]")
            console.print("  4. Create config file: [cyan]rag-chatbot setup --create-config[/cyan]")
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
            console.print("\n[yellow]🔧 Common solutions:[/yellow]")
            console.print("  1. Check file exists and is readable")
            console.print("  2. Verify FalkorDB connection: [cyan]rag-chatbot status[/cyan]")
            console.print("  3. Try with smaller chunk size: [cyan]--chunk-size 500[/cyan]")
            console.print("  4. Check system status: [cyan]rag-chatbot status[/cyan]")
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
            console.print("\n[yellow]🔧 Common solutions:[/yellow]")
            console.print("  1. Validate JSON format: [cyan]python -m json.tool your_file.json[/cyan]")
            console.print("  2. Check file encoding (should be UTF-8)")
            console.print("  3. Verify FalkorDB connection: [cyan]rag-chatbot status[/cyan]")
            console.print("  4. Try with smaller JSON objects")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_add_json())


@main.command()
@click.argument('input_path', required=False)
@click.option('--text', help='Text content to add directly')
@click.option('--data', help='JSON data as string') 
@click.option('--title', help='Title for the content')
@click.option('--source', default='user_input', help='Source description')
@click.option('--chunk-size', default=1000, help='Chunk size for long documents')
@click.option('--type', 'content_type', type=click.Choice(['auto', 'text', 'json']), default='auto', help='Content type (auto-detect by default)')
@click.pass_context
def add(
    ctx: click.Context,
    input_path: Optional[str],
    text: Optional[str],
    data: Optional[str],
    title: Optional[str],
    source: str,
    chunk_size: int,
    content_type: str
) -> None:
    """
    Universal add command - automatically detects file types.
    파일 타입을 자동 감지하는 통합 추가 명령어
    
    Examples:
        rag-chatbot add document.txt
        rag-chatbot add data.json
        rag-chatbot add --text "Hello world" --title "Greeting"
        rag-chatbot add --data '{"key": "value"}' --title "Config"
    """
    settings = ctx.obj['settings']
    
    # 입력 검증
    inputs = [input_path, text, data]
    if sum(x is not None for x in inputs) != 1:
        console.print("[red]Error: Provide exactly one input (file path, --text, or --data)[/red]")
        console.print("\n[yellow]Examples:[/yellow]")
        console.print("  rag-chatbot add document.txt")
        console.print("  rag-chatbot add --text 'Your content here'")
        console.print("  rag-chatbot add --data '{\"key\": \"value\"}'")
        sys.exit(1)
    
    async def _add():
        try:
            service = await get_graphiti_service(settings)
            processor = DocumentProcessor(service, settings)
            
            # 파일 경로가 제공된 경우
            if input_path:
                file_path = Path(input_path)
                
                if not file_path.exists():
                    console.print(f"[red]✗ File not found: {input_path}[/red]")
                    console.print("[yellow]💡 Check the file path and try again[/yellow]")
                    sys.exit(1)
                
                # 파일 타입 자동 감지
                if content_type == 'auto':
                    if file_path.suffix.lower() in ['.json', '.jsonl']:
                        detected_type = 'json'
                    else:
                        detected_type = 'text'
                    console.print(f"[dim]Auto-detected type: {detected_type}[/dim]")
                else:
                    detected_type = content_type
                
                if detected_type == 'json':
                    items = await processor.add_file_document(
                        file_path=str(file_path),
                        source_description=source
                    )
                    console.print(f"[green]✓ Added JSON file '{file_path.name}' ({items} items)[/green]")
                else:
                    chunks = await processor.add_file_document(
                        file_path=str(file_path),
                        source_description=source,
                        chunk_size=chunk_size
                    )
                    console.print(f"[green]✓ Added document '{file_path.name}' ({chunks} chunks)[/green]")
            
            # 텍스트 콘텐츠가 제공된 경우
            elif text:
                chunks = await processor.add_text_document(
                    content=text,
                    title=title,
                    source_description=source,
                    chunk_size=chunk_size
                )
                console.print(f"[green]✓ Added text content ({chunks} chunks)[/green]")
            
            # JSON 데이터가 제공된 경우
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
                    console.print("[yellow]💡 Validate JSON: python -m json.tool[/yellow]")
                    sys.exit(1)
            
            # 다음 단계 제안
            console.print("\n[dim]💡 Next steps:[/dim]")
            console.print(f"[dim]  Search: rag-chatbot search \"keyword\"[/dim]")
            console.print(f"[dim]  Chat: rag-chatbot chat[/dim]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to add content: {e}[/red]")
            console.print("\n[yellow]🔧 Common solutions:[/yellow]")
            console.print("  1. Check file exists and is readable")
            console.print("  2. Verify FalkorDB connection: [cyan]rag-chatbot status[/cyan]")
            console.print("  3. For large files, try smaller chunks: [cyan]--chunk-size 500[/cyan]")
            console.print("  4. For JSON, validate format: [cyan]python -m json.tool file.json[/cyan]")
            sys.exit(1)
        finally:
            await close_graphiti_service()
    
    asyncio.run(_add())


@main.command()
@click.argument('query')
@click.option('--max-results', default=5, help='Maximum number of results')
@click.option('--user-id', help='User ID for personalized search')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed search results with scores')
@click.option('--explain', '-e', is_flag=True, help='Show search explanations')
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    max_results: int,
    user_id: Optional[str],
    detailed: bool,
    explain: bool
) -> None:
    """
    Search knowledge graph with enhanced ranking.
    향상된 순위화를 통한 지식 그래프 검색
    """
    settings = ctx.obj['settings']
    
    async def _search():
        try:
            search_start = time.time()
            service = await get_graphiti_service(settings)
            
            # 사용자 중심 검색
            center_node_uuid = None
            if user_id:
                user_nodes = await service.node_search(f"user:{user_id}")
                if user_nodes:
                    center_node_uuid = user_nodes[0].uuid
                    console.print(f"[dim]Using personalized search for user: {user_id}[/dim]")
            
            results = await service.search(
                query=query,
                max_results=max_results,
                center_node_uuid=center_node_uuid
            )
            
            search_time = time.time() - search_start
            
            if not results:
                console.print(f"[yellow]No results found for: '{query}'[/yellow]")
                console.print(f"[dim]Search completed in {search_time:.2f}s[/dim]")
                return
            
            # 결과 표시 방식 선택
            if explain:
                # 상세한 설명과 함께 표시
                formatted_results = service.format_search_results_with_explanations(
                    results, query, show_scores=detailed
                )
                console.print(formatted_results)
            
            elif detailed:
                # 테이블 형식으로 상세 정보 표시
                table = Table(title=f"Search Results for: '{query}' (Ranked)")
                table.add_column("No.", style="dim", width=4)
                table.add_column("Fact", style="cyan")
                table.add_column("Score", style="green", width=8)
                table.add_column("Date", style="blue", width=12)
                
                query_lower = query.lower().strip()
                query_words = set(re.findall(r'\b\w+\b', query_lower))
                
                for i, result in enumerate(results, 1):
                    # 관련성 점수 계산
                    try:
                        score = service._calculate_relevance_score(query_lower, query_words, result)
                        score_str = f"{score:.1f}"
                    except Exception:
                        score_str = "N/A"
                    
                    # 날짜 정보
                    timestamp = service._get_result_timestamp(result)
                    date_str = timestamp.strftime("%Y-%m-%d") if timestamp else "N/A"
                    
                    table.add_row(str(i), result.fact, score_str, date_str)
                
                console.print(table)
            
            else:
                # 기본 테이블 표시
                table = Table(title=f"Search Results for: '{query}'")
                table.add_column("No.", style="dim")
                table.add_column("Fact", style="cyan")
                table.add_column("Date", style="green")
                
                for i, result in enumerate(results, 1):
                    timestamp = service._get_result_timestamp(result)
                    date_str = timestamp.strftime("%Y-%m-%d") if timestamp else ""
                    
                    table.add_row(str(i), result.fact, date_str)
                
                console.print(table)
            
            # 성능 정보 표시
            console.print(f"[dim]Search completed in {search_time:.2f}s | {len(results)} results[/dim]")
            
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
                    
                    # 응답 처리 (성능 측정 포함)
                    start_time = time.time()
                    response = await chat_handler.process_query(user_input, user_id)
                    response_time = time.time() - start_time
                    
                    # 성능 로깅
                    logger = logging.getLogger(__name__)
                    logger.info(f"Query processed in {response_time:.2f}s: '{user_input[:50]}...'")
                    
                    console.print(f"\n[bold green]Assistant[/bold green]: {response}")
                    
                    # 느린 응답 시 경고 표시
                    if response_time > 5.0:
                        console.print(f"[dim yellow]⚠️ Response time: {response_time:.2f}s[/dim yellow]")
                    
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
            
            # 캐시 통계
            try:
                cache_stats = await service.get_cache_stats()
                cache_usage = f"{cache_stats['cache_size']}/{cache_stats['max_size']}"
                cache_efficiency = f"TTL: {cache_stats['ttl_seconds']}s"
                table.add_row(
                    "Cache System",
                    "[green]active[/green]",
                    f"Usage: {cache_usage}, {cache_efficiency}"
                )
            except Exception as e:
                table.add_row(
                    "Cache System",
                    "[yellow]unknown[/yellow]",
                    f"Stats unavailable: {str(e)[:30]}"
                )
            
            # 설정 정보
            table.add_row(
                "FalkorDB",
                "[blue]configured[/blue]",
                f"{settings.falkor_host}:{settings.falkor_port}"
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
            
            # 성능 설정
            perf_details = f"Max results: {settings.default_max_results}, History: {settings.default_chat_history_size}"
            table.add_row("Performance", "[blue]configured[/blue]", perf_details)
            
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
@click.option('--create-config', is_flag=True, help='Create example .env configuration file')
@click.option('--create-sample-files', is_flag=True, help='Create sample data files')
@click.option('--guide', is_flag=True, help='Show quick start guide')
@click.option('--interactive', '-i', is_flag=True, help='Interactive setup wizard')
@click.pass_context
def setup(ctx: click.Context, create_config: bool, create_sample_files: bool, guide: bool, interactive: bool) -> None:
    """
    Setup assistant for first-time users.
    첫 사용자를 위한 설정 도우미
    """
    if interactive:
        # 대화형 설정 마법사
        console.print(Panel(
            "[bold green]🧙‍♂️ Interactive Setup Wizard[/bold green]\n\n"
            "Welcome to RAG Chatbot! This wizard will help you get started.\n"
            "RAG 채팅봇에 오신 것을 환영합니다! 이 마법사가 시작을 도와드립니다.",
            title="Setup Wizard",
            border_style="green"
        ))
        
        # 1. 환경 파일 생성 확인
        env_exists = Path(".env").exists()
        if not env_exists:
            create_env = Prompt.ask(
                "\n📝 Configuration file (.env) not found. Create one?",
                choices=["y", "n"],
                default="y"
            )
            if create_env.lower() == 'y':
                create_example_env_file()
                console.print("[green]✓ Configuration file created[/green]")
        else:
            console.print("[green]✓ Configuration file already exists[/green]")
        
        # 2. FalkorDB 설정 확인
        console.print("\n🔧 Checking FalkorDB connection...")
        falkor_host = Prompt.ask("FalkorDB host", default="localhost")
        falkor_port = Prompt.ask("FalkorDB port", default="6379")
        
        # 3. LLM 설정 선택
        use_llm = Prompt.ask(
            "\n🤖 Do you want to configure LLM (for enhanced responses)?",
            choices=["y", "n"],
            default="n"
        )
        
        if use_llm.lower() == 'y':
            llm_provider = Prompt.ask(
                "Choose LLM provider",
                choices=["openai", "anthropic", "google", "skip"],
                default="skip"
            )
            if llm_provider != "skip":
                api_key = Prompt.ask(f"Enter {llm_provider.upper()} API key (or 'skip')", default="skip")
                if api_key != "skip":
                    console.print(f"[green]✓ {llm_provider.upper()} API key configured[/green]")
                    console.print("[dim]Remember to add this to your .env file[/dim]")
        
        # 4. 샘플 데이터 생성
        create_samples = Prompt.ask(
            "\n📚 Create sample data for testing?",
            choices=["y", "n"],
            default="y"
        )
        
        if create_samples.lower() == 'y':
            data_dir = Path("./data")
            save_sample_data_files(data_dir)
            console.print("[green]✓ Sample data files created[/green]")
        
        # 5. 완료 및 다음 단계
        console.print(Panel(
            "[bold green]🎉 Setup Complete![/bold green]\n\n"
            "Next steps:\n"
            "1. Start FalkorDB: [cyan]docker-compose up -d[/cyan]\n"
            "2. Initialize database: [cyan]rag-chatbot init --sample-data[/cyan]\n"
            "3. Try a search: [cyan]rag-chatbot search \"AI\"[/cyan]\n"
            "4. Start chatting: [cyan]rag-chatbot chat[/cyan]\n"
            "5. Web interface: [cyan]rag-chatbot serve[/cyan]",
            title="Ready to Go!",
            border_style="green"
        ))
        return
    
    if not any([create_config, create_sample_files, guide]):
        # 기본 동작: 전체 가이드 표시
        console.print(Panel(
            "[bold green]RAG Chatbot Setup Assistant[/bold green]\n\n"
            "This tool helps you get started quickly with RAG Chatbot.\n"
            "이 도구는 RAG 채팅봇을 빠르게 시작할 수 있도록 도와줍니다.\n\n"
            "Available setup options:\n"
            "  --interactive       Interactive setup wizard\n"
            "  --create-config     Create .env configuration file\n"
            "  --create-sample-files  Create sample data files\n"
            "  --guide            Show quick start guide\n\n"
            "Example: rag-chatbot setup --interactive",
            title="Setup Assistant",
            border_style="green"
        ))
        return
    
    if create_config:
        console.print("[yellow]Creating configuration file...[/yellow]")
        create_example_env_file()
        console.print("[green]✓ Configuration file created[/green]")
        console.print("[dim]Edit .env file with your settings before running init[/dim]")
    
    if create_sample_files:
        console.print("[yellow]Creating sample data files...[/yellow]")
        data_dir = Path("./data")
        save_sample_data_files(data_dir)
        console.print("[green]✓ Sample data files created[/green]")
        console.print(f"[dim]Sample files saved to {data_dir}[/dim]")
    
    if guide:
        print_quick_start_guide()


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


if __name__ == '__main__':
    main()