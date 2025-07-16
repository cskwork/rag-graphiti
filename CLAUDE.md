# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG chatbot with Graphiti knowledge graph backend using FalkorDB. CLI-first architecture with optional FastAPI web interface. All code comments are in Korean.

## Essential Commands

### Development Setup
```bash
# Create and activate virtual environment (REQUIRED)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e .[dev]

# Install with specific LLM providers
pip install -e .[anthropic,openai,google]
```

### Core Operations
```bash
# Initialize/reset database
rag-chatbot init [--reset]

# Check system health
rag-chatbot status

# Add documents
rag-chatbot add-doc --text "content" --title "title"
rag-chatbot add-json --data '{"key": "value"}' --title "title"

# Interactive chat
rag-chatbot chat [--user-id user] [--query "question"]

# Search knowledge graph
rag-chatbot search "query" [--max-results N]

# Start web server
rag-chatbot serve [--host HOST] [--port PORT]
```

### Testing and Quality
```bash
# Run tests
pytest

# Format code (line-length: 100)
black src/ tests/

# Lint code (Python 3.9 target)
ruff check src/ tests/

# Type checking
mypy src/
```

### Docker Development
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f rag-chatbot
```

## Architecture

**Core Components:**
- `cli.py` - Click-based CLI interface with all commands
- `graphiti_service.py` - Graphiti wrapper handling knowledge graph operations
- `chat_handler.py` - RAG processing and conversation management
- `document_processor.py` - Document ingestion into episodes
- `config.py` - Pydantic settings with environment variable support
- `web_server.py` - FastAPI server for HTTP API

**Data Flow:**
1. Documents/JSON → Episodes → Graphiti Knowledge Graph
2. User Query → Hybrid Search (semantic + keyword) → Context Retrieval
3. Context + Query → Optional LLM → Response
4. Query/Response → New Episode for learning

**Key Features:**
- Personalized search with user-specific center node ranking
- Hybrid search combining semantic and keyword matching
- Conversation memory through episode storage
- Multi-format document support (text, JSON)

## Configuration

Environment variables (copy from `.env.example`):
- **Required:** `FALKORDB_HOST`, `FALKORDB_PORT`
- **Optional:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- **Settings:** `LOG_LEVEL`, `WEB_HOST`, `WEB_PORT`

Configuration loaded via Pydantic Settings from environment or `.env` file.

## Development Notes

- **Python 3.9+** minimum requirement
- **Korean comments** throughout codebase for explanations
- **CLI-first design** - all functionality accessible via command line
- **Production ready** with Docker deployment and health checks
- **Non-root Docker user** for security
- **Virtual environment required** for local development
- **pytest-asyncio** for async test support

## Dependencies

Core: graphiti-core>=0.17.4, click, pydantic, pydantic-settings, rich
Web: fastapi, uvicorn
Optional LLMs: openai, anthropic, google-generativeai
Dev: pytest, pytest-asyncio, black, ruff, mypy