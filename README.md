# RAG Chatbot with Graphiti Knowledge Graph

Production-ready RAG (Retrieval-Augmented Generation) chatbot using Graphiti 0.17.4 knowledge graph and FalkorDB.

## Features

🚀 **CLI-First Design**
- Full command-line interface for all operations
- Interactive chat mode
- Document processing and ingestion
- System health monitoring

🧠 **Knowledge Graph Integration**
- Graphiti 0.17.4 with FalkorDB backend
- Hybrid search (semantic + keyword)
- Personalized search with center node ranking
- Support for text and JSON episodes

🌐 **Optional Web Interface**
- Simple, responsive web chat interface
- Real-time conversation display
- System status dashboard

🔧 **Production-Ready**
- Docker Compose deployment
- Comprehensive configuration management
- Health checks and monitoring
- Structured logging

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd rag-graphiti

# Copy environment configuration
cp .env.example .env

# Edit .env with your configuration
# At minimum, set FalkorDB connection details
```

### 2. Using Docker (Recommended)

```bash
# Start FalkorDB and RAG Chatbot
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f rag-chatbot
```

### 3. Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or with specific providers
pip install -e .[anthropic,openai,google]

# Initialize database
rag-chatbot init

# Check system status
rag-chatbot status
```

## CLI Usage

### Document Management

```bash
# Add text document
rag-chatbot add-doc --text "Your document content" --title "Document Title"

# Add file
rag-chatbot add-doc --file document.txt --source "user_upload"

# Add JSON data
rag-chatbot add-json --data '{"key": "value"}' --title "Data Title"

# Add JSON file
rag-chatbot add-json --file data.json
```

### Chat Interface

```bash
# Interactive chat
rag-chatbot chat

# Single query
rag-chatbot chat --query "What is machine learning?"

# Personalized chat with user ID
rag-chatbot chat --user-id "john_doe"
```

### Search

```bash
# Search knowledge graph
rag-chatbot search "artificial intelligence"

# Limit results
rag-chatbot search "python programming" --max-results 3

# Personalized search
rag-chatbot search "my preferences" --user-id "john_doe"
```

### Web Interface

```bash
# Start web server
rag-chatbot serve

# Custom host and port
rag-chatbot serve --host 127.0.0.1 --port 9000

# Development mode with auto-reload
rag-chatbot serve --reload
```

### System Monitoring

```bash
# Check system health
rag-chatbot status

# Reset database (⚠️ WARNING: Deletes all data)
rag-chatbot init --reset
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FALKORDB_HOST` | localhost | FalkorDB host |
| `FALKORDB_PORT` | 6379 | FalkorDB port |
| `FALKORDB_USERNAME` | None | FalkorDB username (optional) |
| `FALKORDB_PASSWORD` | None | FalkorDB password (optional) |
| `OPENAI_API_KEY` | None | OpenAI API key (optional) |
| `ANTHROPIC_API_KEY` | None | Anthropic API key (optional) |
| `GOOGLE_API_KEY` | None | Google API key (optional) |
| `LOG_LEVEL` | INFO | Logging level |
| `DEFAULT_MAX_RESULTS` | 5 | Default search result limit |
| `WEB_HOST` | 0.0.0.0 | Web server host |
| `WEB_PORT` | 8000 | Web server port |

### LLM Provider Integration

The chatbot supports multiple LLM providers (optional):

**OpenAI**
```bash
pip install -e .[openai]
export OPENAI_API_KEY="your_key_here"
```

**Anthropic Claude**
```bash
pip install -e .[anthropic]
export ANTHROPIC_API_KEY="your_key_here"
```

**Google Gemini**
```bash
pip install -e .[google]
export GOOGLE_API_KEY="your_key_here"
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Web Interface  │    │  Document API   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │     Chat Handler            │
                    │  (RAG Logic & History)      │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Graphiti Service          │
                    │ (Knowledge Graph Manager)   │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      FalkorDB               │
                    │   (Graph Database)          │
                    └─────────────────────────────┘
```

## Data Flow

1. **Document Ingestion**: Text/JSON → Episodes → Knowledge Graph
2. **Query Processing**: User Query → Hybrid Search → Context Retrieval
3. **Response Generation**: Context + Query → LLM (optional) → Response
4. **Conversation Storage**: User Query + Response → New Episode

## Production Deployment

### Docker Compose (Recommended)

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  falkordb:
    image: falkordb/falkordb:latest
    volumes:
      - ./data/falkordb:/data
    ports:
      - "6379:6379"
  
  rag-chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FALKORDB_HOST=falkordb
    env_file:
      - .env.prod
    depends_on:
      - falkordb
```

### Security Considerations

- Set strong passwords for FalkorDB in production
- Use environment variables for API keys
- Enable authentication for web interface
- Configure proper firewall rules
- Use HTTPS in production

## Development

### Project Structure

```
rag-graphiti/
├── src/rag_chatbot/
│   ├── __init__.py
│   ├── cli.py              # CLI commands
│   ├── config.py           # Configuration management
│   ├── graphiti_service.py # Graphiti wrapper
│   ├── chat_handler.py     # Chat logic
│   ├── document_processor.py # Document ingestion
│   └── web_server.py       # Web interface
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Testing

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Code formatting
black src/
ruff check src/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

**FalkorDB Connection Failed**
```bash
# Check if FalkorDB is running
docker-compose ps falkordb

# Check logs
docker-compose logs falkordb
```

**Import Errors**
```bash
# Reinstall in editable mode
pip install -e .

# Check Python path
echo $PYTHONPATH
```

**Web Interface Not Loading**
```bash
# Check if port is available
netstat -tulpn | grep 8000

# Start with different port
rag-chatbot serve --port 9000
```

## Technical Terms Glossary

- **RAG**: Retrieval-Augmented Generation - AI가 외부 지식을 검색하여 답변을 생성하는 방식
- **Knowledge Graph**: 엔티티와 관계를 그래프로 표현한 지식 데이터베이스
- **Episode**: Graphiti에서 처리하는 정보 단위 (텍스트 또는 JSON)
- **Hybrid Search**: 벡터 검색과 키워드 검색을 결합한 검색 방식
- **Center Node**: 검색 결과를 특정 노드 기준으로 재순위하는 기준점
- **FalkorDB**: Redis 기반의 그래프 데이터베이스

## License

Apache License 2.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request