"""
Sample data initialization for RAG chatbot.
RAG 채팅봇을 위한 샘플 데이터 초기화
"""

import json
from typing import Dict, List, Any
from pathlib import Path

def get_sample_documents() -> List[Dict[str, Any]]:
    """Get sample documents for initialization."""
    return [
        {
            "title": "AI와 머신러닝 기초",
            "content": """
인공지능(AI)은 인간의 지능을 모방하는 컴퓨터 시스템입니다. 
머신러닝은 AI의 한 분야로, 데이터를 통해 학습하고 예측하는 기술입니다.
주요 머신러닝 알고리즘으로는 선형 회귀, 결정 트리, 신경망 등이 있습니다.
RAG(Retrieval-Augmented Generation)는 검색과 생성을 결합한 AI 기술입니다.
            """,
            "source": "sample_doc"
        },
        {
            "title": "Python 프로그래밍 기초",
            "content": """
Python은 간단하고 배우기 쉬운 프로그래밍 언어입니다.
주요 특징으로는 가독성, 다양한 라이브러리, 크로스 플랫폼 지원이 있습니다.
데이터 분석에 자주 사용되는 라이브러리로는 pandas, numpy, matplotlib이 있습니다.
웹 개발에는 Django, Flask 등의 프레임워크를 사용할 수 있습니다.
            """,
            "source": "sample_doc"
        },
        {
            "title": "그래프 데이터베이스 개념",
            "content": """
그래프 데이터베이스는 노드와 엣지로 데이터를 저장하는 NoSQL 데이터베이스입니다.
관계형 데이터베이스와 달리 복잡한 관계를 효율적으로 처리할 수 있습니다.
FalkorDB는 Redis 기반의 그래프 데이터베이스로, 빠른 성능을 제공합니다.
지식 그래프 구축에 적합하며, 추천 시스템, 소셜 네트워크 분석 등에 활용됩니다.
            """,
            "source": "sample_doc"
        }
    ]

def get_sample_json_data() -> List[Dict[str, Any]]:
    """Get sample JSON data for initialization."""
    return [
        {
            "title": "회사 정보",
            "data": {
                "company": "Tech Solutions Inc",
                "founded": 2020,
                "industry": "Software Development",
                "location": "Seoul, South Korea",
                "employees": 150,
                "products": ["AI Platform", "Data Analytics", "Web Solutions"]
            }
        },
        {
            "title": "제품 카탈로그",
            "data": {
                "products": [
                    {
                        "id": "P001",
                        "name": "AI Assistant",
                        "category": "Artificial Intelligence",
                        "price": 99.99,
                        "features": ["NLP", "Machine Learning", "API Integration"]
                    },
                    {
                        "id": "P002", 
                        "name": "Data Analyzer",
                        "category": "Analytics",
                        "price": 149.99,
                        "features": ["Real-time Processing", "Visualization", "Reporting"]
                    }
                ]
            }
        },
        {
            "title": "사용자 가이드",
            "data": {
                "guide": {
                    "getting_started": "시작하기 위해 먼저 환경을 설정하세요",
                    "basic_commands": ["init", "add-doc", "chat", "search", "status"],
                    "advanced_features": {
                        "personalization": "사용자 ID로 개인화된 검색 제공",
                        "web_interface": "브라우저를 통한 채팅 인터페이스",
                        "api_integration": "RESTful API를 통한 외부 시스템 연동"
                    }
                }
            }
        }
    ]

def save_sample_data_files(data_dir: Path) -> None:
    """Save sample data to files."""
    data_dir.mkdir(exist_ok=True)
    
    # 샘플 문서 저장
    for i, doc in enumerate(get_sample_documents(), 1):
        file_path = data_dir / f"sample_doc_{i}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"제목: {doc['title']}\n\n")
            f.write(doc['content'])
    
    # 샘플 JSON 데이터 저장
    for i, item in enumerate(get_sample_json_data(), 1):
        file_path = data_dir / f"sample_data_{i}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(item['data'], f, ensure_ascii=False, indent=2)
    
    print(f"✅ Sample data files saved to {data_dir}")

def get_quick_start_commands() -> List[str]:
    """Get list of commands for quick start demo."""
    return [
        'rag-chatbot init',
        'rag-chatbot add-doc --text "안녕하세요. 이것은 첫 번째 테스트 문서입니다." --title "환영 메시지"',
        'rag-chatbot add-json --data \'{"greeting": "Hello", "language": "Korean", "purpose": "testing"}\' --title "기본 설정"',
        'rag-chatbot search "테스트"',
        'rag-chatbot chat --query "안녕하세요"',
        'rag-chatbot status'
    ]

def print_quick_start_guide() -> None:
    """Print quick start guide."""
    print("\n🚀 RAG Chatbot Quick Start Guide")
    print("=" * 40)
    print("\n1️⃣ 기본 설정:")
    print("   rag-chatbot init")
    print("   rag-chatbot setup --create-config")
    
    print("\n2️⃣ 문서 추가:")
    print("   rag-chatbot add-doc --file document.txt")
    print("   rag-chatbot add-json --file data.json")
    
    print("\n3️⃣ 검색 및 채팅:")
    print("   rag-chatbot search \"your question\"")
    print("   rag-chatbot chat")
    
    print("\n4️⃣ 웹 인터페이스:")
    print("   rag-chatbot serve")
    print("   브라우저에서 http://localhost:8000 접속")
    
    print("\n5️⃣ 상태 확인:")
    print("   rag-chatbot status")
    
    print("\n📖 자세한 도움말: rag-chatbot --help")
    print("🌐 웹 문서: https://github.com/your-repo/rag-graphiti")