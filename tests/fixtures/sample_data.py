"""
테스트용 샘플 데이터
Sample data for testing
"""
from datetime import datetime, timezone
from typing import Dict, List, Any


# 샘플 문서 데이터
SAMPLE_DOCUMENTS = [
    {
        "title": "인공지능 개요",
        "content": """
        인공지능(AI)은 컴퓨터 시스템이 인간의 지적 능력을 모방하는 기술입니다.
        머신러닝과 딥러닝은 AI의 핵심 기술로, 데이터로부터 패턴을 학습합니다.
        자연어 처리, 컴퓨터 비전, 로봇공학 등 다양한 분야에 응용됩니다.
        """,
        "metadata": {"category": "technology", "level": "beginner"}
    },
    {
        "title": "머신러닝 기초",
        "content": """
        머신러닝은 명시적으로 프로그래밍하지 않고도 컴퓨터가 학습할 수 있게 하는 방법입니다.
        지도학습, 비지도학습, 강화학습의 세 가지 주요 유형이 있습니다.
        데이터 전처리, 모델 선택, 평가가 중요한 단계입니다.
        """,
        "metadata": {"category": "technology", "level": "intermediate"}
    },
    {
        "title": "파이썬 프로그래밍",
        "content": """
        파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다.
        데이터 분석, 웹 개발, 인공지능 등 다양한 분야에서 사용됩니다.
        NumPy, Pandas, Scikit-learn 등 풍부한 라이브러리 생태계를 제공합니다.
        """,
        "metadata": {"category": "programming", "level": "beginner"}
    }
]


# 샘플 JSON 데이터
SAMPLE_JSON_DATA = [
    {
        "title": "사용자 프로필",
        "data": {
            "user_id": "user_001",
            "name": "김철수",
            "preferences": {
                "topics": ["AI", "Programming", "Data Science"],
                "difficulty": "intermediate",
                "language": "korean"
            },
            "history": [
                {"query": "머신러닝이란?", "timestamp": "2024-01-01T10:00:00Z"},
                {"query": "파이썬 문법", "timestamp": "2024-01-01T11:00:00Z"}
            ]
        }
    },
    {
        "title": "제품 카탈로그",
        "data": {
            "products": [
                {
                    "id": "P001",
                    "name": "스마트폰 XYZ",
                    "category": "electronics",
                    "price": 800000,
                    "features": ["5G", "카메라", "배터리"],
                    "rating": 4.5
                },
                {
                    "id": "P002", 
                    "name": "노트북 ABC",
                    "category": "computers",
                    "price": 1200000,
                    "features": ["Intel CPU", "16GB RAM", "SSD"],
                    "rating": 4.8
                }
            ]
        }
    },
    {
        "title": "학습 진도",
        "data": {
            "course_id": "AI_101",
            "student_id": "student_001",
            "progress": {
                "completed_lessons": [1, 2, 3],
                "current_lesson": 4,
                "total_lessons": 10,
                "completion_rate": 0.3
            },
            "quiz_scores": [85, 90, 78],
            "assignments": [
                {"name": "과제1", "score": 95, "submitted": True},
                {"name": "과제2", "score": null, "submitted": False}
            ]
        }
    }
]


# 샘플 검색 쿼리와 기대 결과
SAMPLE_QUERIES = [
    {
        "query": "인공지능이란 무엇인가요?",
        "expected_topics": ["AI", "machine learning", "technology"],
        "difficulty": "beginner"
    },
    {
        "query": "파이썬으로 데이터 분석하는 방법",
        "expected_topics": ["Python", "data analysis", "pandas"],
        "difficulty": "intermediate"
    },
    {
        "query": "머신러닝 알고리즘 종류",
        "expected_topics": ["machine learning", "algorithms", "supervised learning"],
        "difficulty": "intermediate"
    },
    {
        "query": "딥러닝과 머신러닝의 차이점",
        "expected_topics": ["deep learning", "machine learning", "comparison"],
        "difficulty": "advanced"
    }
]


# 샘플 사용자 대화 세션
SAMPLE_CONVERSATIONS = [
    {
        "user_id": "user_001",
        "session_id": "session_001",
        "messages": [
            {
                "role": "user",
                "content": "인공지능에 대해 알고 싶어요",
                "timestamp": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
            },
            {
                "role": "assistant", 
                "content": "인공지능은 컴퓨터가 인간의 지적 능력을 모방하는 기술입니다...",
                "timestamp": datetime(2024, 1, 1, 10, 0, 5, tzinfo=timezone.utc)
            },
            {
                "role": "user",
                "content": "머신러닝과 딥러닝의 차이점은?",
                "timestamp": datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc)
            }
        ]
    },
    {
        "user_id": "user_002",
        "session_id": "session_002", 
        "messages": [
            {
                "role": "user",
                "content": "파이썬 기초 문법을 배우고 싶습니다",
                "timestamp": datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
            },
            {
                "role": "assistant",
                "content": "파이썬은 간결하고 읽기 쉬운 프로그래밍 언어입니다...",
                "timestamp": datetime(2024, 1, 1, 11, 0, 5, tzinfo=timezone.utc)
            }
        ]
    }
]


# 샘플 에러 케이스
SAMPLE_ERROR_CASES = [
    {
        "name": "empty_content",
        "data": {"title": "빈 문서", "content": ""},
        "expected_error": "Document content cannot be empty"
    },
    {
        "name": "invalid_json",
        "data": '{ "invalid": json }',
        "expected_error": "Invalid JSON"
    },
    {
        "name": "missing_file",
        "data": {"file_path": "/nonexistent/file.txt"},
        "expected_error": "File not found"
    },
    {
        "name": "unsupported_format", 
        "data": {"file_path": "test.unsupported"},
        "expected_error": "Unsupported file type"
    }
]


# 샘플 설정 변형
SAMPLE_CONFIGURATIONS = [
    {
        "name": "development",
        "settings": {
            "falkor_host": "localhost",
            "falkor_port": "6379",
            "log_level": "DEBUG",
            "default_max_results": 10
        }
    },
    {
        "name": "production",
        "settings": {
            "falkor_host": "prod-db.example.com",
            "falkor_port": "6379", 
            "log_level": "INFO",
            "default_max_results": 5
        }
    },
    {
        "name": "testing",
        "settings": {
            "falkor_host": "test-db",
            "falkor_port": "6380",
            "log_level": "WARNING",
            "default_max_results": 3
        }
    }
]


# 테스트용 Mock 응답
MOCK_RESPONSES = {
    "health_check": {
        "healthy": {
            "service": "graphiti",
            "status": "healthy",
            "connection_ready": True,
            "timestamp": "2024-01-01T12:00:00Z"
        },
        "unhealthy": {
            "service": "graphiti", 
            "status": "unhealthy",
            "connection_ready": False,
            "error": "Connection failed",
            "timestamp": "2024-01-01T12:00:00Z"
        }
    },
    "search_results": [
        {
            "fact": "인공지능은 컴퓨터 시스템이 인간의 지적 능력을 모방하는 기술입니다",
            "valid_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "invalid_at": None,
            "confidence": 0.95
        },
        {
            "fact": "머신러닝은 AI의 핵심 기술 중 하나입니다",
            "valid_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "invalid_at": None,
            "confidence": 0.88
        }
    ],
    "llm_responses": {
        "ai_explanation": "인공지능(AI)은 기계가 인간의 인지 능력을 모방하여 학습, 추론, 인식 등을 수행하는 기술입니다.",
        "python_help": "파이썬은 간단하고 읽기 쉬운 문법을 가진 프로그래밍 언어로, 데이터 분석과 AI 개발에 널리 사용됩니다.",
        "error_response": "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."
    }
}


def get_sample_document(index: int = 0) -> Dict[str, Any]:
    """샘플 문서 반환"""
    return SAMPLE_DOCUMENTS[index % len(SAMPLE_DOCUMENTS)]


def get_sample_json_data(index: int = 0) -> Dict[str, Any]:
    """샘플 JSON 데이터 반환"""
    return SAMPLE_JSON_DATA[index % len(SAMPLE_JSON_DATA)]


def get_sample_query(index: int = 0) -> Dict[str, Any]:
    """샘플 쿼리 반환"""
    return SAMPLE_QUERIES[index % len(SAMPLE_QUERIES)]


def get_sample_conversation(user_id: str = "user_001") -> Dict[str, Any]:
    """특정 사용자의 샘플 대화 반환"""
    for conv in SAMPLE_CONVERSATIONS:
        if conv["user_id"] == user_id:
            return conv
    return SAMPLE_CONVERSATIONS[0]


def get_mock_search_results(count: int = 2) -> List[Any]:
    """Mock 검색 결과 반환"""
    from unittest.mock import MagicMock
    
    results = []
    for i in range(min(count, len(MOCK_RESPONSES["search_results"]))):
        mock_result = MagicMock()
        result_data = MOCK_RESPONSES["search_results"][i]
        
        for key, value in result_data.items():
            setattr(mock_result, key, value)
        
        results.append(mock_result)
    
    return results