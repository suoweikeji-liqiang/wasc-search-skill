from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skill.sources import __all__ as exported_symbols
from skill.sources.academic import build_academic_queries
from skill.sources.web import build_web_queries


def test_sources_public_exports_are_minimal_and_consistent() -> None:
    assert exported_symbols == ["build_web_queries", "build_academic_queries"]


def test_policy_queries_match_expected_values() -> None:
    queries = build_web_queries("数据出境办法", "policy")
    assert queries == ["site:gov.cn 数据出境办法"]


def test_policy_change_queries_add_change_focused_variant() -> None:
    query = "2025年数据出境安全评估办法有哪些变化？"
    queries = build_web_queries(query, "policy")
    assert queries == [
        f"site:gov.cn {query}",
        f"site:gov.cn {query} 修订 调整 变化",
    ]


def test_non_policy_queries_match_expected_values() -> None:
    queries = build_web_queries("数据出境办法", "industry")
    assert queries == ["数据出境办法"]


def test_academic_queries_match_expected_values() -> None:
    queries = build_academic_queries("RAG chunking 论文")
    assert queries == [
        "RAG chunking 论文",
        "RAG chunking 论文 arxiv",
    ]
