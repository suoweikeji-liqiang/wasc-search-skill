import asyncio
from collections.abc import Iterable
from threading import Event
from typing import Any, get_type_hints
from unittest.mock import patch

from skill.fetcher import SearchAdapter
from skill.main import EmptyAdapter, RunQueryResult, run_query


class FakeAdapter:
    async def search(self, query: str) -> list[dict[str, Any]]:
        return [
            {
                "title": "政策标题",
                "url": "https://gov.example/policy",
                "snippet": "数据出境安全评估办法发布",
            }
        ]


def test_fake_adapter_search_annotation_matches_search_adapter_protocol() -> None:
    fake_return_type = get_type_hints(FakeAdapter.search)["return"]
    protocol_return_type = get_type_hints(SearchAdapter.search)["return"]

    assert fake_return_type == protocol_return_type


def test_run_query_return_type_annotation_matches_structured_contract() -> None:
    type_hints = get_type_hints(run_query)

    assert type_hints["return"] is RunQueryResult


def test_run_query_returns_structured_result_contract() -> None:
    result = run_query("数据出境安全评估办法", adapters=[FakeAdapter()])

    assert set(result.keys()) == {
        "summary",
        "key_points",
        "sources",
        "time_or_version",
        "uncertainties",
    }
    assert isinstance(result["summary"], str)
    assert isinstance(result["key_points"], list)
    assert isinstance(result["sources"], list)
    assert isinstance(result["time_or_version"], str)
    assert isinstance(result["uncertainties"], list)
    assert result["summary"]
    assert result["sources"][0]["title"] == "政策标题"


def test_run_query_preserves_explicit_empty_adapters() -> None:
    captured: dict[str, list[SearchAdapter]] = {}

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = query, per_adapter_timeout_seconds
        captured["adapters"] = list(adapters)
        return []

    with patch("skill.main.gather_results", new=fake_gather_results):
        run_query("q", adapters=[])

    assert captured["adapters"] == []


def test_run_query_uses_empty_adapter_when_adapters_is_none(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    captured: dict[str, list[SearchAdapter]] = {}

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = query, per_adapter_timeout_seconds
        captured["adapters"] = list(adapters)
        return []

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.gather_results", new=fake_gather_results),
    ):
        run_query("q", adapters=None)

    assert len(captured["adapters"]) == 1
    assert isinstance(captured["adapters"][0], EmptyAdapter)


def test_run_query_uses_tavily_adapters_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    captured: dict[str, list[SearchAdapter]] = {}

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = query, per_adapter_timeout_seconds
        captured["adapters"] = list(adapters)
        return []

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.gather_results", new=fake_gather_results),
    ):
        run_query("latest arxiv rag chunking", adapters=None)

    assert len(captured["adapters"]) == 2
    assert all(not isinstance(adapter, EmptyAdapter) for adapter in captured["adapters"])


def test_run_query_can_be_called_from_async_context() -> None:
    async def invoke() -> RunQueryResult:
        return run_query("数据出境安全评估办法", adapters=[FakeAdapter()])

    result = asyncio.run(invoke())

    assert result["summary"] == "数据出境安全评估办法发布"


def test_run_query_ignores_malformed_items_from_fetcher() -> None:
    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        return [
            {"title": "missing url", "snippet": "bad"},
            {"title": "bad url type", "url": 123, "snippet": "bad"},
            {
                "title": "good",
                "url": "https://example.com/good",
                "snippet": "good snippet",
            },
        ]

    with patch("skill.main.gather_results", new=fake_gather_results):
        result = run_query("q", adapters=[FakeAdapter()])

    assert result["summary"] == "good snippet"
    assert result["sources"] == [{"title": "good", "url": "https://example.com/good"}]


def test_run_query_returns_fallback_when_all_items_malformed() -> None:
    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        return [{"title": "missing url"}, {"url": "https://example.com/no-title"}]

    with patch("skill.main.gather_results", new=fake_gather_results):
        result = run_query("q", adapters=[FakeAdapter()])

    assert result["summary"] == "来源不足"
    assert result["sources"] == []
    assert result["uncertainties"] == ["来源不足"]


def test_run_query_times_out_when_gather_never_returns() -> None:
    started = Event()

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        started.set()
        await asyncio.Event().wait()
        return []

    with patch("skill.main.gather_results", new=fake_gather_results):
        result = run_query("q", adapters=[FakeAdapter()], orchestration_timeout_seconds=0.01)

    assert started.is_set()
    assert result["summary"] == "来源不足"
    assert result["sources"] == []


def test_run_query_uses_minimax_on_default_path_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_KEY", "minimax-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        return [{"title": "A", "url": "https://a.example", "snippet": "snippet a"}]

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.gather_results", new=fake_gather_results),
        patch(
            "skill.main.generate_with_minimax",
            return_value={
                "summary": "from model",
                "key_points": ["from model"],
                "sources": [{"title": "A", "url": "https://a.example"}],
                "time_or_version": "unknown",
                "uncertainties": [],
            },
        ) as minimax_mock,
    ):
        result = run_query("q", adapters=None)

    assert minimax_mock.called
    assert result["summary"] == "from model"
    assert minimax_mock.call_args.kwargs["timeout_seconds"] == 2.0


def test_run_query_uses_minimax_timeout_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_KEY", "minimax-key")
    monkeypatch.setenv("MINIMAX_TIMEOUT_SECONDS", "2.5")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        return [{"title": "A", "url": "https://a.example", "snippet": "snippet a"}]

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.gather_results", new=fake_gather_results),
        patch(
            "skill.main.generate_with_minimax",
            return_value={
                "summary": "from model",
                "key_points": ["from model"],
                "sources": [{"title": "A", "url": "https://a.example"}],
                "time_or_version": "unknown",
                "uncertainties": [],
            },
        ) as minimax_mock,
    ):
        run_query("q", adapters=None)

    assert minimax_mock.called
    assert minimax_mock.call_args.kwargs["timeout_seconds"] == 2.5


def test_run_query_skips_minimax_for_academic_queries(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_KEY", "minimax-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        return [
            {
                "title": "A",
                "url": "https://arxiv.org/abs/2601.15457",
                "snippet": "Chunking benchmark paper abstract.",
            }
        ]

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.classify_query", return_value="academic"),
        patch("skill.main.gather_results", new=fake_gather_results),
        patch("skill.main.generate_with_minimax") as minimax_mock,
    ):
        result = run_query("RAG chunking 最新论文综述", adapters=None)

    assert minimax_mock.called is False
    assert "Chunking benchmark paper abstract" in result["summary"]


def test_run_query_policy_change_query_falls_back_when_minimax_lacks_change_signal(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINIMAX_KEY", "minimax-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        return [
            {
                "title": "policy update",
                "url": "https://www.gov.cn/policy/change",
                "snippet": (
                    "\u4e0e\u65e7\u7248\u76f8\u6bd4\uff0c2025\u5e74\u65b0\u589e\u4f01\u4e1a\u81ea\u8bc4\u4f30"
                    "\u8981\u6c42\u5e76\u8c03\u6574\u7533\u62a5\u6d41\u7a0b\u3002"
                ),
            }
        ]

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.classify_query", return_value="policy"),
        patch("skill.main.gather_results", new=fake_gather_results),
        patch(
            "skill.main.generate_with_minimax",
            return_value={
                "summary": "\u8be5\u529e\u6cd5\u660e\u786e\u4e86\u9002\u7528\u8303\u56f4\u3002",
                "key_points": ["\u7531\u76d1\u7ba1\u90e8\u95e8\u53d1\u5e03\u3002"],
                "sources": [{"title": "policy update", "url": "https://www.gov.cn/policy/change"}],
                "time_or_version": "unknown",
                "uncertainties": [],
            },
        ) as minimax_mock,
    ):
        result = run_query(
            "2025\u5e74\u6570\u636e\u51fa\u5883\u5b89\u5168\u8bc4\u4f30\u529e\u6cd5\u6709\u54ea\u4e9b\u53d8\u5316\uff1f",
            adapters=None,
        )

    assert minimax_mock.called
    assert result["summary"] != "\u8be5\u529e\u6cd5\u660e\u786e\u4e86\u9002\u7528\u8303\u56f4\u3002"
    output_text = " ".join([result["summary"], *result["key_points"]])
    assert any(term in output_text for term in ("\u53d8\u5316", "\u65b0\u589e", "\u8c03\u6574"))


def test_run_query_policy_change_query_keeps_minimax_when_change_signal_present(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINIMAX_KEY", "minimax-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        return [
            {
                "title": "policy update",
                "url": "https://www.gov.cn/policy/change",
                "snippet": "\u653f\u7b56\u89e3\u8bfb\u3002",
            }
        ]

    minimax_summary = (
        "\u6570\u636e\u51fa\u5883\u5b89\u5168\u8bc4\u4f30\u529e\u6cd5\u7684\u4e3b\u8981\u53d8\u5316"
        "\u662f\u65b0\u589e\u4f01\u4e1a\u81ea\u8bc4\u4f30\u8981\u6c42\u5e76\u8c03\u6574\u7533\u62a5\u6d41\u7a0b\u3002"
    )

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.classify_query", return_value="policy"),
        patch("skill.main.gather_results", new=fake_gather_results),
        patch(
            "skill.main.generate_with_minimax",
            return_value={
                "summary": minimax_summary,
                "key_points": [
                    "\u529e\u6cd5\u53d8\u5316\u5305\u542b\u65b0\u589e\u8981\u6c42\u548c\u6d41\u7a0b\u8c03\u6574\u3002"
                ],
                "sources": [{"title": "policy update", "url": "https://www.gov.cn/policy/change"}],
                "time_or_version": "unknown",
                "uncertainties": [],
            },
        ) as minimax_mock,
    ):
        result = run_query(
            "2025\u5e74\u6570\u636e\u51fa\u5883\u5b89\u5168\u8bc4\u4f30\u529e\u6cd5\u6709\u54ea\u4e9b\u53d8\u5316\uff1f",
            adapters=None,
        )

    assert minimax_mock.called
    assert result["summary"] == minimax_summary


def test_run_query_policy_prefers_gov_sources_when_available() -> None:
    class MixedPolicyAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "Project 2025",
                    "url": "https://en.wikipedia.org/wiki/Project_2025",
                    "snippet": "Not a Chinese policy source.",
                },
                {
                    "title": "数据出境安全评估办法",
                    "url": "https://www.gov.cn/zhengce/2022-07/07/content_5686197.htm",
                    "snippet": "国家网信部门发布数据出境安全评估办法。",
                },
            ]

    result = run_query("2025年数据出境安全评估办法有哪些变化？", adapters=[MixedPolicyAdapter()])

    assert result["sources"]
    assert all("gov.cn" in source["url"] for source in result["sources"])
    assert "gov.cn" in result["sources"][0]["url"]


def test_run_query_academic_prefers_arxiv_like_sources() -> None:
    class MixedAcademicAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "RAG chunking tutorial",
                    "url": "https://www.youtube.com/watch?v=abc",
                    "snippet": "Video tutorial.",
                },
                {
                    "title": "A Systematic Analysis of Chunking Strategies",
                    "url": "https://arxiv.org/abs/2601.14123",
                    "snippet": "Systematic analysis for RAG chunking reliability.",
                },
            ]

    result = run_query("RAG chunking 最新论文综述", adapters=[MixedAcademicAdapter()])

    assert result["sources"]
    assert "arxiv.org" in result["sources"][0]["url"]


def test_run_query_truncates_local_summary_for_long_snippet() -> None:
    long_snippet = "A" * 800

    class LongSnippetAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "Long source",
                    "url": "https://example.com/long",
                    "snippet": long_snippet,
                }
            ]

    result = run_query("some query", adapters=[LongSnippetAdapter()])

    assert len(result["summary"]) <= 183


def test_run_query_extracts_focus_sentence_for_policy() -> None:
    class PolicySnippetAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "政策问答",
                    "url": "https://www.gov.cn/x",
                    "snippet": "这是背景介绍。根据数据出境安全评估办法，重点变化包括范围和流程调整。",
                }
            ]

    result = run_query("2025年数据出境安全评估办法有哪些变化？", adapters=[PolicySnippetAdapter()])

    assert "办法" in result["summary"]
    assert result["summary"].startswith("根据数据出境安全评估办法")


def test_run_query_strips_markdown_heading_noise() -> None:
    class MarkdownAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "RAG review",
                    "url": "https://arxiv.org/abs/2603.25333",
                    "snippet": "## 标题\n[链接文本](https://example.com) RAG chunking 最新研究总结。",
                }
            ]

    result = run_query("RAG chunking 最新论文综述", adapters=[MarkdownAdapter()])

    assert not result["summary"].startswith("#")
    assert "链接文本" in result["summary"]


def test_run_query_strips_metadata_prefix_noise() -> None:
    class MetadataAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "policy",
                    "url": "https://www.gov.cn/policy",
                    "snippet": (
                        "发布：2025-10-11 来源：某站 作者：A 浏览量：99 "
                        "根据数据出境安全评估办法，企业需要按规定评估并申报。"
                    ),
                }
            ]

    result = run_query("2025年数据出境安全评估办法有哪些变化？", adapters=[MetadataAdapter()])

    assert not result["summary"].startswith("发布")
    assert "根据数据出境安全评估办法" in result["summary"]


def test_run_query_policy_change_query_prefers_change_sentences() -> None:
    class ChangeAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "政策变化解读",
                    "url": "https://www.gov.cn/policy/change",
                    "snippet": "该办法用于规范数据出境。与旧版相比，2025年新增企业自评估要求并调整申报流程。",
                }
            ]

    result = run_query("2025年数据出境安全评估办法有哪些变化？", adapters=[ChangeAdapter()])

    assert "新增" in result["summary"] or "调整" in result["summary"] or "变化" in result["summary"]
