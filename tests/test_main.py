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


def test_run_query_uses_empty_adapter_when_adapters_is_none() -> None:
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
        run_query("q", adapters=None)

    assert len(captured["adapters"]) == 1
    assert isinstance(captured["adapters"][0], EmptyAdapter)


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
