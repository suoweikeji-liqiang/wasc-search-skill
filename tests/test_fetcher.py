import asyncio

import pytest

from skill.fetcher import SearchAdapter, gather_results


class SuccessAdapter:
    async def search(self, query: str) -> list[dict[str, str]]:
        return [{"title": "ok", "url": "https://example.com", "snippet": query}]


class FailingAdapter:
    async def search(self, query: str) -> list[dict[str, str]]:
        raise TimeoutError("boom")


class UnexpectedFailureAdapter:
    async def search(self, query: str) -> list[dict[str, str]]:
        raise ValueError("unexpected")


def test_gather_results_keeps_successes_when_one_adapter_fails() -> None:
    adapters: list[SearchAdapter] = [SuccessAdapter(), FailingAdapter()]
    results = asyncio.run(gather_results(adapters, "test"))

    assert len(results) == 1
    assert results[0]["title"] == "ok"


def test_gather_results_ignores_non_timeout_errors() -> None:
    adapters: list[SearchAdapter] = [SuccessAdapter(), UnexpectedFailureAdapter()]
    results = asyncio.run(gather_results(adapters, "test"))

    assert len(results) == 1
    assert results[0]["title"] == "ok"


class HangingAdapter:
    async def search(self, query: str) -> list[dict[str, str]]:
        _ = query
        await asyncio.Event().wait()
        return []


def test_gather_results_times_out_hung_adapter_and_keeps_successes() -> None:
    adapters: list[SearchAdapter] = [SuccessAdapter(), HangingAdapter()]
    results = asyncio.run(gather_results(adapters, "test", per_adapter_timeout_seconds=0.01))

    assert len(results) == 1
    assert results[0]["title"] == "ok"

