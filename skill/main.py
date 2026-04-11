import asyncio
from queue import Queue
from threading import Thread
from typing import Any, TypedDict, cast

from skill.fetcher import SearchAdapter, gather_results
from skill.generator import build_result

DEFAULT_PER_ADAPTER_TIMEOUT_SECONDS = 3.0
DEFAULT_ORCHESTRATION_TIMEOUT_SECONDS = 5.0


class RunQueryResult(TypedDict):
    summary: str
    key_points: list[str]
    sources: list[dict[str, str]]
    time_or_version: str
    uncertainties: list[str]


class EmptyAdapter:
    async def search(self, query: str) -> list[dict[str, Any]]:
        return []


def _run_gather_results(
    adapters: list[SearchAdapter],
    query: str,
    per_adapter_timeout_seconds: float,
    orchestration_timeout_seconds: float,
) -> list[dict[str, Any]]:
    async def _gather_with_timeout() -> list[dict[str, Any]]:
        return await asyncio.wait_for(
            gather_results(adapters, query, per_adapter_timeout_seconds),
            timeout=orchestration_timeout_seconds,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(_gather_with_timeout())
        except TimeoutError:
            return []

    outcome_queue: Queue[list[dict[str, Any]] | BaseException] = Queue()

    def _worker() -> None:
        try:
            outcome_queue.put(asyncio.run(_gather_with_timeout()))
        except BaseException as exc:
            outcome_queue.put(exc)

    worker = Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=orchestration_timeout_seconds)
    if worker.is_alive() or outcome_queue.empty():
        return []

    outcome = outcome_queue.get()
    if isinstance(outcome, BaseException):
        if isinstance(outcome, TimeoutError):
            return []
        raise outcome
    return outcome


def _is_valid_result_item(item: dict[str, Any]) -> bool:
    return (
        isinstance(item.get("title"), str)
        and isinstance(item.get("url"), str)
        and isinstance(item.get("snippet"), str)
    )


def _filter_valid_items(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    valid_items: list[dict[str, str]] = []
    for item in results:
        if _is_valid_result_item(item):
            valid_items.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                }
            )
    return valid_items


def run_query(
    query: str,
    adapters: list[SearchAdapter] | None = None,
    per_adapter_timeout_seconds: float = DEFAULT_PER_ADAPTER_TIMEOUT_SECONDS,
    orchestration_timeout_seconds: float = DEFAULT_ORCHESTRATION_TIMEOUT_SECONDS,
) -> RunQueryResult:
    chosen_adapters: list[SearchAdapter] = [EmptyAdapter()] if adapters is None else adapters
    raw_results: list[dict[str, Any]] = _run_gather_results(
        chosen_adapters,
        query,
        per_adapter_timeout_seconds,
        orchestration_timeout_seconds,
    )
    results = _filter_valid_items(raw_results)
    top_sources: list[dict[str, str]] = [
        {"title": item["title"], "url": item["url"]} for item in results[:3]
    ]
    summary: str = results[0]["snippet"] if results else "来源不足"
    return cast(
        RunQueryResult,
        build_result(
            summary=summary,
            key_points=[summary] if summary else [],
            sources=top_sources,
            time_or_version="unknown",
            uncertainties=[] if results else ["来源不足"],
        ),
    )

