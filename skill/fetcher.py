import asyncio
from collections.abc import Iterable
from typing import Any, Protocol


class SearchAdapter(Protocol):
    async def search(self, query: str) -> list[dict[str, Any]]:
        ...


async def _run_search(
    adapter: SearchAdapter,
    query: str,
    per_adapter_timeout_seconds: float,
) -> list[dict[str, Any]]:
    try:
        return await asyncio.wait_for(adapter.search(query), timeout=per_adapter_timeout_seconds)
    except (TimeoutError, Exception):
        return []


async def gather_results(
    adapters: Iterable[SearchAdapter],
    query: str,
    per_adapter_timeout_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[
            _run_search(adapter, query, per_adapter_timeout_seconds)
            for adapter in adapters
        ]
    )
    flattened: list[dict[str, Any]] = []
    for batch in batches:
        flattened.extend(batch)
    return flattened
