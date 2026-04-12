from typing import Any

import httpx

TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"


def _normalize_result_item(item: dict[str, Any]) -> dict[str, str] | None:
    title = str(item.get("title", "")).strip()
    url = str(item.get("url", "")).strip()
    snippet = str(item.get("content", "")).strip()
    if not title or not url or not snippet:
        return None
    return {"title": title, "url": url, "snippet": snippet}


class TavilyAdapter:
    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout_seconds: float = 5.0,
        endpoint: str = TAVILY_SEARCH_ENDPOINT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.endpoint = endpoint
        self.transport = transport

    async def search(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": self.max_results,
        }

        client_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        if self.transport is not None:
            client_kwargs["transport"] = self.transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(self.endpoint, json=payload)
                response.raise_for_status()
        except httpx.HTTPError:
            return []

        body = response.json()
        raw_items = body.get("results", [])
        if not isinstance(raw_items, list):
            return []

        normalized_items: list[dict[str, str]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_result_item(item)
            if normalized is not None:
                normalized_items.append(normalized)
        return normalized_items
