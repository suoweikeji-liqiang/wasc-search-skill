import asyncio
import json

import httpx

from skill.sources.tavily import TavilyAdapter


def test_tavily_adapter_maps_results_to_internal_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["api_key"] == "key"
        assert payload["query"] == "rag"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Doc A", "url": "https://a.example", "content": "A snippet"},
                    {"title": "", "url": "https://bad.example", "content": "bad"},
                ]
            },
        )

    adapter = TavilyAdapter(api_key="key", transport=httpx.MockTransport(handler))
    results = asyncio.run(adapter.search("rag"))

    assert results == [
        {
            "title": "Doc A",
            "url": "https://a.example",
            "snippet": "A snippet",
        }
    ]


def test_tavily_adapter_returns_empty_list_on_http_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    adapter = TavilyAdapter(api_key="key", transport=httpx.MockTransport(handler))

    assert asyncio.run(adapter.search("rag")) == []
