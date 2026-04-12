import json

import httpx

from skill.generator import generate_with_minimax


def test_generate_with_minimax_parses_structured_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://api.minimax.io/v1/chat/completions")
        assert request.headers["Authorization"] == "Bearer minimax-key"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "MiniMax-M2.7"
        assert payload["messages"][1]["role"] == "user"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "S",
                                    "key_points": ["K1", "K2"],
                                    "sources": [{"title": "T", "url": "https://x.example"}],
                                    "time_or_version": "2026-04",
                                    "uncertainties": [],
                                }
                            )
                        }
                    }
                ]
            },
        )

    result = generate_with_minimax(
        query="q",
        context_items=[{"title": "t", "url": "u", "snippet": "s"}],
        api_key="minimax-key",
        base_url="https://api.minimax.io/v1",
        transport=httpx.MockTransport(handler),
    )

    assert result is not None
    assert result["summary"] == "S"
    assert result["sources"][0]["url"] == "https://x.example"


def test_generate_with_minimax_extracts_json_inside_code_fence() -> None:
    fenced_json = """```json
{"summary":"S","key_points":["K"],"sources":[{"title":"T","url":"https://x.example"}],"time_or_version":"unknown","uncertainties":[]}
```"""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": fenced_json}}]},
        )

    result = generate_with_minimax(
        query="q",
        context_items=[{"title": "t", "url": "u", "snippet": "s"}],
        api_key="minimax-key",
        base_url="https://api.minimax.io/v1",
        transport=httpx.MockTransport(handler),
    )

    assert result is not None
    assert result["key_points"] == ["K"]


def test_generate_with_minimax_falls_back_to_plain_text_when_response_invalid() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    result = generate_with_minimax(
        query="q",
        context_items=[{"title": "t", "url": "u", "snippet": "s"}],
        api_key="minimax-key",
        base_url="https://api.minimax.io/v1",
        transport=httpx.MockTransport(handler),
    )

    assert result is not None
    assert result["summary"] == "not-json"
    assert result["uncertainties"] == ["model output not strict json"]


def test_generate_with_minimax_accepts_string_source_items() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "{\"summary\":\"S\",\"key_points\":[\"K\"],"
                                "\"sources\":[\"Doc A (https://a.example)\"],"
                                "\"time_or_version\":\"unknown\",\"uncertainties\":[]}"
                            )
                        }
                    }
                ]
            },
        )

    result = generate_with_minimax(
        query="q",
        context_items=[{"title": "t", "url": "u", "snippet": "s"}],
        api_key="minimax-key",
        base_url="https://api.minimax.io/v1",
        transport=httpx.MockTransport(handler),
    )

    assert result is not None
    assert result["sources"] == [{"title": "Doc A", "url": "https://a.example"}]


def test_generate_with_minimax_accepts_string_uncertainties() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "{\"summary\":\"S\",\"key_points\":[\"K\"],"
                                "\"sources\":[{\"title\":\"T\",\"url\":\"https://a.example\"}],"
                                "\"time_or_version\":2026,\"uncertainties\":\"limited context\"}"
                            )
                        }
                    }
                ]
            },
        )

    result = generate_with_minimax(
        query="q",
        context_items=[{"title": "t", "url": "u", "snippet": "s"}],
        api_key="minimax-key",
        base_url="https://api.minimax.io/v1",
        transport=httpx.MockTransport(handler),
    )

    assert result is not None
    assert result["time_or_version"] == "2026"
    assert result["uncertainties"] == ["limited context"]


def test_generate_with_minimax_falls_back_when_key_points_invalid() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "{\"summary\":\"S\",\"key_points\":42,"
                                "\"sources\":[],\"time_or_version\":\"unknown\",\"uncertainties\":42}"
                            )
                        }
                    }
                ]
            },
        )

    result = generate_with_minimax(
        query="q",
        context_items=[{"title": "t", "url": "u", "snippet": "s"}],
        api_key="minimax-key",
        base_url="https://api.minimax.io/v1",
        transport=httpx.MockTransport(handler),
    )

    assert result is not None
    assert result["key_points"] == ["S"]
    assert result["uncertainties"] == []


def test_generate_with_minimax_returns_none_when_content_empty() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    result = generate_with_minimax(
        query="q",
        context_items=[{"title": "t", "url": "u", "snippet": "s"}],
        api_key="minimax-key",
        base_url="https://api.minimax.io/v1",
        transport=httpx.MockTransport(handler),
    )

    assert result is None
