from collections.abc import Iterable
from typing import Any
from unittest.mock import patch

from skill.fetcher import SearchAdapter
from skill.main import run_query


def test_policy_change_query_keeps_minimax_when_local_lacks_change_signal(monkeypatch) -> None:
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
                "title": "policy note",
                "url": "https://www.gov.cn/policy/note",
                "snippet": "数据出境安全评估办法明确了监管框架和适用范围。",
            }
        ]

    minimax_summary = "该办法明确了适用范围和监管要求。"
    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.classify_query", return_value="policy"),
        patch("skill.main.gather_results", new=fake_gather_results),
        patch(
            "skill.main.generate_with_minimax",
            return_value={
                "summary": minimax_summary,
                "key_points": ["由主管部门发布。"],
                "sources": [{"title": "policy note", "url": "https://www.gov.cn/policy/note"}],
                "time_or_version": "unknown",
                "uncertainties": [],
            },
        ),
    ):
        result = run_query("2025年数据出境安全评估办法有哪些变化？", adapters=None)

    assert result["summary"] == minimax_summary
