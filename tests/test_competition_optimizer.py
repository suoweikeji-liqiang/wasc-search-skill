from collections.abc import Iterable
from typing import Any
from unittest.mock import patch

from skill.fetcher import SearchAdapter
from skill.main import clear_query_cache, run_query


def test_run_query_skips_minimax_when_local_evidence_is_strong(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_KEY", "minimax-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    clear_query_cache()

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        return [
            {
                "title": "Vision Pro demand outlook",
                "url": "https://www.counterpointresearch.com/vision-pro-demand",
                "snippet": "Vision Pro 销量预测显示 2026 年出货仍受价格与场景限制影响。",
            },
            {
                "title": "Vision Pro market tracker",
                "url": "https://www.idc.com/getdoc.jsp?containerId=vision-pro",
                "snippet": "市场预测认为 Vision Pro 销量将在 2026 年继续小规模增长。",
            },
            {
                "title": "Vision Pro forecast update",
                "url": "https://www.canalys.com/newsroom/vision-pro-forecast",
                "snippet": "Canalys forecast covers Vision Pro shipment and demand trend.",
            },
        ]

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.gather_results", new=fake_gather_results),
        patch("skill.main.generate_with_minimax") as minimax_mock,
    ):
        result = run_query("Vision Pro 当前销量预测如何？", adapters=None)

    assert minimax_mock.called is False
    assert len(result["sources"]) == 3


def test_run_query_rejects_weaker_minimax_output_with_generic_guardrail(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_KEY", "minimax-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    clear_query_cache()

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        return [
            {
                "title": "AI GPU market forecast 2026",
                "url": "https://www.trendforce.com/ai-gpu-2026",
                "snippet": "2026 年 AI 服务器 GPU 市场份额预测显示 NVIDIA 仍然领先。",
            }
        ]

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.gather_results", new=fake_gather_results),
        patch(
            "skill.main.generate_with_minimax",
            return_value={
                "summary": "The market may change.",
                "key_points": ["General commentary without the year or GPU market share."],
                "sources": [],
                "time_or_version": "unknown",
                "uncertainties": [],
            },
        ) as minimax_mock,
    ):
        result = run_query("2026年AI服务器GPU市场份额预测", adapters=None)

    assert minimax_mock.called
    assert result["summary"] != "The market may change."
    merged = " ".join([result["summary"], *result["key_points"]])
    assert "2026" in merged
    assert any(term in merged for term in ("GPU", "市场份额", "预测"))


def test_run_query_caches_high_quality_default_results(monkeypatch) -> None:
    monkeypatch.delenv("MINIMAX_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    clear_query_cache()
    call_counter = {"count": 0}

    async def fake_gather_results(
        adapters: Iterable[SearchAdapter],
        query: str,
        per_adapter_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        _ = adapters, query, per_adapter_timeout_seconds
        call_counter["count"] += 1
        return [
            {
                "title": "AI Act official note",
                "url": "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
                "snippet": "AI Act policy update covers obligations for general-purpose AI models.",
            },
            {
                "title": "Open model adoption forecast",
                "url": "https://www.idc.com/open-model-adoption",
                "snippet": "Industry adoption depends on compliance cost and deployment speed.",
            },
        ]

    with (
        patch("skill.main.load_dotenv_file", return_value=False),
        patch("skill.main.gather_results", new=fake_gather_results),
    ):
        first = run_query("AI Act 对开源模型和产业落地影响", adapters=None)
        second = run_query("AI Act 对开源模型和产业落地影响", adapters=None)

    assert call_counter["count"] == 1
    assert first == second


def test_run_query_mixed_prefers_cross_domain_evidence() -> None:
    clear_query_cache()

    class MixedEvidenceAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "AI Act obligations",
                    "url": "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
                    "snippet": "The AI Act sets obligations for general-purpose AI models.",
                },
                {
                    "title": "AI Act FAQ",
                    "url": "https://digital-strategy.ec.europa.eu/en/faqs/ai-act",
                    "snippet": "FAQ on compliance timeline under the AI Act.",
                },
                {
                    "title": "Portal summary",
                    "url": "https://medium.com/someone/ai-act-summary",
                    "snippet": "Generic summary with little industry detail.",
                },
                {
                    "title": "Open-source model adoption forecast",
                    "url": "https://www.idc.com/getdoc.jsp?containerId=open-models",
                    "snippet": "Industry deployment speed depends on compliance cost and productization.",
                },
            ]

    result = run_query("AI Act 对开源模型和产业落地影响", adapters=[MixedEvidenceAdapter()])
    source_urls = [source["url"] for source in result["sources"]]

    assert any("europa.eu" in url for url in source_urls)
    assert any("idc.com" in url for url in source_urls)
    assert all("medium.com" not in url for url in source_urls)


def test_run_query_academic_adds_research_paper_cue_when_missing() -> None:
    clear_query_cache()

    class AcademicAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "Routine: A Structural Planning Framework",
                    "url": "https://arxiv.org/abs/2507.14447",
                    "snippet": "This work introduces a structured planning framework for LLM agents.",
                }
            ]

    result = run_query("LLM agent planning 最新研究", adapters=[AcademicAdapter()])
    merged = " ".join([result["summary"], *result["key_points"]]).lower()

    assert "paper" in merged or "论文" in merged
    assert "research" in merged or "研究" in merged or "arxiv" in merged


def test_run_query_mixed_adds_supply_chain_hint_when_missing() -> None:
    clear_query_cache()

    class MixedAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "CBAM tariff impact",
                    "url": "https://policy.example/cbam",
                    "snippet": "The EU carbon tariff raises export costs for EV makers.",
                }
            ]

    result = run_query("欧盟碳关税政策对新能源汽车出口和供应链影响", adapters=[MixedAdapter()])
    merged = " ".join([result["summary"], *result["key_points"], *result["uncertainties"]])

    assert "供应链" in merged
    assert "政策" in merged


def test_run_query_policy_prefers_entity_matched_source_over_generic_policy_change() -> None:
    clear_query_cache()

    class PolicyEntityAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "多国贸易新规盘点",
                    "url": "https://policy.example/trade-rules",
                    "snippet": "2025年多国贸易政策出现重大变化，一批外贸新规开始实施。",
                },
                {
                    "title": "数据出境安全评估申报指南（第三版）",
                    "url": "https://www.cac.gov.cn/2025-06/27/c_1752652339765002.htm",
                    "snippet": "国家互联网信息办公室发布《数据出境安全评估申报指南（第三版）》，明确有效期延长条件与申报材料。",
                },
                {
                    "title": "数据出境安全评估办法",
                    "url": "https://www.gov.cn/zhengce/2022-07/07/content_5686197.htm",
                    "snippet": "《数据出境安全评估办法》规定了数据处理者申请安全评估的适用情形和流程。",
                },
            ]

    result = run_query("2025年数据出境安全评估办法有哪些变化？", adapters=[PolicyEntityAdapter()])

    assert result["sources"]
    assert "数据出境安全评估" in result["sources"][0]["title"]
    assert "trade-rules" not in result["sources"][0]["url"]


def test_run_query_extracts_time_or_version_from_local_policy_evidence() -> None:
    clear_query_cache()

    class TimedPolicyAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "个人信息出境认证办法",
                    "url": "https://www.cac.gov.cn/2025-10/17/c_1762449728720008.htm",
                    "snippet": "《个人信息出境认证办法》已经审议通过，自2026年1月1日起施行。",
                }
            ]

    result = run_query("个人信息出境认证办法2025年修订了哪些条款？", adapters=[TimedPolicyAdapter()])

    assert result["time_or_version"] == "2026-01-01"


def test_run_query_policy_penalizes_generic_policy_faq_when_exact_entity_exists() -> None:
    clear_query_cache()

    class PolicyFaqAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "2025年政策变化问答",
                    "url": "https://www.gov.cn/zhengce/2025/policy-faq.htm",
                    "snippet": "2025年政策变化涉及多项外贸和监管新规。",
                },
                {
                    "title": "数据出境安全评估申报指南（第三版）",
                    "url": "https://www.cac.gov.cn/2025-06/27/c_1752652339765002.htm",
                    "snippet": "国家互联网信息办公室发布《数据出境安全评估申报指南（第三版）》，明确有效期延长条件与申报材料。",
                },
                {
                    "title": "数据出境安全评估办法",
                    "url": "https://www.gov.cn/zhengce/2022-07/07/content_5686197.htm",
                    "snippet": "《数据出境安全评估办法》规定了数据处理者申请安全评估的适用情形和流程。",
                },
            ]

    result = run_query("2025年数据出境安全评估办法有哪些变化？", adapters=[PolicyFaqAdapter()])

    assert result["sources"]
    assert "policy-faq" not in result["sources"][0]["url"]
    assert "数据出境安全评估" in result["sources"][0]["title"]
