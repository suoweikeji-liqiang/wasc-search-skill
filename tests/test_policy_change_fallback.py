from typing import Any
from unittest.mock import patch

from skill.main import run_query


def test_policy_change_query_recovers_change_sentence_outside_top_chunks() -> None:
    query = "2025\u5e74\u6570\u636e\u51fa\u5883\u5b89\u5168\u8bc4\u4f30\u529e\u6cd5\u6709\u54ea\u4e9b\u53d8\u5316\uff1f"
    generic_snippet_1 = (
        "\u6570\u636e\u51fa\u5883\u5b89\u5168\u8bc4\u4f30\u529e\u6cd5\u8986\u76d6\u9002\u7528"
        "\u8303\u56f4\u4e0e\u57fa\u672c\u6d41\u7a0b\u3002"
    )
    generic_snippet_2 = (
        "\u8be5\u529e\u6cd5\u7531\u76f8\u5173\u90e8\u95e8\u7ec4\u7ec7\u5b9e\u65bd\uff0c"
        "\u7528\u4e8e\u89c4\u8303\u6570\u636e\u51fa\u5883\u7ba1\u7406\u3002"
    )
    generic_snippet_3 = (
        "\u5408\u89c4\u4f01\u4e1a\u9700\u8981\u6309\u89c4\u5b8c\u6210\u8bc4\u4f30\u4e0e"
        "\u7533\u62a5\u3002"
    )
    change_snippet = (
        "\u4e0e\u65e7\u7248\u76f8\u6bd4\uff0c2025\u5e74\u65b0\u589e\u4f01\u4e1a\u81ea\u8bc4\u4f30"
        "\u8981\u6c42\uff0c\u5e76\u8c03\u6574\u7533\u62a5\u6d41\u7a0b\u3002"
    )

    class LateChangeAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {"title": "p1", "url": "https://www.gov.cn/p1", "snippet": generic_snippet_1},
                {"title": "p2", "url": "https://www.gov.cn/p2", "snippet": generic_snippet_2},
                {"title": "p3", "url": "https://www.gov.cn/p3", "snippet": generic_snippet_3},
                {"title": "p4", "url": "https://www.gov.cn/p4", "snippet": change_snippet},
            ]

    with (
        patch(
            "skill.main.select_top_chunks",
            return_value=[generic_snippet_1, generic_snippet_2, generic_snippet_3],
        ),
        patch("skill.main.classify_query", return_value="policy"),
    ):
        result = run_query(query, adapters=[LateChangeAdapter()])

    assert any(
        term in result["summary"]
        for term in ("\u53d8\u5316", "\u65b0\u589e", "\u8c03\u6574")
    )


def test_policy_change_query_prioritizes_change_keypoint_for_summary() -> None:
    query = "2025年数据出境安全评估办法有哪些变化？"
    generic_snippet = "数据出境安全评估办法用于规范数据跨境活动和申报流程。"
    change_snippet = "2025年新增企业自评估要求，并调整申报流程和材料。"

    class TwoSnippetAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {"title": "p1", "url": "https://www.gov.cn/p1", "snippet": generic_snippet},
                {"title": "p2", "url": "https://www.gov.cn/p2", "snippet": change_snippet},
            ]

    with (
        patch(
            "skill.main.select_top_chunks",
            return_value=[generic_snippet, change_snippet],
        ),
        patch("skill.main.classify_query", return_value="policy"),
    ):
        result = run_query(query, adapters=[TwoSnippetAdapter()])

    assert any(term in result["summary"] for term in ("变化", "新增", "调整"))


def test_policy_change_query_avoids_offtopic_change_sentence() -> None:
    query = "2025年数据出境安全评估办法有哪些变化？"
    noisy_change_snippet = (
        "数据出境安全评估、订立个人信息出境标准合同，或通过认证。"
        "美国财政部OFAC 2025年9月11日宣布，依经修订的13224号令制裁。"
    )
    clean_change_snippet = "《个人信息出境认证办法》2025年修订后新增企业自评估要求，并调整申报流程。"

    class NoisyFirstAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {"title": "p1", "url": "https://www.gov.cn/p1", "snippet": noisy_change_snippet},
                {"title": "p2", "url": "https://www.gov.cn/p2", "snippet": clean_change_snippet},
            ]

    with (
        patch(
            "skill.main.select_top_chunks",
            return_value=[noisy_change_snippet, clean_change_snippet],
        ),
        patch("skill.main.classify_query", return_value="policy"),
    ):
        result = run_query(query, adapters=[NoisyFirstAdapter()])

    assert "OFAC" not in result["summary"]
    assert any(term in result["summary"] for term in ("新增", "调整"))


def test_policy_change_query_uses_source_title_when_local_summary_is_noisy() -> None:
    query = "2025年数据出境安全评估办法有哪些变化？"

    class NoisySummaryAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "数据出境安全管理政策问答（2025年10月） - 中国网信网",
                    "url": "https://www.cac.gov.cn/2025-10/31/c_1763633376984070.htm",
                    "snippet": "数据 收集或者处理 活动，法律 、行政 法规另有规定的除外。",
                }
            ]

    with patch("skill.main.classify_query", return_value="policy"):
        result = run_query(query, adapters=[NoisySummaryAdapter()])

    assert result["summary"].startswith("数据出境安全管理政策问答")


def test_policy_change_query_filters_offtopic_key_points_when_possible() -> None:
    query = "2025年数据出境安全评估办法有哪些变化？"

    class MixedPointAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "数据出境安全管理政策问答（2025年10月） - 中国网信网",
                    "url": "https://www.cac.gov.cn/2025-10/31/c_1763633376984070.htm",
                    "snippet": "数据 收集或者处理 活动，法律 、行政 法规另有规定的除外。",
                },
                {
                    "title": "个人信息出境认证办法 - 中国网信网",
                    "url": "https://www.cac.gov.cn/2025-10/17/c_1762449728720008.htm",
                    "snippet": "《个人信息出境认证办法》2025年修订后新增企业自评估要求，并调整申报流程。",
                },
            ]

    with (
        patch("skill.main.classify_query", return_value="policy"),
        patch(
            "skill.main.select_top_chunks",
            return_value=[
                "美国财政部OFAC 2025年9月11日宣布，依经修订的13224号令制裁。",
                "《个人信息出境认证办法》2025年修订后新增企业自评估要求，并调整申报流程。",
            ],
        ),
    ):
        result = run_query(query, adapters=[MixedPointAdapter()])

    assert any(term in result["summary"] for term in ("新增", "调整", "变化"))
    assert all("OFAC" not in point for point in result["key_points"])


def test_policy_change_query_adds_uncertainty_when_change_evidence_missing() -> None:
    query = "2025年数据出境安全评估办法有哪些变化？"

    class GenericPolicyAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "数据出境安全管理政策问答（2025年10月） - 中国网信网",
                    "url": "https://www.cac.gov.cn/2025-10/31/c_1763633376984070.htm",
                    "snippet": "该问答围绕数据跨境场景和一般合规要求进行说明。",
                }
            ]

    with patch("skill.main.classify_query", return_value="policy"):
        result = run_query(query, adapters=[GenericPolicyAdapter()])

    assert any("未提取到明确变化条款" in item for item in result["uncertainties"])


def test_policy_change_query_injects_hint_terms_when_evidence_missing() -> None:
    query = "个人信息出境认证办法2025年修订了哪些条款？"

    class GenericPolicyAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "个人信息出境认证办法 - 中国网信网",
                    "url": "https://www.cac.gov.cn/2025-10/17/c_1762449728720008.htm",
                    "snippet": "该办法明确了总体框架和实施流程。",
                }
            ]

    with patch("skill.main.classify_query", return_value="policy"):
        result = run_query(query, adapters=[GenericPolicyAdapter()])

    merged = " ".join([result["summary"], *result["key_points"], *result["uncertainties"]])
    assert "修订" in merged
    assert "条款" in merged
    assert "新增" in merged
    assert "调整" in merged


def test_policy_exemption_query_injects_scene_hint_when_evidence_missing() -> None:
    query = "促进和规范数据跨境流动规定中哪些场景可豁免？"

    class GenericPolicyAdapter:
        async def search(self, query: str) -> list[dict[str, Any]]:
            _ = query
            return [
                {
                    "title": "促进和规范数据跨境流动规定",
                    "url": "https://www.cac.gov.cn/policy/rule",
                    "snippet": "该规定明确了数据跨境流动的总体框架和管理原则。",
                }
            ]

    with patch("skill.main.classify_query", return_value="policy"):
        result = run_query(query, adapters=[GenericPolicyAdapter()])

    merged = " ".join([result["summary"], *result["key_points"], *result["uncertainties"]])
    assert "豁免" in merged
    assert "场景" in merged
