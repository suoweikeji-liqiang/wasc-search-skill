from typing import get_type_hints

from skill.main import RunQueryResult, run_query
from skill.router import analyze_query, classify_query


def test_classify_policy_query() -> None:
    assert classify_query("2025年数据出境安全评估办法有哪些变化？") == "policy"


def test_classify_policy_query_with_regulation_keyword() -> None:
    assert classify_query("促进和规范数据跨境流动规定中哪些场景可豁免？") == "policy"


def test_classify_academic_query() -> None:
    assert classify_query("2025 RAG chunking 最新论文综述") == "academic"


def test_classify_industry_query() -> None:
    assert classify_query("Vision Pro 当前销量预测如何？") == "industry"


def test_classify_mixed_query() -> None:
    assert classify_query("AI Act 对开源模型和产业落地影响") == "mixed"


def test_classify_mixed_query_with_policy_and_chip_supply() -> None:
    assert classify_query("美国出口管制规则对大模型训练芯片供给影响") == "mixed"


def test_non_policy_word_containing_act_does_not_match_policy() -> None:
    assert classify_query("最新市场 tactics 与销量策略") == "industry"


def test_run_query_return_type_is_precise() -> None:
    assert get_type_hints(run_query)["return"] is RunQueryResult


def test_classify_unmatched_query_defaults_to_mixed() -> None:
    assert classify_query("今天天气怎么样？") == "mixed"


def test_analyze_query_keeps_yearless_policy_entity_anchor() -> None:
    analysis = analyze_query("个人信息出境认证办法 2025 年修订了哪些条款？")

    assert analysis.intent == "policy"
    assert "个人信息出境认证办法" in analysis.anchor_terms
