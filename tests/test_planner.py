from skill.planner import build_query_plan
from skill.router import analyze_query


def _flatten_queries(query: str) -> list[tuple[str, str]]:
    waves = build_query_plan(query)
    return [(plan.lane, plan.query) for wave in waves for plan in wave]


def test_analyze_query_extracts_policy_change_signal_and_anchor_terms() -> None:
    analysis = analyze_query("2025年数据出境安全评估办法有哪些变化？")

    assert analysis.intent == "policy"
    assert "change" in analysis.aspects
    assert "2025" in analysis.years
    assert any("数据出境安全评估办法" in term for term in analysis.anchor_terms)


def test_build_query_plan_policy_change_is_budgeted_and_official_first() -> None:
    flat = _flatten_queries("2025年数据出境安全评估办法有哪些变化？")
    queries = [query for _, query in flat]

    assert len(queries) <= 3
    assert queries[0].startswith("site:gov.cn ")
    assert any(any(term in query for term in ("修订", "调整", "变化")) for query in queries)


def test_build_query_plan_mixed_spans_policy_and_industry_lanes() -> None:
    flat = _flatten_queries("AI Act 对开源模型和产业落地影响")
    lanes = {lane for lane, _ in flat}

    assert len(flat) <= 4
    assert "policy" in lanes
    assert "industry" in lanes
