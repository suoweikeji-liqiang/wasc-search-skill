from dataclasses import dataclass
import re

from skill.router import QueryAnalysis, analyze_query


@dataclass(frozen=True)
class QueryPlan:
    query: str
    lane: str


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _normalize_query(query: str) -> str:
    return " ".join(query.split()).strip()


def _aspect_suffix(analysis: QueryAnalysis, lane: str, use_cjk: bool) -> str:
    if lane == "policy":
        if "change" in analysis.aspects:
            return "修订 调整 变化" if use_cjk else "revision change update"
        if "exemption" in analysis.aspects:
            return "豁免 适用范围 问答" if use_cjk else "exemption scope FAQ"
        if "effective" in analysis.aspects:
            return "实施时间 适用条件" if use_cjk else "effective date scope"
        return "政策 规则 官方" if use_cjk else "policy regulation official"
    if lane == "academic":
        if "benchmark" in analysis.aspects:
            return "paper benchmark survey" if not use_cjk else "论文 基准 综述"
        return "paper arxiv survey"
    if "forecast" in analysis.aspects or "trend" in analysis.aspects:
        return "市场 预测 趋势" if use_cjk else "market forecast trend"
    return "产业 供应链 市场 影响" if use_cjk else "industry supply chain market impact"


def _dedupe_plans(plans: list[QueryPlan], limit: int) -> list[QueryPlan]:
    deduped: list[QueryPlan] = []
    seen: set[str] = set()
    for plan in plans:
        normalized = _normalize_query(plan.query).lower()
        if not normalized or normalized in seen:
            continue
        deduped.append(QueryPlan(query=_normalize_query(plan.query), lane=plan.lane))
        seen.add(normalized)
        if len(deduped) >= limit:
            break
    return deduped


def _build_policy_plan(query: str, analysis: QueryAnalysis) -> list[list[QueryPlan]]:
    base = analysis.entity_query or analysis.core_query or query
    use_cjk = _has_cjk(base)
    wave_one = [
        QueryPlan(query=f"site:gov.cn {base}", lane="policy"),
    ]
    aspect_suffix = _aspect_suffix(analysis, "policy", use_cjk)
    wave_one.append(QueryPlan(query=f"site:gov.cn {base} {aspect_suffix}", lane="policy"))
    wave_two = [
        QueryPlan(query=f"{base} 官方 {aspect_suffix}" if use_cjk else f"{base} official {aspect_suffix}", lane="policy")
    ]
    return [
        _dedupe_plans(wave_one, limit=2),
        _dedupe_plans(wave_two, limit=1),
    ]


def _build_industry_plan(query: str, analysis: QueryAnalysis) -> list[list[QueryPlan]]:
    base = analysis.core_query or query
    entity = analysis.entity_query or base
    use_cjk = _has_cjk(base)
    wave_one = [QueryPlan(query=base, lane="industry")]
    wave_one.append(QueryPlan(query=f"{entity} {_aspect_suffix(analysis, 'industry', use_cjk)}", lane="industry"))
    wave_two = [
        QueryPlan(
            query=f"{entity} vendor market share" if not use_cjk else f"{entity} 厂商 市场份额",
            lane="industry",
        )
    ]
    return [
        _dedupe_plans(wave_one, limit=2),
        _dedupe_plans(wave_two, limit=1),
    ]


def _build_academic_plan(query: str, analysis: QueryAnalysis) -> list[list[QueryPlan]]:
    base = analysis.core_query or query
    entity = analysis.entity_query or base
    wave_one = [
        QueryPlan(query=base, lane="academic"),
        QueryPlan(query=f"{entity} paper arxiv", lane="academic"),
    ]
    return [_dedupe_plans(wave_one, limit=2)]


def _build_mixed_plan(query: str, analysis: QueryAnalysis) -> list[list[QueryPlan]]:
    base = analysis.core_query or query
    entity = analysis.entity_query or base
    use_cjk = _has_cjk(base)
    secondary_lane = "academic" if analysis.academic_score > analysis.industry_score else "industry"
    policy_terms = _aspect_suffix(analysis, "policy", use_cjk)
    secondary_terms = _aspect_suffix(analysis, secondary_lane, use_cjk)
    wave_one = [
        QueryPlan(
            query=f"{entity} {policy_terms}",
            lane="policy",
        ),
        QueryPlan(
            query=f"{entity} {secondary_terms}",
            lane=secondary_lane,
        ),
    ]
    wave_two = [QueryPlan(query=base, lane=secondary_lane)]
    return [
        _dedupe_plans(wave_one, limit=2),
        _dedupe_plans(wave_two, limit=1),
    ]


def build_query_plan(query: str) -> list[list[QueryPlan]]:
    analysis = analyze_query(query)
    if analysis.intent == "policy":
        waves = _build_policy_plan(query, analysis)
        limit = 3
    elif analysis.intent == "industry":
        waves = _build_industry_plan(query, analysis)
        limit = 3
    elif analysis.intent == "academic":
        waves = _build_academic_plan(query, analysis)
        limit = 2
    else:
        waves = _build_mixed_plan(query, analysis)
        limit = 4

    flattened = [plan for wave in waves for plan in wave]
    trimmed = _dedupe_plans(flattened, limit=limit)
    rebuilt: list[list[QueryPlan]] = []
    cursor = 0
    for wave in waves:
        current_wave = trimmed[cursor : cursor + len(wave)]
        if current_wave:
            rebuilt.append(current_wave)
        cursor += len(current_wave)
        if cursor >= len(trimmed):
            break
    if not rebuilt:
        rebuilt.append([QueryPlan(query=_normalize_query(query), lane=analysis.intent)])
    return rebuilt
