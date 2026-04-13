from dataclasses import dataclass
import re
from typing import Literal

Intent = Literal["policy", "academic", "industry", "mixed"]

POLICY_KEYWORDS: dict[str, int] = {
    "政策": 3,
    "法规": 3,
    "规定": 3,
    "规则": 3,
    "办法": 3,
    "条例": 3,
    "通知": 2,
    "标准": 2,
    "指导意见": 2,
    "豁免": 2,
    "修订": 2,
    "调整": 2,
    "新增": 2,
    "实施": 2,
    "生效": 2,
    "认证": 2,
    "act": 3,
    "regulation": 3,
    "guideline": 2,
    "policy": 2,
    "cbam": 3,
    "tariff": 2,
    "export control": 3,
}

ACADEMIC_KEYWORDS: dict[str, int] = {
    "论文": 3,
    "paper": 3,
    "papers": 3,
    "arxiv": 3,
    "综述": 3,
    "研究": 2,
    "citation": 2,
    "benchmark": 3,
    "survey": 3,
    "review": 3,
    "openreview": 2,
}

INDUSTRY_KEYWORDS: dict[str, int] = {
    "公司": 1,
    "市场": 3,
    "销量": 3,
    "出货": 3,
    "出货量": 3,
    "行业": 2,
    "产业": 2,
    "芯片": 2,
    "供给": 2,
    "供应链": 3,
    "融资": 1,
    "产品": 1,
    "预测": 3,
    "趋势": 3,
    "份额": 3,
    "vendor": 2,
    "forecast": 3,
    "shipment": 3,
    "shipments": 3,
    "market share": 3,
    "cagr": 2,
    "trend": 3,
}

ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "change": ("变化", "修订", "调整", "新增", "更新", "修正"),
    "exemption": ("豁免", "适用范围", "适用条件", "例外"),
    "effective": ("实施时间", "生效", "effective date", "effective"),
    "trend": ("趋势", "走势"),
    "forecast": ("预测", "forecast", "market share", "shipment", "shipments", "销量", "出货量"),
    "benchmark": ("benchmark", "survey", "review", "papers", "paper", "综述", "论文"),
    "impact": ("影响", "impact", "对比", "比较"),
}

QUESTION_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[？?]+$"),
    re.compile(r"\b(latest|recent|current)\b", re.IGNORECASE),
    re.compile(r"(有哪些|有何|什么|如何|怎样|怎么样)$"),
    re.compile(r"(有哪些|有何|什么|如何|怎样|怎么样)"),
    re.compile(r"(吗|呢)$"),
)

ASPECT_STRIP_TERMS: tuple[str, ...] = (
    "有哪些变化",
    "有何变化",
    "什么变化",
    "变化",
    "修订了哪些条款",
    "修订条款",
    "修订",
    "调整",
    "新增",
    "更新",
    "哪些场景可豁免",
    "哪些场景豁免",
    "可豁免",
    "豁免",
    "适用范围",
    "适用条件",
    "实施时间",
    "生效时间",
    "影响",
    "对比",
    "比较",
    "趋势",
    "预测",
    "综述",
    "最新研究",
    "recent benchmark papers",
    "recent benchmark paper",
    "latest papers",
    "latest paper",
    "recent papers",
)

GENERIC_PREFIXES: tuple[str, ...] = ("当前", "最新", "最近", "关于", "请问")
GENERIC_SUFFIXES: tuple[str, ...] = (
    "政策",
    "法规",
    "规定",
    "规则",
    "办法",
    "条例",
    "影响",
    "趋势",
    "预测",
    "综述",
    "研究",
)


@dataclass(frozen=True)
class QueryAnalysis:
    intent: Intent
    policy_score: int
    academic_score: int
    industry_score: int
    aspects: tuple[str, ...]
    years: tuple[str, ...]
    core_query: str
    entity_query: str
    anchor_terms: tuple[str, ...]


def _matches_keyword(query: str, keyword: str) -> bool:
    if keyword.isascii() and re.fullmatch(r"[A-Za-z][A-Za-z ]*", keyword):
        return re.search(rf"\b{re.escape(keyword)}\b", query) is not None
    return keyword in query


def _score_keywords(query: str, mapping: dict[str, int]) -> int:
    return sum(weight for keyword, weight in mapping.items() if _matches_keyword(query, keyword))


def _normalize_query(query: str) -> str:
    normalized = " ".join(query.split()).strip()
    for pattern in QUESTION_NOISE_PATTERNS:
        normalized = pattern.sub("", normalized).strip()
    return normalized.strip(" ,.;:，。！？")


def _detect_aspects(lowered: str) -> tuple[str, ...]:
    aspects = [name for name, keywords in ASPECT_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)]
    return tuple(dict.fromkeys(aspects))


def _strip_aspects(query: str) -> str:
    stripped = query
    for term in ASPECT_STRIP_TERMS:
        stripped = stripped.replace(term, " ")
    stripped = " ".join(stripped.split()).strip()
    return stripped.strip(" ,.;:，。！？")


def _extract_anchor_terms(core_query: str, entity_query: str, years: tuple[str, ...]) -> tuple[str, ...]:
    candidates: list[str] = list(years)
    ascii_phrases = re.findall(
        r"[A-Za-z][A-Za-z0-9.+/-]*(?:\s+[A-Za-z0-9.+/-]+){0,2}",
        core_query,
    )
    for phrase in ascii_phrases:
        cleaned = " ".join(phrase.split()).strip()
        if len(cleaned) >= 2:
            candidates.append(cleaned)

    split_parts = re.split(r"[、,/]|对|和|与|及", entity_query)
    for raw_part in split_parts:
        cleaned = raw_part.strip()
        if not cleaned:
            continue
        for prefix in GENERIC_PREFIXES:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        for suffix in GENERIC_SUFFIXES:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 1:
                cleaned = cleaned[: -len(suffix)].strip()
        cleaned = cleaned.strip(" ,.;:，。！？")
        if len(cleaned) >= 2:
            candidates.append(cleaned)
            yearless = re.sub(r"20\d{2}年?", "", cleaned).strip()
            if len(yearless) >= 2:
                candidates.append(yearless)
                for suffix in ("办法", "规定", "条例", "规则", "政策", "指南"):
                    if yearless.endswith(suffix) and len(yearless) > len(suffix) + 1:
                        candidates.append(yearless[: -len(suffix)].strip())

    if entity_query and len(entity_query) >= 2:
        candidates.append(entity_query)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        lowered = candidate.lower()
        if lowered in seen or not candidate:
            continue
        deduped.append(candidate)
        seen.add(lowered)
    return tuple(deduped)


def analyze_query(query: str) -> QueryAnalysis:
    normalized = _normalize_query(query)
    lowered = normalized.lower()

    policy_score = _score_keywords(lowered, POLICY_KEYWORDS)
    academic_score = _score_keywords(lowered, ACADEMIC_KEYWORDS)
    industry_score = _score_keywords(lowered, INDUSTRY_KEYWORDS)

    matched = [policy_score > 0, academic_score > 0, industry_score > 0]
    if sum(matched) > 1:
        intent: Intent = "mixed"
    elif policy_score > 0:
        intent = "policy"
    elif academic_score > 0:
        intent = "academic"
    elif industry_score > 0:
        intent = "industry"
    else:
        intent = "mixed"

    aspects = _detect_aspects(lowered)
    years = tuple(dict.fromkeys(re.findall(r"20\d{2}", normalized)))
    entity_query = _strip_aspects(normalized) or normalized
    anchor_terms = _extract_anchor_terms(normalized, entity_query, years)

    return QueryAnalysis(
        intent=intent,
        policy_score=policy_score,
        academic_score=academic_score,
        industry_score=industry_score,
        aspects=aspects,
        years=years,
        core_query=normalized,
        entity_query=entity_query,
        anchor_terms=anchor_terms,
    )


def classify_query(query: str) -> Intent:
    return analyze_query(query).intent
