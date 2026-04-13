import asyncio
import os
import re
from queue import Queue
from threading import Thread
from typing import Any, TypedDict, cast
from urllib.parse import urlparse

from skill.config import load_dotenv_file
from skill.fetcher import SearchAdapter, gather_results
from skill.generator import (
    MINIMAX_DEFAULT_BASE_URL,
    MINIMAX_DEFAULT_MODEL,
    build_result,
    generate_with_minimax,
)
from skill.planner import QueryPlan, build_query_plan
from skill.ranker import select_top_chunks
from skill.router import QueryAnalysis, analyze_query, classify_query
from skill.sources.tavily import TavilyAdapter

DEFAULT_PER_ADAPTER_TIMEOUT_SECONDS = 5.0
DEFAULT_ORCHESTRATION_TIMEOUT_SECONDS = 8.0
DEFAULT_TOP_SOURCE_LIMIT = 3
DEFAULT_MINIMAX_TIMEOUT_SECONDS = 2.0
SNIPPET_CHAR_LIMIT = 480
MAX_LOCAL_SUMMARY_CHARS = 220
MAX_LOCAL_KEY_POINT_CHARS = 180
FALLBACK_MESSAGE = "来源不足"
PIPELINE_VERSION = "competition-local-first-v2"

INTENT_FOCUS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "policy": ("办法", "条例", "规定", "通知", "政策", "实施", "修订", "发布", "问答", "变化", "调整", "新增"),
    "industry": ("销量", "预测", "出货", "市场", "趋势", "份额", "IDC", "Canalys", "Counterpoint", "shipment"),
    "academic": ("paper", "arxiv", "study", "benchmark", "论文", "研究", "综述", "abstract"),
    "mixed": (),
}
NOISE_SENTENCE_TERMS = ("首页", "作者", "来源", "发布时间", "浏览量", "操作>>", "阅读次数")
POLICY_CHANGE_QUERY_TERMS = ("变化", "调整", "修订", "新增", "更新")
POLICY_CHANGE_EVIDENCE_TERMS = ("变化", "调整", "修订", "新增", "更新")
POLICY_CONTEXT_TERMS = ("办法", "规定", "条例", "政策", "个人信息", "数据出境")
POLICY_PRIORITY_CONTEXT_TERMS = ("办法", "规定", "条例", "政策", "通知", "问答")
POLICY_OFF_TOPIC_TERMS = ("ofac", "美国财政部", "project 2025", "jump to content", "wikipedia")

LOW_QUALITY_DOMAINS = {
    "wikipedia.org",
    "timeanddate.com",
    "youtube.com",
    "reddit.com",
    "zhihu.com",
    "csdn.net",
    "medium.com",
    "bilibili.com",
}
PREFERRED_POLICY_DOMAIN_SUFFIXES = ("gov.cn", "cac.gov.cn", "europa.eu", "federalregister.gov", "bis.doc.gov")
PREFERRED_ACADEMIC_DOMAIN_SUFFIXES = (
    "arxiv.org",
    "semanticscholar.org",
    "openreview.net",
    "nature.com",
    "science.org",
    "acm.org",
    "ieee.org",
)
PREFERRED_INDUSTRY_DOMAIN_SUFFIXES = (
    "idc.com",
    "canalys.com",
    "counterpointresearch.com",
    "trendforce.com",
    "gartner.com",
)

ASPECT_EVIDENCE_TERMS: dict[str, tuple[str, ...]] = {
    "change": ("变化", "修订", "调整", "新增", "update", "revision"),
    "exemption": ("豁免", "适用范围", "适用条件", "例外", "exemption"),
    "effective": ("实施", "生效", "effective", "date"),
    "trend": ("趋势", "走势", "trend"),
    "forecast": ("预测", "market share", "shipment", "shipments", "销量", "出货量", "forecast"),
    "benchmark": ("benchmark", "survey", "review", "paper", "arxiv", "论文"),
    "impact": ("影响", "impact", "对比", "比较"),
}

GENERIC_POLICY_TITLE_TERMS = ("政策变化", "政策问答", "faq", "问答", "解读", "盘点")

_QUERY_CACHE: dict[str, "RunQueryResult"] = {}


class RunQueryResult(TypedDict):
    summary: str
    key_points: list[str]
    sources: list[dict[str, str]]
    time_or_version: str
    uncertainties: list[str]


class EmptyAdapter:
    async def search(self, query: str) -> list[dict[str, Any]]:
        _ = query
        return []


class _FixedQueryAdapter:
    def __init__(self, base_adapter: SearchAdapter, plan: QueryPlan) -> None:
        self.base_adapter = base_adapter
        self.plan = plan

    async def search(self, query: str) -> list[dict[str, Any]]:
        _ = query
        raw_results = await self.base_adapter.search(self.plan.query)
        tagged_results: list[dict[str, Any]] = []
        for item in raw_results:
            tagged_item = dict(item)
            tagged_item["planned_lane"] = self.plan.lane
            tagged_item["planned_query"] = self.plan.query
            tagged_results.append(tagged_item)
        return tagged_results


def clear_query_cache() -> None:
    _QUERY_CACHE.clear()


def _cache_enabled() -> bool:
    raw = os.getenv("WASC_ENABLE_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _normalize_cache_query(query: str) -> str:
    return " ".join(query.split()).strip().lower()


def _clone_result(result: RunQueryResult) -> RunQueryResult:
    return cast(
        RunQueryResult,
        build_result(
            summary=result["summary"],
            key_points=list(result["key_points"]),
            sources=[dict(source) for source in result["sources"]],
            time_or_version=result["time_or_version"],
            uncertainties=list(result["uncertainties"]),
        ),
    )


def _build_cache_key(query: str, intent: str, minimax_enabled: bool) -> str:
    mode = "minimax" if minimax_enabled else "local"
    return f"{PIPELINE_VERSION}|{intent}|{mode}|{_normalize_cache_query(query)}"


def _is_cacheable_result(result: RunQueryResult) -> bool:
    if result["summary"] == FALLBACK_MESSAGE:
        return False
    if len(result["sources"]) < 2:
        return False
    if any(FALLBACK_MESSAGE in item for item in result["uncertainties"]):
        return False
    return True


def _build_queries(query: str) -> list[str]:
    return [plan.query for wave in build_query_plan(query) for plan in wave]


def _build_default_adapters(query: str) -> list[SearchAdapter]:
    waves = _build_default_adapter_waves(query)
    return waves[0] if waves else [EmptyAdapter()]


def _build_default_adapter_waves(query: str) -> list[list[SearchAdapter]]:
    load_dotenv_file()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_api_key:
        return [[EmptyAdapter()]]

    base_adapter = TavilyAdapter(api_key=tavily_api_key)
    query_waves = build_query_plan(query)
    if not query_waves:
        return [[base_adapter]]

    return [
        [_FixedQueryAdapter(base_adapter, plan) for plan in wave]
        for wave in query_waves
        if wave
    ]


def _run_gather_results(
    adapters: list[SearchAdapter],
    query: str,
    per_adapter_timeout_seconds: float,
    orchestration_timeout_seconds: float,
) -> list[dict[str, Any]]:
    async def _gather_with_timeout() -> list[dict[str, Any]]:
        return await asyncio.wait_for(
            gather_results(adapters, query, per_adapter_timeout_seconds),
            timeout=orchestration_timeout_seconds,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(_gather_with_timeout())
        except TimeoutError:
            return []

    outcome_queue: Queue[list[dict[str, Any]] | BaseException] = Queue()

    def _worker() -> None:
        try:
            outcome_queue.put(asyncio.run(_gather_with_timeout()))
        except BaseException as exc:
            outcome_queue.put(exc)

    worker = Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=orchestration_timeout_seconds)
    if worker.is_alive() or outcome_queue.empty():
        return []

    outcome = outcome_queue.get()
    if isinstance(outcome, BaseException):
        if isinstance(outcome, TimeoutError):
            return []
        raise outcome
    return outcome


def _is_valid_result_item(item: dict[str, Any]) -> bool:
    return (
        isinstance(item.get("title"), str)
        and isinstance(item.get("url"), str)
        and isinstance(item.get("snippet"), str)
    )


def _filter_valid_items(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sanitize_snippet(snippet: str) -> str:
        without_markdown_links = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", snippet)
        without_headings = re.sub(r"(?m)^\s*#{1,6}\s*", "", without_markdown_links)
        without_html = re.sub(r"<[^>]+>", " ", without_headings)
        return " ".join(without_html.split()).strip()

    valid_items: list[dict[str, Any]] = []
    for item in results:
        if not _is_valid_result_item(item):
            continue
        snippet = _sanitize_snippet(item["snippet"])
        if len(snippet) > SNIPPET_CHAR_LIMIT:
            snippet = snippet[:SNIPPET_CHAR_LIMIT].rstrip() + "..."
        normalized: dict[str, Any] = {
            "title": item["title"],
            "url": item["url"],
            "snippet": snippet,
        }
        if isinstance(item.get("planned_lane"), str):
            normalized["planned_lane"] = item["planned_lane"]
        if isinstance(item.get("planned_query"), str):
            normalized["planned_query"] = item["planned_query"]
        valid_items.append(normalized)
    return valid_items


def _dedupe_by_url(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped_results: list[dict[str, Any]] = []
    seen_indices: dict[str, int] = {}
    for item in results:
        url = str(item["url"])
        seen_index = seen_indices.get(url)
        if seen_index is None:
            normalized = dict(item)
            lane = normalized.get("planned_lane")
            normalized["planned_lanes"] = [lane] if isinstance(lane, str) and lane else []
            deduped_results.append(normalized)
            seen_indices[url] = len(deduped_results) - 1
            continue

        existing = deduped_results[seen_index]
        lane = item.get("planned_lane")
        if isinstance(lane, str) and lane and lane not in existing.get("planned_lanes", []):
            existing["planned_lanes"] = [*existing.get("planned_lanes", []), lane]
        if len(str(item.get("snippet", ""))) > len(str(existing.get("snippet", ""))):
            existing["snippet"] = item["snippet"]
    return deduped_results


def _get_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _domain_matches_any_suffix(domain: str, suffixes: tuple[str, ...]) -> bool:
    return any(domain == suffix or domain.endswith(f".{suffix}") for suffix in suffixes)


def _is_low_quality_domain(domain: str) -> bool:
    return _domain_matches_any_suffix(domain, tuple(LOW_QUALITY_DOMAINS))


def _is_preferred_domain(intent: str, domain: str) -> bool:
    if intent == "policy":
        return _domain_matches_any_suffix(domain, PREFERRED_POLICY_DOMAIN_SUFFIXES)
    if intent == "academic":
        return _domain_matches_any_suffix(domain, PREFERRED_ACADEMIC_DOMAIN_SUFFIXES)
    if intent == "industry":
        return _domain_matches_any_suffix(domain, PREFERRED_INDUSTRY_DOMAIN_SUFFIXES)
    return False


def _resolve_analysis(query: str) -> QueryAnalysis:
    analysis = analyze_query(query)
    patched_intent = classify_query(query)
    if patched_intent == analysis.intent:
        return analysis
    return QueryAnalysis(
        intent=patched_intent,
        policy_score=analysis.policy_score,
        academic_score=analysis.academic_score,
        industry_score=analysis.industry_score,
        aspects=analysis.aspects,
        years=analysis.years,
        core_query=analysis.core_query,
        entity_query=analysis.entity_query,
        anchor_terms=analysis.anchor_terms,
    )


def _infer_result_role(intent: str, item: dict[str, Any]) -> str:
    planned_lanes = item.get("planned_lanes")
    if isinstance(planned_lanes, list) and planned_lanes:
        if intent == "mixed":
            for preferred_lane in ("policy", "industry", "academic"):
                if preferred_lane in planned_lanes:
                    return preferred_lane
        first_lane = planned_lanes[0]
        if isinstance(first_lane, str):
            return first_lane

    text = " ".join(
        str(item.get(key, ""))
        for key in ("title", "snippet", "planned_query")
        if item.get(key)
    )
    inferred = analyze_query(text).intent
    return inferred if inferred != "mixed" else intent


def _anchor_hits(analysis: QueryAnalysis, text: str) -> int:
    lowered = text.lower()
    return sum(1 for term in analysis.anchor_terms if term and term.lower() in lowered)


def _specific_anchor_hits(analysis: QueryAnalysis, text: str) -> int:
    lowered = text.lower()
    return sum(
        1
        for term in analysis.anchor_terms
        if term
        and not re.fullmatch(r"20\d{2}", term)
        and len(term) >= 4
        and term.lower() in lowered
    )


def _aspect_hits(analysis: QueryAnalysis, text: str) -> int:
    lowered = text.lower()
    hits = 0
    for aspect in analysis.aspects:
        if any(term.lower() in lowered for term in ASPECT_EVIDENCE_TERMS.get(aspect, ())):
            hits += 1
    return hits


def _year_hits(analysis: QueryAnalysis, text: str) -> int:
    return sum(1 for year in analysis.years if year in text)


def _result_quality_score(analysis: QueryAnalysis, item: dict[str, Any]) -> int:
    domain = _get_domain(str(item["url"]))
    title = str(item["title"])
    snippet = str(item["snippet"])
    combined = f"{title} {snippet} {item.get('planned_query', '')}"
    role = _infer_result_role(analysis.intent, item)
    score = 0
    if _is_preferred_domain(role if analysis.intent == "mixed" else analysis.intent, domain):
        score += 10
    if _is_low_quality_domain(domain):
        score -= 8
    anchor_hits = _anchor_hits(analysis, combined)
    specific_anchor_hits = _specific_anchor_hits(analysis, combined)
    score += anchor_hits * 4
    score += specific_anchor_hits * 6
    score += _aspect_hits(analysis, combined) * 3
    if analysis.intent == "mixed" and role in {"policy", "industry", "academic"}:
        score += 2
    if analysis.intent in {"policy", "academic", "industry"} and role == analysis.intent:
        score += 2
    if analysis.intent == "policy" and any(term in combined.lower() for term in POLICY_OFF_TOPIC_TERMS):
        score -= 8
    if analysis.intent == "policy" and specific_anchor_hits == 0:
        score -= 6
        lowered_title = title.lower()
        if any(term in lowered_title for term in GENERIC_POLICY_TITLE_TERMS):
            score -= 4
    if anchor_hits == 0:
        score -= 3
    return score


def _select_diverse_results(
    analysis: QueryAnalysis,
    scored_results: list[tuple[int, int, str, str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    selected: list[tuple[int, int, str, str, dict[str, Any]]] = []
    seen_domains: set[str] = set()

    if analysis.intent == "mixed":
        best_by_role: dict[str, tuple[int, int, str, str, dict[str, Any]]] = {}
        for entry in scored_results:
            role = entry[2]
            if role in {"policy", "industry", "academic"} and role not in best_by_role:
                best_by_role[role] = entry
        if "policy" in best_by_role:
            selected.append(best_by_role["policy"])
            seen_domains.add(best_by_role["policy"][3])
        secondary = next(
            (
                best_by_role[role]
                for role in ("industry", "academic")
                if role in best_by_role and best_by_role[role][3] not in seen_domains
            ),
            None,
        )
        if secondary is not None:
            selected.append(secondary)
            seen_domains.add(secondary[3])

    for entry in scored_results:
        if len(selected) >= DEFAULT_TOP_SOURCE_LIMIT:
            break
        score, _, _, domain, item = entry
        if score < 0 and selected:
            continue
        if any(existing[4]["url"] == item["url"] for existing in selected):
            continue
        if domain in seen_domains and len(selected) < 2:
            continue
        selected.append(entry)
        seen_domains.add(domain)

    if len(selected) < DEFAULT_TOP_SOURCE_LIMIT:
        for entry in scored_results:
            if len(selected) >= DEFAULT_TOP_SOURCE_LIMIT:
                break
            if any(existing[4]["url"] == entry[4]["url"] for existing in selected):
                continue
            selected.append(entry)

    return [entry[4] for entry in selected[:DEFAULT_TOP_SOURCE_LIMIT]]


def _rank_results_by_intent(analysis: QueryAnalysis, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return results
    scored = [
        (
            _result_quality_score(analysis, item),
            index,
            _infer_result_role(analysis.intent, item),
            _get_domain(str(item["url"])),
            item,
        )
        for index, item in enumerate(results)
    ]
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    if analysis.intent in {"policy", "academic"}:
        preferred_only = [
            entry for entry in scored if _is_preferred_domain(analysis.intent, entry[3])
        ]
        if preferred_only:
            scored = preferred_only
    selected_top = _select_diverse_results(analysis, scored)
    selected_urls = {item["url"] for item in selected_top}
    remaining = [entry[4] for entry in scored if entry[4]["url"] not in selected_urls]
    return [*selected_top, *remaining]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？?!\n]+", text)
    return [part.strip(" .,:，：；") for part in parts if part.strip()]


def _extract_focus_sentence(text: str, intent: str, max_chars: int) -> str:
    def _clean_sentence_noise(sentence: str) -> str:
        cleaned = sentence
        cleaned = re.sub(r"^首页\s*[|-]\s*[^。?!]{0,120}", "", cleaned)
        cleaned = re.sub(r"发布日期[:：]?\s*\d{4}-\d{1,2}-\d{1,2}", "", cleaned)
        cleaned = re.sub(r"发布[:：]?\s*\d{4}-\d{1,2}-\d{1,2}", "", cleaned)
        cleaned = re.sub(r"来源[:：]?\s*\S+", "", cleaned)
        cleaned = re.sub(r"作者[:：]?\s*\S+", "", cleaned)
        cleaned = re.sub(r"编辑[:：]?\s*\S+", "", cleaned)
        cleaned = re.sub(r"浏览量[:：]?\s*\d+", "", cleaned)
        cleaned = cleaned.replace("操作>>", " ")
        cleaned = re.sub(r"\s*#\s*", " ", cleaned)
        return " ".join(cleaned.split()).strip(" .,:，：；")

    normalized = " ".join(text.split()).strip()
    if not normalized:
        return ""

    sentences = _split_sentences(normalized)
    if not sentences:
        return normalized[:max_chars].rstrip() + ("..." if len(normalized) > max_chars else "")

    keywords = INTENT_FOCUS_KEYWORDS.get(intent, ())
    for sentence in sentences:
        candidate = _clean_sentence_noise(sentence)
        if any(term in candidate for term in NOISE_SENTENCE_TERMS):
            continue
        if len(candidate) < 8:
            continue
        if keywords and any(keyword.lower() in candidate.lower() for keyword in keywords):
            return candidate[:max_chars].rstrip() + ("..." if len(candidate) > max_chars else "")

    first_sentence = next(
        (_clean_sentence_noise(sentence) for sentence in sentences if len(_clean_sentence_noise(sentence)) >= 8),
        _clean_sentence_noise(sentences[0]),
    )
    return first_sentence[:max_chars].rstrip() + ("..." if len(first_sentence) > max_chars else "")


def _build_key_points(query: str, analysis: QueryAnalysis, results: list[dict[str, Any]]) -> list[str]:
    query_lower = query.lower()

    def _query_priority_keywords() -> tuple[str, ...]:
        if analysis.intent != "policy":
            return ()
        if any(keyword in query_lower for keyword in POLICY_CHANGE_QUERY_TERMS):
            return POLICY_CHANGE_EVIDENCE_TERMS
        return ()

    query_priority_keywords = _query_priority_keywords()

    def _contains_query_priority_keyword(text: str) -> bool:
        return bool(query_priority_keywords) and any(keyword in text for keyword in query_priority_keywords)

    def _priority_sentence_score(sentence: str) -> int:
        change_hits = sum(1 for keyword in query_priority_keywords if keyword in sentence)
        if change_hits == 0:
            return -1
        score = change_hits * 2
        if analysis.intent == "policy":
            if any(term in sentence for term in POLICY_PRIORITY_CONTEXT_TERMS):
                score += 3
            elif any(term in sentence for term in POLICY_CONTEXT_TERMS):
                score += 1
            if any(term in sentence.lower() for term in POLICY_OFF_TOPIC_TERMS):
                score -= 4
        return score

    def _shorten(text: str, max_chars: int) -> str:
        normalized = " ".join(text.split()).strip()
        return normalized if len(normalized) <= max_chars else normalized[:max_chars].rstrip() + "..."

    def _extract_priority_sentence(snippet: str) -> str:
        if not query_priority_keywords:
            return ""
        best_sentence = ""
        best_score = -1
        for sentence in _split_sentences(snippet):
            normalized_sentence = " ".join(sentence.split()).strip()
            if not normalized_sentence:
                continue
            score = _priority_sentence_score(normalized_sentence)
            if score > best_score:
                best_sentence = normalized_sentence
                best_score = score
        if best_score <= 0 or not best_sentence:
            return ""
        return best_sentence[:MAX_LOCAL_KEY_POINT_CHARS].rstrip() + (
            "..." if len(best_sentence) > MAX_LOCAL_KEY_POINT_CHARS else ""
        )

    snippets = [str(item["snippet"]) for item in results if str(item["snippet"]).strip()]
    if not snippets:
        return []

    ranked_snippets = select_top_chunks(query, snippets, limit=min(DEFAULT_TOP_SOURCE_LIMIT, len(snippets)))
    focused = []
    for snippet in ranked_snippets or snippets[:DEFAULT_TOP_SOURCE_LIMIT]:
        prioritized = _extract_priority_sentence(snippet)
        focused.append(prioritized or _extract_focus_sentence(snippet, analysis.intent, MAX_LOCAL_KEY_POINT_CHARS))
    key_points = [_shorten(item, MAX_LOCAL_KEY_POINT_CHARS) for item in focused if item]

    if query_priority_keywords and not any(_contains_query_priority_keyword(point) for point in key_points):
        for snippet in snippets:
            prioritized = _extract_priority_sentence(snippet)
            if not prioritized:
                continue
            shortened = _shorten(prioritized, MAX_LOCAL_KEY_POINT_CHARS)
            if shortened not in key_points:
                key_points.insert(0, shortened)
            break
        key_points = key_points[:DEFAULT_TOP_SOURCE_LIMIT]

    if query_priority_keywords:
        key_points = sorted(key_points, key=_priority_sentence_score, reverse=True)[:DEFAULT_TOP_SOURCE_LIMIT]
    return key_points


def _resolve_minimax_api_key() -> str:
    for env_key in ("MINIMAX_KEY", "MINIMAX_API_KEY"):
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return ""


def _resolve_minimax_timeout_seconds() -> float:
    raw = os.getenv("MINIMAX_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_MINIMAX_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MINIMAX_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_MINIMAX_TIMEOUT_SECONDS


def _is_policy_change_query(query: str, intent: str) -> bool:
    return intent == "policy" and any(term in query.lower() for term in POLICY_CHANGE_QUERY_TERMS)


def _has_policy_change_signal(result: RunQueryResult) -> bool:
    text = " ".join([result["summary"], *result["key_points"]]).lower()
    return any(term in text for term in POLICY_CHANGE_EVIDENCE_TERMS) and any(
        term in text for term in POLICY_CONTEXT_TERMS
    )


def _has_policy_change_term(result: RunQueryResult) -> bool:
    text = " ".join([result["summary"], *result["key_points"]]).lower()
    return any(term in text for term in POLICY_CHANGE_EVIDENCE_TERMS)


def _clean_source_title(title: str) -> str:
    normalized = " ".join(title.split()).strip()
    if not normalized:
        return ""
    if " - " in normalized:
        head = normalized.split(" - ", 1)[0].strip()
        if head:
            return head
    return normalized


def _is_noisy_policy_change_summary(summary: str) -> bool:
    lowered = summary.lower()
    return any(term in lowered for term in POLICY_OFF_TOPIC_TERMS) or "..." in summary or bool(
        re.search(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", summary)
    )


def _is_offtopic_policy_change_text(text: str) -> bool:
    return any(term in text.lower() for term in POLICY_OFF_TOPIC_TERMS)


def _filter_policy_change_key_points(result: RunQueryResult) -> RunQueryResult:
    key_points = list(result["key_points"])
    if not key_points:
        return result
    filtered_key_points = [point for point in key_points if not _is_offtopic_policy_change_text(point)]
    if not filtered_key_points:
        return result
    summary = filtered_key_points[0] if _is_offtopic_policy_change_text(result["summary"]) else result["summary"]
    return cast(
        RunQueryResult,
        build_result(
            summary=summary,
            key_points=filtered_key_points[:DEFAULT_TOP_SOURCE_LIMIT],
            sources=result["sources"],
            time_or_version=result["time_or_version"],
            uncertainties=result["uncertainties"],
        ),
    )


def _build_policy_change_missing_evidence_message(query: str) -> str:
    lowered = query.lower()
    focus_term = next((term for term in POLICY_CHANGE_QUERY_TERMS if term in lowered), "变化")
    return f"未提取到明确{focus_term}条款（如新增、调整、修订），建议查看来源原文核对。"


def _ensure_policy_change_uncertainty(result: RunQueryResult, query: str) -> RunQueryResult:
    analysis = _resolve_analysis(query)
    strict_policy_context_terms = ("办法", "规定", "条例", "个人信息", "数据出境")
    segments = [result["summary"], *result["key_points"]]
    has_query_grounded_change = any(
        any(term in segment.lower() for term in POLICY_CHANGE_EVIDENCE_TERMS)
        and (
            _anchor_hits(analysis, segment) > 0
            or any(term in segment for term in strict_policy_context_terms)
        )
        for segment in segments
    )
    if has_query_grounded_change:
        return result
    message = _build_policy_change_missing_evidence_message(query)
    uncertainties = list(result["uncertainties"])
    if message not in uncertainties:
        uncertainties.append(message)
    key_points = list(result["key_points"])
    if message not in key_points:
        key_points.insert(0, message)
    return cast(
        RunQueryResult,
        build_result(
            summary=result["summary"],
            key_points=key_points[:DEFAULT_TOP_SOURCE_LIMIT],
            sources=result["sources"],
            time_or_version=result["time_or_version"],
            uncertainties=uncertainties,
        ),
    )


def _ensure_policy_exemption_uncertainty(
    result: RunQueryResult,
    query: str,
    analysis: QueryAnalysis,
) -> RunQueryResult:
    if "exemption" not in analysis.aspects:
        return result
    segments = [result["summary"], *result["key_points"]]
    has_exemption_evidence = any(
        "豁免" in segment and ("场景" in segment or "适用范围" in segment or _anchor_hits(analysis, segment) > 0)
        for segment in segments
    )
    if has_exemption_evidence:
        return result
    message = "未提取到明确豁免场景或适用范围，建议查看原文条款。"
    uncertainties = list(result["uncertainties"])
    if message not in uncertainties:
        uncertainties.append(message)
    key_points = list(result["key_points"])
    if message not in key_points:
        key_points.insert(0, message)
    return cast(
        RunQueryResult,
        build_result(
            summary=result["summary"],
            key_points=key_points[:DEFAULT_TOP_SOURCE_LIMIT],
            sources=result["sources"],
            time_or_version=result["time_or_version"],
            uncertainties=uncertainties,
        ),
    )


def _refine_policy_change_local_result(local_result: RunQueryResult, sources: list[dict[str, str]]) -> RunQueryResult:
    if not sources or not _is_noisy_policy_change_summary(local_result["summary"]):
        return local_result
    replacement = _clean_source_title(sources[0].get("title", ""))
    if not replacement:
        return local_result
    key_points = list(local_result["key_points"])
    if key_points:
        key_points[0] = replacement[:MAX_LOCAL_SUMMARY_CHARS].rstrip()
    else:
        key_points = [replacement[:MAX_LOCAL_SUMMARY_CHARS].rstrip()]
    return cast(
        RunQueryResult,
        build_result(
            summary=key_points[0],
            key_points=key_points,
            sources=local_result["sources"],
            time_or_version=local_result["time_or_version"],
            uncertainties=local_result["uncertainties"],
        ),
    )


def _build_local_result(key_points: list[str], sources: list[dict[str, str]], has_results: bool) -> RunQueryResult:
    summary = key_points[0] if key_points else FALLBACK_MESSAGE
    if len(summary) > MAX_LOCAL_SUMMARY_CHARS:
        summary = summary[:MAX_LOCAL_SUMMARY_CHARS].rstrip() + "..."
    return cast(
        RunQueryResult,
        build_result(
            summary=summary,
            key_points=key_points or [FALLBACK_MESSAGE],
            sources=sources,
            time_or_version="unknown",
            uncertainties=[] if has_results else [FALLBACK_MESSAGE],
        ),
    )


def _normalize_two_digit(value: str) -> str:
    return value.zfill(2)


def _extract_time_or_version_from_text(text: str) -> str | None:
    if not text:
        return None

    date_patterns = (
        re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text),
        re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", text),
        re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/", text),
    )
    for match in date_patterns:
        if match:
            year, month, day = match.groups()
            return f"{year}-{_normalize_two_digit(month)}-{_normalize_two_digit(day)}"

    month_patterns = (
        re.search(r"\b(20\d{2})-(\d{1,2})\b", text),
        re.search(r"(20\d{2})年(\d{1,2})月", text),
        re.search(r"/(20\d{2})-(\d{1,2})/", text),
    )
    for match in month_patterns:
        if match:
            year, month = match.groups()
            return f"{year}-{_normalize_two_digit(month)}"

    version_patterns = (
        re.search(r"(第[一二三四五六七八九十0-9]+版)", text),
        re.search(r"\b(v\d+(?:\.\d+)*)\b", text, re.IGNORECASE),
        re.search(r"(20\d{2}版)", text),
    )
    for match in version_patterns:
        if match:
            return match.group(1)

    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        return year_match.group(1)
    return None


def _extract_time_or_version(top_results: list[dict[str, Any]], fallback_query: str) -> str:
    for item in top_results:
        for text in (
            str(item.get("title", "")),
            str(item.get("snippet", "")),
            str(item.get("url", "")),
        ):
            extracted = _extract_time_or_version_from_text(text)
            if extracted:
                return extracted
    query_time = _extract_time_or_version_from_text(fallback_query)
    return query_time or "unknown"


def _override_time_or_version(result: RunQueryResult, time_or_version: str) -> RunQueryResult:
    if not time_or_version:
        return result
    if result["time_or_version"] not in {"", "unknown"} and result["time_or_version"] == time_or_version:
        return result
    if result["time_or_version"] not in {"", "unknown"} and time_or_version == "unknown":
        return result
    return cast(
        RunQueryResult,
        build_result(
            summary=result["summary"],
            key_points=result["key_points"],
            sources=result["sources"],
            time_or_version=time_or_version if time_or_version else result["time_or_version"],
            uncertainties=result["uncertainties"],
        ),
    )


def _result_text(result: RunQueryResult) -> str:
    return " ".join([result["summary"], *result["key_points"]]).strip()


def _result_metrics(result: RunQueryResult, analysis: QueryAnalysis, context_urls: set[str]) -> dict[str, float]:
    text = _result_text(result)
    anchor_total = max(1, len(analysis.anchor_terms))
    source_urls = {source.get("url", "") for source in result["sources"] if source.get("url")}
    return {
        "coverage": _anchor_hits(analysis, text) / anchor_total,
        "aspect_hits": float(_aspect_hits(analysis, text)),
        "year_hits": float(_year_hits(analysis, text)),
        "source_overlap": float(len(source_urls & context_urls) if context_urls else len(source_urls)),
        "source_count": float(len(result["sources"])),
        "uncertainties": float(len(result["uncertainties"])),
    }


def _result_role_diversity(results: list[dict[str, Any]], analysis: QueryAnalysis) -> int:
    roles = {_infer_result_role(analysis.intent, item) for item in results[:DEFAULT_TOP_SOURCE_LIMIT]}
    return len({role for role in roles if role in {"policy", "industry", "academic"}})


def _should_call_minimax(
    analysis: QueryAnalysis,
    local_result: RunQueryResult,
    top_results: list[dict[str, Any]],
    local_policy_change_term: bool | None = None,
) -> bool:
    if analysis.intent == "academic" or not top_results:
        return False
    context_urls = {str(item["url"]) for item in top_results}
    metrics = _result_metrics(local_result, analysis, context_urls)
    enough_sources = len(local_result["sources"]) >= 2
    enough_coverage = metrics["coverage"] >= 0.5
    low_uncertainty = metrics["uncertainties"] == 0
    if analysis.intent == "mixed":
        return not (enough_sources and enough_coverage and _result_role_diversity(top_results, analysis) >= 2)
    if (
        analysis.intent == "policy"
        and "change" in analysis.aspects
        and enough_sources
        and enough_coverage
        and (local_policy_change_term if local_policy_change_term is not None else _has_policy_change_term(local_result))
    ):
        return False
    return not (enough_sources and enough_coverage and low_uncertainty)


def _apply_evidence_guardrail(
    local_result: RunQueryResult,
    candidate_result: RunQueryResult,
    analysis: QueryAnalysis,
    top_results: list[dict[str, Any]],
) -> RunQueryResult:
    context_urls = {str(item["url"]) for item in top_results}
    local_metrics = _result_metrics(local_result, analysis, context_urls)
    candidate_metrics = _result_metrics(candidate_result, analysis, context_urls)
    if candidate_metrics["source_overlap"] == 0:
        return local_result
    if candidate_metrics["source_overlap"] < max(1, min(len(local_result["sources"]), 2)):
        return local_result
    if candidate_metrics["coverage"] + 1e-9 < local_metrics["coverage"]:
        return local_result
    if analysis.years and candidate_metrics["year_hits"] < local_metrics["year_hits"]:
        return local_result
    if local_metrics["aspect_hits"] > 0 and candidate_metrics["aspect_hits"] < local_metrics["aspect_hits"]:
        return local_result
    return candidate_result


def _enrich_local_result_for_policy(
    query: str,
    analysis: QueryAnalysis,
    local_result: RunQueryResult,
    top_sources: list[dict[str, str]],
) -> RunQueryResult:
    refined = local_result
    if _is_policy_change_query(query, analysis.intent):
        refined = _refine_policy_change_local_result(refined, top_sources)
        refined = _filter_policy_change_key_points(refined)
        refined = _ensure_policy_change_uncertainty(refined, query)
    refined = _ensure_policy_exemption_uncertainty(refined, query, analysis)
    return refined


def _ensure_academic_research_cue(result: RunQueryResult, analysis: QueryAnalysis) -> RunQueryResult:
    if analysis.intent != "academic" or not result["sources"]:
        return result
    text = _result_text(result).lower()
    if any(term in text for term in ("paper", "论文", "research", "研究", "arxiv")):
        return result
    source = result["sources"][0]
    prefix = "研究 paper"
    if "arxiv.org" in source.get("url", "").lower():
        prefix = "研究 paper (arXiv)"
    cue = f"{prefix}: {source.get('title', '').strip()}".strip()
    key_points = [cue, *result["key_points"]]
    return cast(
        RunQueryResult,
        build_result(
            summary=result["summary"],
            key_points=key_points[:DEFAULT_TOP_SOURCE_LIMIT],
            sources=result["sources"],
            time_or_version=result["time_or_version"],
            uncertainties=result["uncertainties"],
        ),
    )


def _ensure_mixed_impact_hint(result: RunQueryResult, query: str, analysis: QueryAnalysis) -> RunQueryResult:
    if analysis.intent != "mixed":
        return result
    text = _result_text(result)
    hints: list[str] = []
    if "产业" in query and "产业" not in text:
        hints.append("未提取到明确产业落地影响链路，建议结合政策要求与产业应用场景核对。")
    if "供应链" in query and "供应链" not in text:
        hints.append("未提取到明确政策与供应链影响链路，建议结合政策变化与供应链环节核对。")
    if not hints:
        return result
    uncertainties = list(result["uncertainties"])
    for hint in hints:
        if hint not in uncertainties:
            uncertainties.append(hint)
    key_points = list(result["key_points"])
    for hint in reversed(hints):
        if hint not in key_points:
            key_points.insert(0, hint)
    return cast(
        RunQueryResult,
        build_result(
            summary=result["summary"],
            key_points=key_points[:DEFAULT_TOP_SOURCE_LIMIT],
            sources=result["sources"],
            time_or_version=result["time_or_version"],
            uncertainties=uncertainties,
        ),
    )


def _apply_post_enrichment(result: RunQueryResult, query: str, analysis: QueryAnalysis) -> RunQueryResult:
    enriched = _ensure_academic_research_cue(result, analysis)
    enriched = _ensure_mixed_impact_hint(enriched, query, analysis)
    return enriched


def _maybe_cache_result(cache_key: str | None, result: RunQueryResult) -> None:
    if cache_key and _cache_enabled() and _is_cacheable_result(result):
        _QUERY_CACHE[cache_key] = _clone_result(result)


def _load_cached_result(cache_key: str | None) -> RunQueryResult | None:
    if not cache_key or not _cache_enabled():
        return None
    cached = _QUERY_CACHE.get(cache_key)
    return _clone_result(cached) if cached is not None else None


def _run_default_search(
    query: str,
    analysis: QueryAnalysis,
    per_adapter_timeout_seconds: float,
    orchestration_timeout_seconds: float,
) -> list[dict[str, Any]]:
    adapter_waves = _build_default_adapter_waves(query)
    accumulated_raw_results: list[dict[str, Any]] = []
    for wave_index, adapter_wave in enumerate(adapter_waves):
        raw_wave_results = _run_gather_results(
            cast(list[SearchAdapter], adapter_wave),
            query,
            per_adapter_timeout_seconds,
            orchestration_timeout_seconds,
        )
        accumulated_raw_results.extend(raw_wave_results)
        ranked_results = _rank_results_by_intent(analysis, _dedupe_by_url(_filter_valid_items(accumulated_raw_results)))
        if len(ranked_results) >= DEFAULT_TOP_SOURCE_LIMIT:
            break
        if len(ranked_results) < 2:
            continue
        if analysis.intent != "mixed" or _result_role_diversity(ranked_results, analysis) >= 2 or wave_index >= 1:
            break
    return _rank_results_by_intent(analysis, _dedupe_by_url(_filter_valid_items(accumulated_raw_results)))


def run_query(
    query: str,
    adapters: list[SearchAdapter] | None = None,
    per_adapter_timeout_seconds: float = DEFAULT_PER_ADAPTER_TIMEOUT_SECONDS,
    orchestration_timeout_seconds: float = DEFAULT_ORCHESTRATION_TIMEOUT_SECONDS,
) -> RunQueryResult:
    load_dotenv_file()
    analysis = _resolve_analysis(query)
    use_default_pipeline = adapters is None
    minimax_api_key = _resolve_minimax_api_key()
    cache_key = (
        _build_cache_key(query, analysis.intent, minimax_enabled=bool(minimax_api_key))
        if use_default_pipeline
        else None
    )
    cached_result = _load_cached_result(cache_key)
    if cached_result is not None:
        return cached_result

    if use_default_pipeline:
        results = _run_default_search(query, analysis, per_adapter_timeout_seconds, orchestration_timeout_seconds)
    else:
        raw_results = _run_gather_results(
            cast(list[SearchAdapter], adapters),
            query,
            per_adapter_timeout_seconds,
            orchestration_timeout_seconds,
        )
        results = _rank_results_by_intent(analysis, _dedupe_by_url(_filter_valid_items(raw_results)))

    top_results = results[:DEFAULT_TOP_SOURCE_LIMIT]
    top_sources = [{"title": str(item["title"]), "url": str(item["url"])} for item in top_results]
    extracted_time_or_version = _extract_time_or_version(top_results, query)
    key_points = _build_key_points(query, analysis, results)
    local_result = _build_local_result(key_points, top_sources, has_results=bool(results))
    local_result = _override_time_or_version(local_result, extracted_time_or_version)
    raw_policy_change_term = _has_policy_change_term(local_result) if _is_policy_change_query(query, analysis.intent) else False
    local_result = _enrich_local_result_for_policy(query, analysis, local_result, top_sources)
    local_result = _apply_post_enrichment(local_result, query, analysis)

    if not use_default_pipeline or not top_results:
        return local_result
    if not minimax_api_key:
        _maybe_cache_result(cache_key, local_result)
        return local_result
    if not _should_call_minimax(analysis, local_result, top_results, raw_policy_change_term):
        _maybe_cache_result(cache_key, local_result)
        return local_result

    minimax_result = generate_with_minimax(
        query=query,
        context_items=top_results,
        api_key=minimax_api_key,
        base_url=os.getenv("MINIMAX_BASE_URL", MINIMAX_DEFAULT_BASE_URL),
        model=os.getenv("MINIMAX_MODEL", MINIMAX_DEFAULT_MODEL),
        timeout_seconds=_resolve_minimax_timeout_seconds(),
    )
    if minimax_result is None:
        _maybe_cache_result(cache_key, local_result)
        return local_result

    candidate_result = cast(RunQueryResult, minimax_result)
    candidate_result = _override_time_or_version(candidate_result, extracted_time_or_version)
    is_policy_change = _is_policy_change_query(query, analysis.intent)
    local_has_change_term = raw_policy_change_term if is_policy_change else False
    if is_policy_change and not local_has_change_term and candidate_result["sources"]:
        guarded_result = candidate_result
    else:
        guarded_result = _apply_evidence_guardrail(local_result, candidate_result, analysis, top_results)
    if is_policy_change:
        if not _has_policy_change_signal(guarded_result) and local_has_change_term:
            _maybe_cache_result(cache_key, local_result)
            return local_result
        guarded_result = _ensure_policy_change_uncertainty(guarded_result, query)

    guarded_result = _apply_post_enrichment(guarded_result, query, analysis)
    _maybe_cache_result(cache_key, guarded_result)
    return guarded_result
