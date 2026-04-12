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
from skill.ranker import select_top_chunks
from skill.router import classify_query
from skill.sources.academic import build_academic_queries
from skill.sources.tavily import TavilyAdapter
from skill.sources.web import build_web_queries

DEFAULT_PER_ADAPTER_TIMEOUT_SECONDS = 5.0
DEFAULT_ORCHESTRATION_TIMEOUT_SECONDS = 8.0
DEFAULT_TOP_SOURCE_LIMIT = 3
FALLBACK_MESSAGE = "来源不足"
SNIPPET_CHAR_LIMIT = 480
DEFAULT_MINIMAX_TIMEOUT_SECONDS = 2.0
MAX_LOCAL_SUMMARY_CHARS = 220
MAX_LOCAL_KEY_POINT_CHARS = 180

INTENT_FOCUS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "policy": ("办法", "条例", "规定", "通知", "政策", "实施", "修订", "发布", "问答", "变化", "调整", "新增"),
    "industry": ("销量", "预测", "出货", "市场", "预计", "同比", "IDC", "Canalys", "Counterpoint"),
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
PREFERRED_POLICY_DOMAIN_SUFFIXES = ("gov.cn",)
PREFERRED_ACADEMIC_DOMAIN_SUFFIXES = (
    "arxiv.org",
    "semanticscholar.org",
    "openreview.net",
    "nature.com",
    "science.org",
    "acm.org",
    "ieee.org",
)


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
    def __init__(self, base_adapter: SearchAdapter, fixed_query: str) -> None:
        self.base_adapter = base_adapter
        self.fixed_query = fixed_query

    async def search(self, query: str) -> list[dict[str, Any]]:
        _ = query
        return await self.base_adapter.search(self.fixed_query)


def _build_queries(query: str) -> list[str]:
    intent = classify_query(query)
    if intent == "academic":
        return build_academic_queries(query)
    if intent == "mixed":
        mixed_queries = build_web_queries(query, "industry") + build_academic_queries(query)
        return list(dict.fromkeys(mixed_queries))
    return build_web_queries(query, intent)


def _build_default_adapters(query: str) -> list[SearchAdapter]:
    load_dotenv_file()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_api_key:
        return [EmptyAdapter()]

    base_adapter = TavilyAdapter(api_key=tavily_api_key)
    queries = _build_queries(query)
    if not queries:
        return [base_adapter]
    return [_FixedQueryAdapter(base_adapter, built_query) for built_query in queries]


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


def _filter_valid_items(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    def _sanitize_snippet(snippet: str) -> str:
        without_markdown_links = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", snippet)
        without_headings = re.sub(r"(?m)^\s*#{1,6}\s*", "", without_markdown_links)
        without_html = re.sub(r"<[^>]+>", " ", without_headings)
        normalized_whitespace = " ".join(without_html.split()).strip()
        return normalized_whitespace

    valid_items: list[dict[str, str]] = []
    for item in results:
        if _is_valid_result_item(item):
            snippet = _sanitize_snippet(item["snippet"])
            if len(snippet) > SNIPPET_CHAR_LIMIT:
                snippet = snippet[:SNIPPET_CHAR_LIMIT].rstrip() + "..."
            valid_items.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": snippet,
                }
            )
    return valid_items


def _dedupe_by_url(results: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped_results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in results:
        if item["url"] in seen_urls:
            continue
        seen_urls.add(item["url"])
        deduped_results.append(item)
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
    return False


def _result_quality_score(intent: str, item: dict[str, str]) -> int:
    domain = _get_domain(item["url"])
    score = 0

    if _is_preferred_domain(intent, domain):
        score += 10
    if _is_low_quality_domain(domain):
        score -= 6

    title = item["title"].lower()
    snippet = item["snippet"].lower()
    if intent == "policy":
        if any(keyword in title or keyword in snippet for keyword in ("政策", "法规", "办法", "条例")):
            score += 2
    if intent == "academic":
        if any(keyword in title or keyword in snippet for keyword in ("paper", "arxiv", "论文", "research")):
            score += 2

    return score


def _rank_results_by_intent(
    intent: str,
    results: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not results:
        return results

    scored = [
        (index, item, _result_quality_score(intent, item))
        for index, item in enumerate(results)
    ]
    scored.sort(key=lambda entry: (-entry[2], entry[0]))
    ranked = [item for _, item, _ in scored]

    if intent in {"policy", "academic"}:
        preferred = [item for item in ranked if _is_preferred_domain(intent, _get_domain(item["url"]))]
        if preferred:
            if intent in {"policy", "academic"}:
                return preferred
    return ranked


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!?；;\n]+", text)
    return [part.strip(" .,:，：；;") for part in parts if part.strip()]


def _extract_focus_sentence(text: str, intent: str, max_chars: int) -> str:
    def _clean_sentence_noise(sentence: str) -> str:
        cleaned = sentence
        cleaned = re.sub(r"^首页\s*[›>]\s*[^。.!?]{0,120}", "", cleaned)
        cleaned = re.sub(r"发布[:：]\s*\d{4}-\d{1,2}-\d{1,2}", "", cleaned)
        cleaned = re.sub(r"来源[:：]\s*\S+", "", cleaned)
        cleaned = re.sub(r"作者[:：]\s*\S+", "", cleaned)
        cleaned = re.sub(r"编审[:：]\s*\S+", "", cleaned)
        cleaned = re.sub(r"浏览量[:：]\s*\d+", "", cleaned)
        cleaned = cleaned.replace("操作>>", " ")
        cleaned = re.sub(r"\s*#\s*", " ", cleaned)
        return " ".join(cleaned.split()).strip(" .,:，：；;")

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
    for sentence in sentences:
        candidate = _clean_sentence_noise(sentence)
        if len(candidate) < 8:
            continue
        if keywords and any(keyword.lower() in candidate.lower() for keyword in keywords):
            return candidate[:max_chars].rstrip() + ("..." if len(candidate) > max_chars else "")

    first_sentence = next(
        (_clean_sentence_noise(sentence) for sentence in sentences if len(_clean_sentence_noise(sentence)) >= 8),
        _clean_sentence_noise(sentences[0]),
    )
    return first_sentence[:max_chars].rstrip() + ("..." if len(first_sentence) > max_chars else "")


def _build_key_points(query: str, intent: str, results: list[dict[str, str]]) -> list[str]:
    query_lower = query.lower()

    def _query_priority_keywords() -> tuple[str, ...]:
        if intent != "policy":
            return ()
        if any(keyword in query_lower for keyword in POLICY_CHANGE_QUERY_TERMS):
            return POLICY_CHANGE_EVIDENCE_TERMS
        return ()

    query_priority_keywords = _query_priority_keywords()

    def _contains_query_priority_keyword(text: str) -> bool:
        if not query_priority_keywords:
            return False
        return any(keyword in text for keyword in query_priority_keywords)

    def _priority_sentence_score(sentence: str) -> int:
        change_hits = sum(1 for keyword in query_priority_keywords if keyword in sentence)
        if change_hits == 0:
            return -1

        score = change_hits * 2
        if intent == "policy":
            if any(term in sentence for term in POLICY_PRIORITY_CONTEXT_TERMS):
                score += 3
            elif any(term in sentence for term in POLICY_CONTEXT_TERMS):
                score += 1

            lowered_sentence = sentence.lower()
            if any(term in lowered_sentence for term in POLICY_OFF_TOPIC_TERMS):
                score -= 4
        return score

    def _shorten(text: str, max_chars: int) -> str:
        normalized = " ".join(text.split()).strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip() + "..."

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
            if score <= best_score:
                continue
            best_sentence = normalized_sentence
            best_score = score

        if best_score <= 0 or not best_sentence:
            return ""
        return best_sentence[:MAX_LOCAL_KEY_POINT_CHARS].rstrip() + (
            "..." if len(best_sentence) > MAX_LOCAL_KEY_POINT_CHARS else ""
        )

    def _extract_with_query_priority(snippet: str) -> str:
        prioritized_sentence = _extract_priority_sentence(snippet)
        if prioritized_sentence:
            return prioritized_sentence
        return _extract_focus_sentence(snippet, intent, MAX_LOCAL_KEY_POINT_CHARS)

    snippets = [item["snippet"] for item in results if item["snippet"].strip()]
    if not snippets:
        return []

    ranked_snippets = select_top_chunks(
        query,
        snippets,
        limit=min(DEFAULT_TOP_SOURCE_LIMIT, len(snippets)),
    )
    focused = [
        _extract_with_query_priority(snippet)
        for snippet in (ranked_snippets or snippets[:DEFAULT_TOP_SOURCE_LIMIT])
    ]
    key_points = [_shorten(item, MAX_LOCAL_KEY_POINT_CHARS) for item in focused if item]

    if query_priority_keywords and not any(_contains_query_priority_keyword(point) for point in key_points):
        for snippet in snippets:
            prioritized_sentence = _extract_priority_sentence(snippet)
            if not prioritized_sentence:
                continue
            shortened_prioritized_sentence = _shorten(prioritized_sentence, MAX_LOCAL_KEY_POINT_CHARS)
            if shortened_prioritized_sentence in key_points:
                continue
            key_points.insert(0, shortened_prioritized_sentence)
            break
        key_points = key_points[:DEFAULT_TOP_SOURCE_LIMIT]

    if query_priority_keywords:
        key_points = sorted(
            key_points,
            key=lambda point: _priority_sentence_score(point),
            reverse=True,
        )[:DEFAULT_TOP_SOURCE_LIMIT]

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
    if value <= 0:
        return DEFAULT_MINIMAX_TIMEOUT_SECONDS
    return value


def _is_policy_change_query(query: str, intent: str) -> bool:
    if intent != "policy":
        return False
    lowered = query.lower()
    return any(term in lowered for term in POLICY_CHANGE_QUERY_TERMS)


def _has_policy_change_signal(result: RunQueryResult) -> bool:
    text = " ".join([result["summary"], *result["key_points"]]).lower()
    has_change_term = any(term in text for term in POLICY_CHANGE_EVIDENCE_TERMS)
    has_policy_context = any(term in text for term in POLICY_CONTEXT_TERMS)
    return has_change_term and has_policy_context


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
    if any(term in lowered for term in POLICY_OFF_TOPIC_TERMS):
        return True
    if "..." in summary:
        return True
    if re.search(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", summary):
        return True
    return False


def _is_offtopic_policy_change_text(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in POLICY_OFF_TOPIC_TERMS)


def _filter_policy_change_key_points(result: RunQueryResult) -> RunQueryResult:
    key_points = list(result["key_points"])
    if not key_points:
        return result

    filtered_key_points = [
        point for point in key_points if not _is_offtopic_policy_change_text(point)
    ]
    if not filtered_key_points:
        return result

    summary = result["summary"]
    if _is_offtopic_policy_change_text(summary):
        summary = filtered_key_points[0]

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
    focus_term = next(
        (term for term in POLICY_CHANGE_QUERY_TERMS if term in lowered),
        "变化",
    )
    return f"未提取到明确{focus_term}条款（如新增、调整、修订），建议查看来源原文核对。"


def _ensure_policy_change_uncertainty(result: RunQueryResult, query: str) -> RunQueryResult:
    if _has_policy_change_term(result):
        return result

    message = _build_policy_change_missing_evidence_message(query)
    uncertainties = list(result["uncertainties"])
    if message not in uncertainties:
        uncertainties.append(message)

    key_points = list(result["key_points"])
    if message not in key_points:
        key_points.insert(0, message)
    key_points = key_points[:DEFAULT_TOP_SOURCE_LIMIT]

    return cast(
        RunQueryResult,
        build_result(
            summary=result["summary"],
            key_points=key_points,
            sources=result["sources"],
            time_or_version=result["time_or_version"],
            uncertainties=uncertainties,
        ),
    )


def _refine_policy_change_local_result(
    local_result: RunQueryResult,
    sources: list[dict[str, str]],
) -> RunQueryResult:
    if not sources:
        return local_result
    if not _is_noisy_policy_change_summary(local_result["summary"]):
        return local_result

    replacement = _clean_source_title(sources[0].get("title", ""))
    if not replacement:
        return local_result

    replacement = replacement[:MAX_LOCAL_SUMMARY_CHARS].rstrip()
    key_points = list(local_result["key_points"])
    if key_points:
        key_points[0] = replacement
    else:
        key_points = [replacement]

    return cast(
        RunQueryResult,
        build_result(
            summary=replacement,
            key_points=key_points,
            sources=local_result["sources"],
            time_or_version=local_result["time_or_version"],
            uncertainties=local_result["uncertainties"],
        ),
    )


def _build_local_result(
    key_points: list[str],
    sources: list[dict[str, str]],
    has_results: bool,
) -> RunQueryResult:
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


def run_query(
    query: str,
    adapters: list[SearchAdapter] | None = None,
    per_adapter_timeout_seconds: float = DEFAULT_PER_ADAPTER_TIMEOUT_SECONDS,
    orchestration_timeout_seconds: float = DEFAULT_ORCHESTRATION_TIMEOUT_SECONDS,
) -> RunQueryResult:
    load_dotenv_file()
    query_intent = classify_query(query)
    use_default_pipeline = adapters is None
    chosen_adapters = _build_default_adapters(query) if use_default_pipeline else adapters

    raw_results: list[dict[str, Any]] = _run_gather_results(
        cast(list[SearchAdapter], chosen_adapters),
        query,
        per_adapter_timeout_seconds,
        orchestration_timeout_seconds,
    )
    results = _rank_results_by_intent(
        query_intent,
        _dedupe_by_url(_filter_valid_items(raw_results)),
    )
    top_results = results[:DEFAULT_TOP_SOURCE_LIMIT]
    top_sources: list[dict[str, str]] = [
        {"title": item["title"], "url": item["url"]} for item in top_results
    ]
    key_points = _build_key_points(query, query_intent, results)
    local_result = _build_local_result(key_points, top_sources, has_results=bool(results))
    is_policy_change = _is_policy_change_query(query, query_intent)
    local_has_change_term = False
    if is_policy_change:
        local_result = _refine_policy_change_local_result(local_result, top_sources)
        local_result = _filter_policy_change_key_points(local_result)
        local_has_change_term = _has_policy_change_term(local_result)
        local_result = _ensure_policy_change_uncertainty(local_result, query)

    if not use_default_pipeline or not top_results:
        return local_result

    minimax_api_key = _resolve_minimax_api_key()
    if not minimax_api_key:
        return local_result

    if query_intent == "academic":
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
        return local_result

    normalized_minimax_result = cast(RunQueryResult, minimax_result)
    if is_policy_change:
        minimax_has_change_signal = _has_policy_change_signal(normalized_minimax_result)
        if not minimax_has_change_signal and local_has_change_term:
            return local_result
        return _ensure_policy_change_uncertainty(normalized_minimax_result, query)

    return normalized_minimax_result
