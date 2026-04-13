import json
import re
from typing import Any

import httpx

MINIMAX_DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_FALLBACK_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_DEFAULT_MODEL = "MiniMax-M2.7"
MAX_SUMMARY_CHARS = 220
MAX_KEY_POINT_CHARS = 180
MAX_RETURN_SOURCES = 3

REQUIRED_RESPONSE_KEYS = {
    "summary",
    "key_points",
    "sources",
    "time_or_version",
    "uncertainties",
}


def _clean_text(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


def build_result(
    summary: str,
    key_points: list[str],
    sources: list[dict[str, str]],
    time_or_version: str,
    uncertainties: list[str],
) -> dict[str, Any]:
    return {
        "summary": summary,
        "key_points": key_points,
        "sources": sources,
        "time_or_version": time_or_version,
        "uncertainties": uncertainties,
    }


def _build_context_block(context_items: list[dict[str, str]]) -> str:
    lines: list[str] = []
    snippet_char_limit = 480
    for index, item in enumerate(context_items, start=1):
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        snippet = item.get("snippet", "").strip()
        if not (title and url and snippet):
            continue
        snippet = _clean_text(snippet, snippet_char_limit)
        lines.append(f"[{index}] {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(lines)


def _extract_json_payload(raw_content: str) -> dict[str, Any] | None:
    content = raw_content.strip()
    if not content:
        return None

    candidate_payloads: list[str] = [content]
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced_match:
        candidate_payloads.append(fenced_match.group(1))

    decoder = json.JSONDecoder()
    parsed_candidates: list[dict[str, Any]] = []
    best_effort: dict[str, Any] | None = None
    best_effort_span = -1
    for start_index, char in enumerate(content):
        if char != "{":
            continue
        try:
            parsed, end_offset = decoder.raw_decode(content[start_index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        parsed_candidates.append(parsed)
        remainder = content[start_index + end_offset :].strip()
        if not remainder:
            if REQUIRED_RESPONSE_KEYS.issubset(parsed.keys()):
                return parsed
            best_effort = parsed
            best_effort_span = end_offset
            continue
        if end_offset > best_effort_span:
            best_effort = parsed
            best_effort_span = end_offset

    for payload in candidate_payloads:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed_candidates.append(parsed)

    for candidate in parsed_candidates:
        if REQUIRED_RESPONSE_KEYS.issubset(candidate.keys()):
            return candidate

    if best_effort is not None:
        return best_effort
    return None


def _normalize_sources(raw_sources: Any) -> list[dict[str, str]]:
    if not isinstance(raw_sources, list):
        return []

    normalized: list[dict[str, str]] = []
    for source in raw_sources:
        if not isinstance(source, dict):
            continue
        title = source.get("title")
        url = source.get("url")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        title_value = title.strip()
        url_value = url.strip()
        if not title_value:
            continue
        normalized.append({"title": title_value, "url": url_value})
    if normalized:
        return normalized[:MAX_RETURN_SOURCES]

    for source in raw_sources:
        if not isinstance(source, str):
            continue
        source_text = source.strip()
        if not source_text:
            continue
        url_match = re.search(r"(https?://[^\s)]+)", source_text)
        if url_match:
            url = url_match.group(1)
            title = source_text.replace(f"({url})", "").strip(" -")
            normalized.append({"title": title or source_text, "url": url})
        else:
            normalized.append({"title": source_text, "url": ""})
    return normalized[:MAX_RETURN_SOURCES]


def _normalize_model_result(result: dict[str, Any]) -> dict[str, Any] | None:
    summary = result.get("summary")
    key_points = result.get("key_points")
    time_or_version = result.get("time_or_version")
    uncertainties = result.get("uncertainties")
    sources = _normalize_sources(result.get("sources"))

    if not isinstance(summary, str):
        return None

    if isinstance(key_points, str):
        normalized_key_points = [key_points]
    elif isinstance(key_points, list) and all(isinstance(item, str) for item in key_points):
        normalized_key_points = key_points
    else:
        normalized_key_points = [summary]

    if isinstance(time_or_version, str):
        normalized_time_or_version = time_or_version
    elif isinstance(time_or_version, (int, float)):
        normalized_time_or_version = str(time_or_version)
    else:
        normalized_time_or_version = "unknown"

    if isinstance(uncertainties, str):
        normalized_uncertainties = [uncertainties]
    elif isinstance(uncertainties, list) and all(isinstance(item, str) for item in uncertainties):
        normalized_uncertainties = uncertainties
    else:
        normalized_uncertainties = []

    cleaned_summary = _clean_text(summary, MAX_SUMMARY_CHARS)
    cleaned_key_points = [
        _clean_text(item, MAX_KEY_POINT_CHARS)
        for item in (normalized_key_points or [cleaned_summary])
    ][:3]

    return build_result(
        summary=cleaned_summary,
        key_points=cleaned_key_points,
        sources=sources,
        time_or_version=normalized_time_or_version,
        uncertainties=normalized_uncertainties,
    )


def _build_plain_text_fallback(
    raw_content: str,
    context_items: list[dict[str, str]],
) -> dict[str, Any] | None:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    if not cleaned:
        return None

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    summary = _clean_text(lines[0] if lines else cleaned, MAX_SUMMARY_CHARS)

    bullet_points = [
        line.lstrip("-• ").strip()
        for line in lines
        if line.startswith("-") or line.startswith("•")
    ]
    key_points = [
        _clean_text(point, MAX_KEY_POINT_CHARS)
        for point in (bullet_points[:3] if bullet_points else [summary])
    ]
    sources = [
        {"title": item["title"], "url": item["url"]}
        for item in context_items
        if item.get("title") and item.get("url")
    ][:MAX_RETURN_SOURCES]

    return build_result(
        summary=summary,
        key_points=key_points,
        sources=sources,
        time_or_version="unknown",
        uncertainties=["model output not strict json"],
    )


def generate_with_minimax(
    query: str,
    context_items: list[dict[str, str]],
    api_key: str,
    base_url: str | None = None,
    model: str = MINIMAX_DEFAULT_MODEL,
    timeout_seconds: float = 10.0,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any] | None:
    if not api_key.strip():
        return None
    context_block = _build_context_block(context_items)
    if not context_block:
        return None

    system_prompt = (
        "You are a grounded search synthesis assistant. "
        "Return valid JSON only with keys: summary, key_points, sources, time_or_version, uncertainties. "
        "Use only provided context. Preserve exact years, product names, regulation names, and source URLs from context. "
        "If evidence is incomplete, state that in uncertainties instead of guessing. Keep summary concise."
    )
    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Context:\n{context_block}\n\n"
        "Output JSON now."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    endpoint_candidates = (
        [base_url]
        if base_url is not None
        else [MINIMAX_DEFAULT_BASE_URL, MINIMAX_FALLBACK_BASE_URL]
    )
    client_kwargs: dict[str, Any] = {"timeout": timeout_seconds}
    if transport is not None:
        client_kwargs["transport"] = transport

    response: httpx.Response | None = None
    for candidate_base_url in endpoint_candidates:
        if candidate_base_url is None:
            continue
        endpoint = f"{candidate_base_url.rstrip('/')}/chat/completions"
        try:
            with httpx.Client(**client_kwargs) as client:
                current_response = client.post(endpoint, json=payload, headers=headers)
            if current_response.status_code == 401 and base_url is None:
                continue
            current_response.raise_for_status()
            response = current_response
            break
        except httpx.HTTPError:
            if base_url is not None:
                return None
            continue
    if response is None:
        return None

    response_body = response.json()
    try:
        raw_content = response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(raw_content, str):
        return None

    parsed = _extract_json_payload(raw_content)
    if parsed is not None:
        normalized_result = _normalize_model_result(parsed)
        if normalized_result is not None:
            return normalized_result
    return _build_plain_text_fallback(raw_content, context_items)
