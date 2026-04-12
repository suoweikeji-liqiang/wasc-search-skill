import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, TypedDict, cast

from skill.main import RunQueryResult, run_query
from skill.router import classify_query

Intent = Literal["policy", "academic", "industry", "mixed"]
VALID_INTENTS: set[Intent] = {"policy", "academic", "industry", "mixed"}


class EvalCase(TypedDict):
    id: str
    query: str
    expected_intent: Intent | None
    expected_terms: list[str]
    min_sources: int
    max_latency_ms: float
    min_keyword_coverage: float
    require_low_uncertainty: bool


class EvalCaseReport(TypedDict):
    id: str
    query: str
    passed: bool
    failed_checks: list[str]
    elapsed_ms: float
    summary: str
    sources_count: int
    uncertainties_count: int
    intent_predicted: Intent
    intent_match: bool | None
    keyword_coverage: float | None


class EvalSummary(TypedDict):
    total_cases: int
    passed_cases: int
    pass_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_sources: float
    uncertainty_rate: float
    intent_accuracy: float | None
    avg_keyword_coverage: float | None


def _to_non_negative_int(raw: Any, field_name: str) -> int:
    if not isinstance(raw, int):
        raise ValueError(f"{field_name} must be int")
    if raw < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return raw


def _to_positive_float(raw: Any, field_name: str) -> float:
    if not isinstance(raw, (int, float)):
        raise ValueError(f"{field_name} must be number")
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def _to_non_negative_float(raw: Any, field_name: str) -> float:
    if not isinstance(raw, (int, float)):
        raise ValueError(f"{field_name} must be number")
    value = float(raw)
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _normalize_expected_terms(raw_terms: Any) -> list[str]:
    if raw_terms is None:
        return []
    if not isinstance(raw_terms, list):
        raise ValueError("expected_terms must be list[str]")
    normalized: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        if not isinstance(term, str):
            raise ValueError("expected_terms must be list[str]")
        cleaned = term.strip()
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
    return normalized


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("dataset root must be list")

    cases: list[EvalCase] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError("each case must be object")

        raw_query = item.get("query")
        if not isinstance(raw_query, str) or not raw_query.strip():
            raise ValueError("query must be non-empty string")
        query = raw_query.strip()

        raw_case_id = item.get("id")
        if raw_case_id is None:
            case_id = f"case-{index}"
        elif isinstance(raw_case_id, str) and raw_case_id.strip():
            case_id = raw_case_id.strip()
        else:
            raise ValueError("id must be non-empty string when provided")

        raw_expected_intent = item.get("expected_intent")
        if raw_expected_intent is None:
            expected_intent = None
        elif isinstance(raw_expected_intent, str) and raw_expected_intent in VALID_INTENTS:
            expected_intent = cast(Intent, raw_expected_intent)
        else:
            raise ValueError("expected_intent must be one of policy/academic/industry/mixed")

        expected_terms = _normalize_expected_terms(item.get("expected_terms"))
        min_sources = _to_non_negative_int(item.get("min_sources", 1), "min_sources")
        max_latency_ms = _to_positive_float(item.get("max_latency_ms", 8000), "max_latency_ms")
        min_keyword_coverage = _to_non_negative_float(
            item.get("min_keyword_coverage", 0.5),
            "min_keyword_coverage",
        )
        if not expected_terms:
            min_keyword_coverage = 0.0
        require_low_uncertainty = bool(item.get("require_low_uncertainty", False))

        cases.append(
            EvalCase(
                id=case_id,
                query=query,
                expected_intent=expected_intent,
                expected_terms=expected_terms,
                min_sources=min_sources,
                max_latency_ms=max_latency_ms,
                min_keyword_coverage=min_keyword_coverage,
                require_low_uncertainty=require_low_uncertainty,
            )
        )

    return cases


def _keyword_coverage(expected_terms: list[str], text: str) -> float | None:
    if not expected_terms:
        return None
    lowered_text = text.lower()
    hits = sum(1 for term in expected_terms if term.lower() in lowered_text)
    return hits / len(expected_terms)


def evaluate_case(
    case: EvalCase | dict[str, Any],
    runner: Callable[[str], RunQueryResult | dict[str, Any]] = run_query,
    elapsed_ms: float | None = None,
) -> EvalCaseReport:
    query = str(case["query"])
    if elapsed_ms is None:
        started = perf_counter()
        raw_result = runner(query)
        elapsed_ms_value = (perf_counter() - started) * 1000
    else:
        raw_result = runner(query)
        elapsed_ms_value = float(elapsed_ms)

    result = cast(RunQueryResult, raw_result)
    summary = result["summary"]
    key_points = result["key_points"]
    source_text_parts = [
        " ".join([source.get("title", ""), source.get("url", "")]).strip()
        for source in result["sources"]
    ]
    sources_count = len(result["sources"])
    uncertainties_count = len(result["uncertainties"])

    merged_text = " ".join([summary, *key_points, *source_text_parts])
    coverage = _keyword_coverage(list(case.get("expected_terms", [])), merged_text)
    intent_predicted = classify_query(query)
    expected_intent = case.get("expected_intent")
    intent_match = None if expected_intent is None else intent_predicted == expected_intent

    failed_checks: list[str] = []
    if sources_count < int(case.get("min_sources", 1)):
        failed_checks.append("sources")
    if elapsed_ms_value > float(case.get("max_latency_ms", 8000)):
        failed_checks.append("latency")
    min_keyword_coverage = float(case.get("min_keyword_coverage", 0.0))
    if coverage is not None and coverage < min_keyword_coverage:
        failed_checks.append("keywords")
    if bool(case.get("require_low_uncertainty", False)) and uncertainties_count > 0:
        failed_checks.append("uncertainty")
    if intent_match is False:
        failed_checks.append("intent")

    return EvalCaseReport(
        id=str(case.get("id", "")),
        query=query,
        passed=not failed_checks,
        failed_checks=failed_checks,
        elapsed_ms=round(elapsed_ms_value, 2),
        summary=summary,
        sources_count=sources_count,
        uncertainties_count=uncertainties_count,
        intent_predicted=intent_predicted,
        intent_match=intent_match,
        keyword_coverage=coverage,
    )


def _safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[index], 2)


def summarize_reports(reports: list[EvalCaseReport] | list[dict[str, Any]]) -> EvalSummary:
    if not reports:
        return EvalSummary(
            total_cases=0,
            passed_cases=0,
            pass_rate=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            avg_sources=0.0,
            uncertainty_rate=0.0,
            intent_accuracy=None,
            avg_keyword_coverage=None,
        )

    total_cases = len(reports)
    passed_cases = sum(1 for item in reports if bool(item.get("passed")))
    latency_values = [float(item.get("elapsed_ms", 0.0)) for item in reports]
    source_values = [float(item.get("sources_count", 0.0)) for item in reports]
    uncertain_cases = sum(1 for item in reports if int(item.get("uncertainties_count", 0)) > 0)

    intent_matches: list[float] = []
    for item in reports:
        value = item.get("intent_match")
        if value is None:
            continue
        intent_matches.append(1.0 if bool(value) else 0.0)

    coverages: list[float] = []
    for item in reports:
        coverage = item.get("keyword_coverage")
        if coverage is None:
            continue
        coverages.append(float(coverage))

    return EvalSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        pass_rate=round(passed_cases / total_cases, 4),
        avg_latency_ms=_safe_avg(latency_values),
        p95_latency_ms=_percentile_95(latency_values),
        avg_sources=_safe_avg(source_values),
        uncertainty_rate=round(uncertain_cases / total_cases, 4),
        intent_accuracy=round(sum(intent_matches) / len(intent_matches), 4) if intent_matches else None,
        avg_keyword_coverage=round(sum(coverages) / len(coverages), 4) if coverages else None,
    )
