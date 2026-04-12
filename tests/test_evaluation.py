import json
from pathlib import Path

from skill.evaluation import evaluate_case, load_eval_cases, summarize_reports


def test_load_eval_cases_applies_defaults(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "policy-1",
                    "query": "2025\u5e74\u6570\u636e\u51fa\u5883\u5b89\u5168\u8bc4\u4f30\u529e\u6cd5\u6709\u54ea\u4e9b\u53d8\u5316\uff1f",
                    "expected_intent": "policy",
                    "expected_terms": ["\u65b0\u589e", "\u8c03\u6574"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_eval_cases(dataset_path)

    assert len(cases) == 1
    case = cases[0]
    assert case["id"] == "policy-1"
    assert case["min_sources"] == 1
    assert case["max_latency_ms"] == 8000
    assert case["min_keyword_coverage"] == 0.5
    assert case["require_low_uncertainty"] is False


def test_evaluate_case_computes_metrics_and_passes() -> None:
    case = {
        "id": "case-1",
        "query": "2025\u5e74\u6570\u636e\u51fa\u5883\u529e\u6cd5\u53d8\u5316",
        "expected_intent": "policy",
        "expected_terms": ["\u53d8\u5316", "\u65b0\u589e"],
        "min_sources": 1,
        "max_latency_ms": 2000,
        "min_keyword_coverage": 0.5,
        "require_low_uncertainty": True,
    }

    def fake_runner(query: str) -> dict:
        _ = query
        return {
            "summary": "2025\u5e74\u65b0\u589e\u7533\u62a5\u6761\u6b3e\uff0c\u540c\u6b65\u8c03\u6574\u6d41\u7a0b\u3002",
            "key_points": ["\u53d8\u5316\u805a\u7126\u5728\u7533\u62a5\u8d23\u4efb\u548c\u6750\u6599\u8981\u6c42\u3002"],
            "sources": [{"title": "A", "url": "https://www.gov.cn/a"}],
            "time_or_version": "unknown",
            "uncertainties": [],
        }

    report = evaluate_case(case, runner=fake_runner, elapsed_ms=1234.0)

    assert report["passed"] is True
    assert report["intent_match"] is True
    assert report["keyword_coverage"] == 1.0
    assert report["sources_count"] == 1
    assert report["uncertainties_count"] == 0
    assert report["failed_checks"] == []


def test_evaluate_case_collects_failed_checks() -> None:
    case = {
        "id": "case-2",
        "query": "Vision Pro \u9500\u91cf\u9884\u6d4b",
        "expected_intent": "industry",
        "expected_terms": ["\u9500\u91cf"],
        "min_sources": 2,
        "max_latency_ms": 1000,
        "min_keyword_coverage": 1.0,
        "require_low_uncertainty": True,
    }

    def fake_runner(query: str) -> dict:
        _ = query
        return {
            "summary": "\u673a\u6784\u9884\u6d4b\u8868\u73b0\u4e00\u822c\u3002",
            "key_points": [],
            "sources": [{"title": "A", "url": "https://example.com"}],
            "time_or_version": "unknown",
            "uncertainties": ["\u6765\u6e90\u4e0d\u8db3"],
        }

    report = evaluate_case(case, runner=fake_runner, elapsed_ms=1500.0)

    assert report["passed"] is False
    assert "sources" in report["failed_checks"]
    assert "latency" in report["failed_checks"]
    assert "keywords" in report["failed_checks"]
    assert "uncertainty" in report["failed_checks"]


def test_evaluate_case_keyword_coverage_includes_sources() -> None:
    case = {
        "id": "case-3",
        "query": "RAG chunking \u6700\u65b0\u8bba\u6587\u7efc\u8ff0",
        "expected_intent": "academic",
        "expected_terms": ["arxiv"],
        "min_sources": 1,
        "max_latency_ms": 2000,
        "min_keyword_coverage": 1.0,
        "require_low_uncertainty": False,
    }

    def fake_runner(query: str) -> dict:
        _ = query
        return {
            "summary": "Chunking strategies overview.",
            "key_points": ["Focused on retrieval reliability."],
            "sources": [{"title": "Paper list", "url": "https://arxiv.org/abs/2601.15457"}],
            "time_or_version": "unknown",
            "uncertainties": [],
        }

    report = evaluate_case(case, runner=fake_runner, elapsed_ms=500.0)

    assert report["keyword_coverage"] == 1.0
    assert report["passed"] is True


def test_summarize_reports_aggregates() -> None:
    reports = [
        {
            "passed": True,
            "elapsed_ms": 1000.0,
            "sources_count": 2,
            "uncertainties_count": 0,
            "intent_match": True,
            "keyword_coverage": 1.0,
        },
        {
            "passed": False,
            "elapsed_ms": 3000.0,
            "sources_count": 1,
            "uncertainties_count": 1,
            "intent_match": False,
            "keyword_coverage": 0.0,
        },
    ]

    summary = summarize_reports(reports)

    assert summary["total_cases"] == 2
    assert summary["passed_cases"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["avg_latency_ms"] == 2000.0
    assert summary["p95_latency_ms"] == 3000.0
    assert summary["avg_sources"] == 1.5
    assert summary["uncertainty_rate"] == 0.5
    assert summary["intent_accuracy"] == 0.5
    assert summary["avg_keyword_coverage"] == 0.5
