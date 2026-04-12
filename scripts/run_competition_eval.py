import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skill.evaluation import evaluate_case, load_eval_cases, summarize_reports


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run competition-style evaluation on query set")
    parser.add_argument(
        "--dataset",
        default="ref/competition_eval_cases.json",
        help="Path to evaluation dataset json file",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path for json report",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Evaluate only first N cases (0 means all)",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Exit with code 1 when pass_rate is below this threshold",
    )
    parser.add_argument(
        "--print-cases",
        action="store_true",
        help="Print each case result",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    cases = load_eval_cases(dataset_path)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    reports = [evaluate_case(case) for case in cases]
    summary = summarize_reports(reports)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "summary": summary,
        "cases": reports,
    }

    print(f"Dataset: {dataset_path}")
    print(f"Cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed_cases']} ({summary['pass_rate']:.2%})")
    print(f"Avg latency: {summary['avg_latency_ms']} ms")
    print(f"P95 latency: {summary['p95_latency_ms']} ms")
    print(f"Avg sources: {summary['avg_sources']}")
    print(f"Uncertainty rate: {summary['uncertainty_rate']:.2%}")
    if summary["intent_accuracy"] is not None:
        print(f"Intent accuracy: {summary['intent_accuracy']:.2%}")
    if summary["avg_keyword_coverage"] is not None:
        print(f"Keyword coverage: {summary['avg_keyword_coverage']:.2%}")

    if args.print_cases:
        for case_report in reports:
            status = "PASS" if case_report["passed"] else "FAIL"
            print(
                f"[{status}] {case_report['id']} {case_report['elapsed_ms']}ms "
                f"checks={case_report['failed_checks']} query={case_report['query']}"
            )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Report saved to: {output_path}")

    return 1 if summary["pass_rate"] < args.min_pass_rate else 0


if __name__ == "__main__":
    raise SystemExit(main())
