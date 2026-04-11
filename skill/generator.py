from typing import Any


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
