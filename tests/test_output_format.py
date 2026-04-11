from skill.generator import build_result


def test_build_result_returns_required_fields() -> None:
    result = build_result(
        summary="结论",
        key_points=["点1"],
        sources=[{"title": "A", "url": "https://example.com"}],
        time_or_version="2025",
        uncertainties=["待确认"],
    )

    assert set(result.keys()) == {
        "summary",
        "key_points",
        "sources",
        "time_or_version",
        "uncertainties",
    }
