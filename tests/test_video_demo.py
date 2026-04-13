from pathlib import Path

from scripts.generate_bilibili_demo import (
    _scene_frames,
    build_scoreboard_lines,
    extract_hook_metrics,
    wrap_text,
)


def test_extract_hook_metrics_reads_competition_summary() -> None:
    report_path = Path("ref/competition_eval_report.json")
    metrics = extract_hook_metrics(report_path)

    assert metrics["passed_cases"] == 12
    assert metrics["total_cases"] == 12
    assert metrics["avg_latency_ms"] > 0


def test_build_scoreboard_lines_formats_core_metrics() -> None:
    lines = build_scoreboard_lines(
        {
            "passed_cases": 12,
            "total_cases": 12,
            "avg_latency_ms": 4264.01,
            "avg_keyword_coverage": 0.9583,
            "intent_accuracy": 1.0,
        }
    )

    merged = " ".join(lines)
    assert "12/12" in merged
    assert "4264.01 ms" in merged
    assert "95.83%" in merged
    assert "100.00%" in merged


def test_wrap_text_breaks_long_lines_without_losing_content() -> None:
    text = "这是一个很长的视频字幕句子，需要被拆成多行，但不能丢掉关键词 local-first 和 guardrail。"
    lines = wrap_text(text, max_chars=18)

    assert len(lines) >= 2
    assert "local-first" in "".join(lines)
    assert "guardrail" in "".join(lines)


def test_wrap_text_keeps_mixed_prefix_and_punctuation_attached() -> None:
    text = (
        "2025 版政策题输出会优先压缩成版本、变化点、适用条件和"
        "实施时间四类信息。"
    )

    lines = wrap_text(text, max_chars=28)

    assert lines[0] != "2025 "
    assert "。" not in lines
    assert not any(line.startswith("、") or line.startswith("，") for line in lines)


def test_video_scenes_keep_competition_language_in_opening_only() -> None:
    scenes = _scene_frames()

    assert "WASC" in scenes[0]["subtitle"] or "比赛" in scenes[0]["subtitle"]
    assert all("评委" not in scene["subtitle"] for scene in scenes)
    assert all("比赛答案" not in scene["subtitle"] for scene in scenes[1:])
    assert all("比赛解" not in scene["subtitle"] for scene in scenes[1:])


def test_scoreboard_lines_use_general_metrics_language() -> None:
    lines = build_scoreboard_lines(
        {
            "passed_cases": 12,
            "total_cases": 12,
            "avg_latency_ms": 4264.01,
            "avg_keyword_coverage": 0.9583,
            "intent_accuracy": 1.0,
        }
    )

    assert lines[0].startswith("verified cases:")
    assert "competition" not in " ".join(lines).lower()
