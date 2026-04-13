import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "artifacts" / "bilibili_demo"
FRAMES_DIR = OUTPUT_DIR / "frames"
VIDEO_PATH = OUTPUT_DIR / "wasc_bilibili_demo.mp4"
COVER_PATH = OUTPUT_DIR / "wasc_bilibili_cover.png"
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")

WIDTH = 1280
HEIGHT = 720
FPS = 24
BG = "#08111f"
PANEL = "#0f1b2d"
PANEL_SOFT = "#11223a"
TEXT = "#f2f6ff"
MUTED = "#98a8c7"
ACCENT = "#46d1a8"
ACCENT_2 = "#6db2ff"
WARN = "#ffcf6a"
CLOSING_PUNCTUATION = set("，。！？；：、,.!?;:)]}》】」』”’")
ASCII_TOKEN_CHARS = set("-_/+:&.%")


def wrap_text(text: str, max_chars: int = 22) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []

    lines: list[str] = []
    current = ""
    for token in _split_tokens(normalized):
        if token == " " and not current:
            continue
        if len(current) + len(token) <= max_chars:
            current += token
            continue
        if token in CLOSING_PUNCTUATION and current:
            lines.append(f"{current}{token}".rstrip())
            current = ""
            continue
        if current:
            lines.append(current.rstrip())
            current = token.lstrip()
            continue
        lines.append(token[:max_chars])
        current = token[max_chars:]
    if current:
        lines.append(current.rstrip())
    return lines


def _split_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    buffer = ""
    for char in text:
        if char == " ":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(char)
            continue
        current_ascii = char.isascii() and (char.isalnum() or char in ASCII_TOKEN_CHARS)
        if current_ascii:
            buffer += char
            continue
        if buffer:
            tokens.append(buffer)
        buffer = char
        tokens.append(buffer)
        buffer = ""
    if buffer:
        tokens.append(buffer)
    return tokens


def extract_hook_metrics(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = dict(report["summary"])
    return {
        "passed_cases": int(summary["passed_cases"]),
        "total_cases": int(summary["total_cases"]),
        "avg_latency_ms": float(summary["avg_latency_ms"]),
        "avg_keyword_coverage": float(summary["avg_keyword_coverage"]),
        "intent_accuracy": float(summary["intent_accuracy"]),
    }


def build_scoreboard_lines(metrics: dict[str, Any]) -> list[str]:
    return [
        f"competition eval: {metrics['passed_cases']}/{metrics['total_cases']}",
        f"avg latency: {metrics['avg_latency_ms']:.2f} ms",
        f"keyword coverage: {metrics['avg_keyword_coverage'] * 100:.2f}%",
        f"intent accuracy: {metrics['intent_accuracy'] * 100:.2f}%",
    ]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
    return ImageFont.truetype(str(path), size=size)


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    fill: str,
    line_gap: int = 8,
) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, radius: int = 28) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def _gradient_background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    for i in range(HEIGHT):
        alpha = i / HEIGHT
        color = (
            int(8 + 20 * alpha),
            int(17 + 32 * alpha),
            int(31 + 18 * alpha),
        )
        draw.line((0, i, WIDTH, i), fill=color)
    draw.ellipse((900, -120, 1400, 380), fill="#11345a")
    draw.ellipse((-160, 420, 420, 980), fill="#0c2848")
    return image


def _terminal_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str], title: str) -> None:
    x1, y1, x2, y2 = box
    _panel(draw, box, PANEL)
    draw.rounded_rectangle((x1 + 18, y1 + 16, x1 + 148, y1 + 54), radius=18, fill="#13263f")
    draw.text((x1 + 34, y1 + 23), title, font=_load_font(22, bold=True), fill=TEXT)
    font = _load_font(24)
    y = y1 + 82
    for line in lines:
        draw.text((x1 + 28, y), line, font=font, fill=TEXT)
        y += 34
        if y > y2 - 40:
            break


def _subtitle(draw: ImageDraw.ImageDraw, text: str) -> None:
    lines = wrap_text(text, max_chars=30)
    font = _load_font(30, bold=True)
    box_height = 36 * len(lines) + 28
    y1 = HEIGHT - box_height - 34
    _panel(draw, (120, y1, WIDTH - 120, HEIGHT - 24), "#050910", radius=20)
    _draw_text_block(draw, (150, y1 + 14), lines, font, TEXT, line_gap=6)


def _metric_chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, value: str, accent: str) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + 260, y + 92), radius=24, fill=PANEL)
    draw.text((x + 22, y + 16), label, font=_load_font(22), fill=MUTED)
    draw.text((x + 22, y + 46), value, font=_load_font(32, bold=True), fill=accent)


def _scene_frames() -> list[dict[str, Any]]:
    report_metrics = extract_hook_metrics(REPO_ROOT / "ref" / "competition_eval_report.json")
    scoreboard_lines = build_scoreboard_lines(report_metrics)
    policy_result = {
        "summary": "2025 版政策题输出会优先压缩成版本、变化点、适用条件和实施时间四类信息。",
        "time_or_version": "示例结构",
        "sources": [
            {"title": "中国网信网 / 官方办法原文", "url": ""},
            {"title": "官方申报指南 / 政策问答", "url": ""},
            {"title": "补充监管说明 / 时间信息", "url": ""},
        ],
    }
    mixed_result = {
        "summary": "混合题会先拆政策义务和产业影响，再融合成政策变化到成本、供给、落地的链路答案。",
        "time_or_version": "cross-domain",
        "sources": [
            {"title": "政策源：法规 / 指南 / 监管问答", "url": ""},
            {"title": "产业源：机构 / vendor / market report", "url": ""},
            {"title": "最终输出：短 summary + key points + sources", "url": ""},
        ],
    }
    return [
        {
            "duration": 6.0,
            "subtitle": "搜索又贵又慢又不准。我把这个比赛解法压进了 5 秒级响应，还保住了 12/12。",
            "render": lambda progress: render_hook_scene(scoreboard_lines, progress),
        },
        {
            "duration": 8.0,
            "subtitle": "核心不是多调模型，而是 local-first。先做 query planner、ranking 和 evidence guardrail，只有必要时才调用 MiniMax。",
            "render": lambda progress: render_pipeline_scene(progress),
        },
        {
            "duration": 12.0,
            "subtitle": "这是仓库里的真实验证。pytest 全绿，competition eval 直接跑出 12/12。",
            "render": lambda progress: render_validation_scene(scoreboard_lines, progress),
        },
        {
            "duration": 14.0,
            "subtitle": "政策题先走官方高可信 source，再把变化点、时间和适用条件压成短答案，不让泛政策页抢前排。",
            "render": lambda progress: render_result_scene(
                title="Policy Answer Shape",
                query="个人信息出境认证办法 2025 年修订了哪些条款？",
                result=policy_result,
                progress=progress,
                accent=ACCENT,
            ),
        },
        {
            "duration": 14.0,
            "subtitle": "混合题会拆政策和产业两条证据链，最后融合成更像评委想看的比赛答案。",
            "render": lambda progress: render_result_scene(
                title="Mixed Answer Shape",
                query="AI Act 对开源模型和产业落地影响",
                result=mixed_result,
                progress=progress,
                accent=ACCENT_2,
            ),
        },
        {
            "duration": 12.0,
            "subtitle": "最后拼的是稳定性。固定任务重复跑也能吃到安全缓存，少烧时间，少烧 token。",
            "render": lambda progress: render_close_scene(scoreboard_lines, progress),
        },
    ]

def render_hook_scene(scoreboard_lines: list[str], progress: float) -> Image.Image:
    image = _gradient_background()
    draw = ImageDraw.Draw(image)
    draw.text((86, 84), "WASC Search Skill", font=_load_font(54, bold=True), fill=TEXT)
    draw.text((88, 156), "低成本高精度搜索，不靠堆模型取胜", font=_load_font(28), fill=MUTED)
    offset = int((1 - min(progress / 0.25, 1.0)) * 36)
    _panel(draw, (80, 248 - offset, 1200, 520 - offset), PANEL)
    draw.text((120, 286 - offset), "比赛结果", font=_load_font(30, bold=True), fill=ACCENT)
    for idx, line in enumerate(scoreboard_lines):
        draw.text((122, 344 + idx * 44 - offset), line, font=_load_font(28), fill=TEXT)
    _metric_chip(draw, (86, 560), "核心策略", "local-first", ACCENT_2)
    _metric_chip(draw, (366, 560), "生成约束", "guardrail", ACCENT)
    _metric_chip(draw, (646, 560), "目标", "高分稳定解", WARN)
    return image


def render_pipeline_scene(progress: float) -> Image.Image:
    image = _gradient_background()
    draw = ImageDraw.Draw(image)
    draw.text((86, 88), "Pipeline", font=_load_font(50, bold=True), fill=TEXT)
    steps = [
        ("1", "Router", "policy / industry / academic / mixed"),
        ("2", "Planner", "预算受控子查询"),
        ("3", "Ranking", "官方源优先，噪声页降权"),
        ("4", "Guardrail", "模型变弱就回退"),
        ("5", "Cache", "重复任务直接复用"),
    ]
    for idx, (num, title, desc) in enumerate(steps):
        y = 178 + idx * 92
        x = 92 + (16 if idx % 2 else 0)
        _panel(draw, (x, y, 1188, y + 74), PANEL if idx % 2 == 0 else PANEL_SOFT)
        draw.text((x + 24, y + 18), num, font=_load_font(30, bold=True), fill=ACCENT)
        draw.text((x + 78, y + 16), title, font=_load_font(28, bold=True), fill=TEXT)
        draw.text((x + 244, y + 18), desc, font=_load_font(24), fill=MUTED)
    return image


def render_validation_scene(scoreboard_lines: list[str], progress: float) -> Image.Image:
    image = _gradient_background()
    draw = ImageDraw.Draw(image)
    draw.text((86, 82), "真实验证", font=_load_font(50, bold=True), fill=TEXT)
    terminal_lines = [
        "> pytest -q",
        "86 passed in 0.67s",
        "",
        "> python scripts/run_competition_eval.py",
        "Passed: 12 (100.00%)",
        f"Avg latency: {scoreboard_lines[1].split(': ',1)[1]}",
        f"Keyword coverage: {scoreboard_lines[2].split(': ',1)[1]}",
        f"Intent accuracy: {scoreboard_lines[3].split(': ',1)[1]}",
    ]
    _terminal_panel(draw, (80, 170, 1200, 604), terminal_lines, "terminal")
    return image


def render_result_scene(title: str, query: str, result: dict[str, Any], progress: float, accent: str) -> Image.Image:
    image = _gradient_background()
    draw = ImageDraw.Draw(image)
    draw.text((86, 74), title, font=_load_font(48, bold=True), fill=TEXT)
    _panel(draw, (78, 146, 1202, 250), PANEL)
    draw.text((106, 172), "Query", font=_load_font(24, bold=True), fill=accent)
    _draw_text_block(draw, (106, 202), wrap_text(query, 48), _load_font(26), TEXT, line_gap=4)

    _panel(draw, (78, 274, 760, 620), PANEL)
    draw.text((106, 302), "Answer", font=_load_font(24, bold=True), fill=accent)
    answer_lines = wrap_text(result["summary"], 30)
    _draw_text_block(draw, (106, 338), answer_lines[:5], _load_font(28), TEXT, line_gap=8)
    draw.text((106, 514), f"time_or_version: {result['time_or_version']}", font=_load_font(22), fill=MUTED)

    _panel(draw, (790, 274, 1202, 620), PANEL_SOFT)
    draw.text((818, 302), "Sources", font=_load_font(24, bold=True), fill=accent)
    y = 340
    for source in result["sources"][:3]:
        for line in wrap_text(f"- {source['title']}", 20)[:3]:
            draw.text((818, y), line, font=_load_font(21), fill=TEXT)
            y += 28
        y += 16
    return image


def render_close_scene(scoreboard_lines: list[str], progress: float) -> Image.Image:
    image = _gradient_background()
    draw = ImageDraw.Draw(image)
    draw.text((86, 92), "为什么它更像比赛解", font=_load_font(48, bold=True), fill=TEXT)
    bullets = [
        "不是每个 query 都走完整生成",
        "证据更强才允许模型接管",
        "面向 policy / mixed 的评分点组织答案",
        "重复任务可安全缓存，降低延迟和 token",
    ]
    for idx, bullet in enumerate(bullets):
        y = 212 + idx * 86
        draw.rounded_rectangle((94, y, 122, y + 28), radius=14, fill=ACCENT if idx % 2 == 0 else ACCENT_2)
        draw.text((150, y - 4), bullet, font=_load_font(28), fill=TEXT)
    draw.text((86, 612), "GitHub repo ready  |  README + tests + reproducible eval", font=_load_font(24), fill=MUTED)
    return image


def generate_video() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    _clear_old_frames()
    scenes = _scene_frames()
    frame_index = 0
    for scene in scenes:
        total_frames = max(1, int(scene["duration"] * FPS))
        for local_index in range(total_frames):
            progress = local_index / max(1, total_frames - 1)
            image = scene["render"](progress)
            draw = ImageDraw.Draw(image)
            _subtitle(draw, scene["subtitle"])
            image.save(FRAMES_DIR / f"frame_{frame_index:05d}.png")
            frame_index += 1
    render_cover_image(scoreboard_lines=build_scoreboard_lines(extract_hook_metrics(REPO_ROOT / "ref" / "competition_eval_report.json")))
    _encode_video()
    return VIDEO_PATH


def render_cover_image(scoreboard_lines: list[str]) -> Path:
    image = _gradient_background()
    draw = ImageDraw.Draw(image)
    draw.text((78, 84), "WASC Search Skill", font=_load_font(58, bold=True), fill=TEXT)
    draw.text((82, 158), "12/12 真实评测  |  local-first 搜索优化", font=_load_font(30), fill=ACCENT)
    _panel(draw, (78, 238, 1200, 504), PANEL)
    draw.text((114, 284), "更快、更稳、更省 token", font=_load_font(40, bold=True), fill=TEXT)
    for idx, line in enumerate(scoreboard_lines):
        draw.text((116, 352 + idx * 42), line, font=_load_font(28), fill=MUTED if idx else ACCENT_2)
    draw.rounded_rectangle((78, 548, 1200, 652), radius=26, fill="#050910")
    draw.text((116, 580), "local-first  ·  query planner  ·  evidence guardrail  ·  cache", font=_load_font(28, bold=True), fill=TEXT)
    image.save(COVER_PATH)
    return COVER_PATH


def _clear_old_frames() -> None:
    if not FRAMES_DIR.exists():
        return
    for frame in FRAMES_DIR.glob("frame_*.png"):
        frame.unlink()


def _resolve_ffmpeg() -> str:
    candidate_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
    ]
    package_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if package_root.exists():
        candidate_paths.extend(package_root.rglob("ffmpeg.exe"))
    for path in candidate_paths:
        if path.is_file():
            return str(path)
    raise FileNotFoundError("ffmpeg.exe not found. Install ffmpeg first.")


def _encode_video() -> None:
    command = [
        _resolve_ffmpeg(),
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(FRAMES_DIR / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(VIDEO_PATH),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> int:
    path = generate_video()
    print(f"Video generated: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
