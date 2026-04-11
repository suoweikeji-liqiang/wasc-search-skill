from skill.cleaner import extract_text_chunks
from skill.ranker import select_top_chunks


def test_extract_text_chunks_strips_html() -> None:
    html = "<html><body><h1>政策标题</h1><p>这是第一段。</p><p>这是第二段。</p></body></html>"
    chunks = extract_text_chunks(html)
    assert any("政策标题" in chunk for chunk in chunks)


def test_select_top_chunks_returns_relevant_chunk_first() -> None:
    chunks = ["苹果发布了新设备", "数据出境安全评估办法发布", "学术论文摘要"]
    selected = select_top_chunks("数据出境安全评估", chunks, limit=2)
    assert selected[0] == "数据出境安全评估办法发布"


def test_select_top_chunks_returns_empty_for_empty_chunks() -> None:
    selected = select_top_chunks("任意查询", [], limit=3)
    assert selected == []


def test_select_top_chunks_returns_empty_when_limit_is_non_positive() -> None:
    chunks = ["苹果发布了新设备", "数据出境安全评估办法发布"]
    selected_zero = select_top_chunks("数据出境安全评估", chunks, limit=0)
    selected_negative = select_top_chunks("数据出境安全评估", chunks, limit=-1)
    assert selected_zero == []
    assert selected_negative == []


def test_select_top_chunks_returns_empty_for_blank_query() -> None:
    chunks = ["苹果发布了新设备", "数据出境安全评估办法发布"]
    selected_empty = select_top_chunks("", chunks, limit=2)
    selected_whitespace = select_top_chunks("   ", chunks, limit=2)
    assert selected_empty == []
    assert selected_whitespace == []


def test_select_top_chunks_returns_empty_for_whitespace_only_chunks() -> None:
    chunks = ["   ", "", "  "]
    selected = select_top_chunks("数据出境安全评估", chunks, limit=2)
    assert selected == []
