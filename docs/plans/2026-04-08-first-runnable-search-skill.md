# First Runnable Search Skill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first runnable Python version of the WASC search skill with rule-based query routing, async retrieval orchestration, local BM25 compression, and a single structured `MiniMax-M2.7` generation boundary.

**Architecture:** Start from a minimal Python package because the repository currently has no runnable code. Keep the first version deterministic and testable by injecting adapters and generator clients, so routing, ranking, and formatting can be verified without live network calls. Add heavier retrieval sources, MCP packaging, and advanced pruning only after the baseline flow is stable.

**Tech Stack:** Python, pytest, httpx, beautifulsoup4 or trafilatura, rank_bm25, asyncio

---

### Task 1: Bootstrap the Python project and test harness

**Files:**
- Create: `pyproject.toml`
- Create: `skill/__init__.py`
- Create: `skill/main.py`
- Create: `tests/test_smoke.py`

**Step 1: Write the failing test**

```python
from skill.main import run_query


def test_run_query_is_importable():
    assert callable(run_query)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py::test_run_query_is_importable -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'skill'`

**Step 3: Write minimal implementation**

`pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "wasc-search-skill"
version = "0.1.0"
description = "First runnable WASC search skill baseline"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`skill/main.py`
```python
def run_query(query: str):
    raise NotImplementedError("Implemented in later tasks")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py::test_run_query_is_importable -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml skill/__init__.py skill/main.py tests/test_smoke.py
git commit -m "chore: scaffold python skill baseline"
```

### Task 2: Implement query classification

**Files:**
- Create: `skill/router.py`
- Create: `tests/test_router.py`
- Modify: `skill/main.py`

**Step 1: Write the failing test**

```python
from skill.router import classify_query


def test_classify_policy_query():
    assert classify_query("2025年数据出境安全评估办法有哪些变化？") == "policy"


def test_classify_academic_query():
    assert classify_query("2025 RAG chunking 最新论文综述") == "academic"


def test_classify_industry_query():
    assert classify_query("Vision Pro 当前销量预测如何？") == "industry"


def test_classify_mixed_query():
    assert classify_query("AI Act 对开源模型和产业落地影响") == "mixed"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `skill.router`

**Step 3: Write minimal implementation**

`skill/router.py`
```python
POLICY_KEYWORDS = {"政策", "法规", "办法", "条例", "通知", "标准", "guideline", "act"}
ACADEMIC_KEYWORDS = {"论文", "paper", "arxiv", "综述", "研究", "citation", "benchmark"}
INDUSTRY_KEYWORDS = {"公司", "市场", "销量", "行业", "供应链", "融资", "产品"}


def classify_query(query: str) -> str:
    lowered = query.lower()
    has_policy = any(keyword in lowered for keyword in POLICY_KEYWORDS)
    has_academic = any(keyword in lowered for keyword in ACADEMIC_KEYWORDS)
    has_industry = any(keyword in lowered for keyword in INDUSTRY_KEYWORDS)

    matched = [has_policy, has_academic, has_industry]
    if sum(matched) > 1:
        return "mixed"
    if has_policy:
        return "policy"
    if has_academic:
        return "academic"
    return "industry"
```

`skill/main.py`
```python
from skill.router import classify_query


def run_query(query: str):
    return {"query": query, "intent": classify_query(query)}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_router.py tests/test_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add skill/router.py skill/main.py tests/test_router.py tests/test_smoke.py
git commit -m "feat: add rule-based query classification"
```

### Task 3: Define the structured output contract

**Files:**
- Create: `skill/models.py`
- Create: `skill/generator.py`
- Create: `tests/test_output_format.py`

**Step 1: Write the failing test**

```python
from skill.generator import build_result


def test_build_result_returns_required_fields():
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_format.py -v`
Expected: FAIL with `ImportError` for `skill.generator`

**Step 3: Write minimal implementation**

`skill/generator.py`
```python
def build_result(summary, key_points, sources, time_or_version, uncertainties):
    return {
        "summary": summary,
        "key_points": key_points,
        "sources": sources,
        "time_or_version": time_or_version,
        "uncertainties": uncertainties,
    }
```

`skill/models.py`
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source_type: str
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_output_format.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add skill/models.py skill/generator.py tests/test_output_format.py
git commit -m "feat: add structured output contract"
```

### Task 4: Implement text cleaning and BM25 ranking

**Files:**
- Create: `skill/cleaner.py`
- Create: `skill/ranker.py`
- Create: `tests/test_ranker.py`

**Step 1: Write the failing test**

```python
from skill.cleaner import extract_text_chunks
from skill.ranker import select_top_chunks


def test_extract_text_chunks_strips_html():
    html = "<html><body><h1>政策标题</h1><p>这是第一段。</p><p>这是第二段。</p></body></html>"
    chunks = extract_text_chunks(html)
    assert any("政策标题" in chunk for chunk in chunks)


def test_select_top_chunks_returns_relevant_chunk_first():
    chunks = ["苹果发布了新设备", "数据出境安全评估办法发布", "学术论文摘要"]
    selected = select_top_chunks("数据出境安全评估", chunks, limit=2)
    assert selected[0] == "数据出境安全评估办法发布"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ranker.py -v`
Expected: FAIL with missing module errors

**Step 3: Write minimal implementation**

`skill/cleaner.py`
```python
from bs4 import BeautifulSoup


def extract_text_chunks(html: str) -> list[str]:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return [chunk.strip() for chunk in text.split("。") if chunk.strip()]
```

`skill/ranker.py`
```python
from rank_bm25 import BM25Okapi


def select_top_chunks(query: str, chunks: list[str], limit: int = 3) -> list[str]:
    tokenized_chunks = [chunk.split() for chunk in chunks]
    tokenized_query = query.split()
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25.get_top_n(tokenized_query, chunks, n=min(limit, len(chunks)))
```

Update `pyproject.toml` dependencies:
```toml
dependencies = [
  "beautifulsoup4>=4.12",
  "rank_bm25>=0.2.2",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ranker.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml skill/cleaner.py skill/ranker.py tests/test_ranker.py
git commit -m "feat: add local text cleaning and bm25 ranking"
```

### Task 5: Implement source adapter interfaces

**Files:**
- Create: `skill/sources/__init__.py`
- Create: `skill/sources/base.py`
- Create: `skill/sources/web.py`
- Create: `skill/sources/academic.py`
- Create: `tests/test_sources.py`

**Step 1: Write the failing test**

```python
from skill.sources.web import build_web_queries
from skill.sources.academic import build_academic_queries


def test_policy_queries_add_site_filter():
    queries = build_web_queries("数据出境办法", "policy")
    assert any("site:gov.cn" in query for query in queries)


def test_academic_queries_preserve_paper_intent():
    queries = build_academic_queries("RAG chunking 论文")
    assert queries[0].startswith("RAG chunking 论文")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

`skill/sources/base.py`
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceQuery:
    provider: str
    query: str
```

`skill/sources/web.py`
```python
def build_web_queries(query: str, intent: str) -> list[str]:
    if intent == "policy":
        return [f"site:gov.cn {query}", query]
    return [query]
```

`skill/sources/academic.py`
```python
def build_academic_queries(query: str) -> list[str]:
    return [query, f"{query} arxiv", f"{query} semantic scholar"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add skill/sources/__init__.py skill/sources/base.py skill/sources/web.py skill/sources/academic.py tests/test_sources.py
git commit -m "feat: add source query adapters"
```

### Task 6: Implement async retrieval orchestration with partial-failure tolerance

**Files:**
- Create: `skill/fetcher.py`
- Create: `tests/test_fetcher.py`

**Step 1: Write the failing test**

```python
import asyncio

from skill.fetcher import gather_results


class SuccessAdapter:
    async def search(self, query: str):
        return [{"title": "ok", "url": "https://example.com", "snippet": query}]


class FailingAdapter:
    async def search(self, query: str):
        raise TimeoutError("boom")


def test_gather_results_keeps_successes_when_one_adapter_fails():
    adapters = [SuccessAdapter(), FailingAdapter()]
    results = asyncio.run(gather_results(adapters, "test"))
    assert len(results) == 1
    assert results[0]["title"] == "ok"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetcher.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

`skill/fetcher.py`
```python
import asyncio


async def _run_search(adapter, query: str):
    try:
        return await adapter.search(query)
    except Exception:
        return []


async def gather_results(adapters, query: str):
    results = await asyncio.gather(*[_run_search(adapter, query) for adapter in adapters])
    flattened = []
    for batch in results:
        flattened.extend(batch)
    return flattened
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_fetcher.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add skill/fetcher.py tests/test_fetcher.py
git commit -m "feat: add async retrieval orchestration"
```

### Task 7: Wire the end-to-end baseline flow

**Files:**
- Modify: `skill/main.py`
- Create: `tests/test_main.py`

**Step 1: Write the failing test**

```python
from skill.main import run_query


class FakeAdapter:
    async def search(self, query: str):
        return [
            {
                "title": "政策标题",
                "url": "https://gov.example/policy",
                "snippet": "数据出境安全评估办法发布",
            }
        ]


def test_run_query_returns_structured_result():
    result = run_query("数据出境安全评估办法", adapters=[FakeAdapter()])
    assert result["summary"]
    assert result["sources"][0]["title"] == "政策标题"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL because `run_query` does not build structured results yet

**Step 3: Write minimal implementation**

`skill/main.py`
```python
import asyncio

from skill.fetcher import gather_results
from skill.generator import build_result
from skill.router import classify_query


class EmptyAdapter:
    async def search(self, query: str):
        return []


def run_query(query: str, adapters=None):
    intent = classify_query(query)
    chosen_adapters = adapters or [EmptyAdapter()]
    results = asyncio.run(gather_results(chosen_adapters, query))
    top_sources = [{"title": item["title"], "url": item["url"]} for item in results[:3]]
    summary = results[0]["snippet"] if results else "来源不足"
    return build_result(
        summary=summary,
        key_points=[summary] if summary else [],
        sources=top_sources,
        time_or_version="unknown",
        uncertainties=[] if results else ["来源不足"],
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py tests/test_router.py tests/test_fetcher.py tests/test_output_format.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add skill/main.py tests/test_main.py
git commit -m "feat: wire first runnable search flow"
```

### Task 8: Add required competition-facing docs and benchmark script

**Files:**
- Create: `README.md`
- Create: `SETUP.md`
- Create: `skill/SKILL.md`
- Create: `scripts/run_shadow_benchmark.py`
- Create: `tests/test_shadow_queries.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_required_files_exist():
    assert Path("README.md").exists()
    assert Path("SETUP.md").exists()
    assert Path("skill/SKILL.md").exists()
    assert Path("scripts/run_shadow_benchmark.py").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_shadow_queries.py::test_required_files_exist -v`
Expected: FAIL because the docs and script do not exist yet

**Step 3: Write minimal implementation**

`README.md`
```md
# WASC Search Skill

First runnable benchmark-oriented search skill baseline.
```

`SETUP.md`
```md
## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```
```

`skill/SKILL.md`
```md
# SKILL

Input: natural-language search query.
Output: structured result with sources.
```

`scripts/run_shadow_benchmark.py`
```python
from skill.main import run_query

QUERIES = [
    "2025年数据出境安全评估办法有哪些变化？",
    "Vision Pro 当前销量预测如何？",
    "RAG chunking 最新论文综述",
]

for query in QUERIES:
    print(query)
    print(run_query(query))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_shadow_queries.py::test_required_files_exist -v`
Expected: PASS

**Step 5: Commit**

```bash
git add README.md SETUP.md skill/SKILL.md scripts/run_shadow_benchmark.py tests/test_shadow_queries.py
git commit -m "docs: add competition-facing baseline documentation"
```

### Task 9: Verify the full baseline locally

**Files:**
- Modify as needed based on failures from earlier tasks
- Test: `tests/test_smoke.py`
- Test: `tests/test_router.py`
- Test: `tests/test_output_format.py`
- Test: `tests/test_ranker.py`
- Test: `tests/test_sources.py`
- Test: `tests/test_fetcher.py`
- Test: `tests/test_main.py`
- Test: `tests/test_shadow_queries.py`

**Step 1: Run the full test suite**

Run: `pytest -v`
Expected: PASS

**Step 2: Run the benchmark script**

Run: `python scripts/run_shadow_benchmark.py`
Expected: 3 structured outputs printed without crashes

**Step 3: Run one manual CLI smoke check**

Run: `python -c "from skill.main import run_query; print(run_query('2025年数据出境安全评估办法有哪些变化？'))"`
Expected: dictionary with `summary`, `key_points`, `sources`, `time_or_version`, `uncertainties`

**Step 4: Fix any failing paths and rerun**

Repeat until all three checks pass.

**Step 5: Commit**

```bash
git add pyproject.toml skill tests scripts README.md SETUP.md
git commit -m "test: verify first runnable baseline end to end"
```
