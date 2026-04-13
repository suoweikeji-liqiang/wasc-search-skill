# WASC Search Skill

Search and answer-generation pipeline focused on speed, evidence quality, and stable outputs.

It combines local-first retrieval, budget-controlled query planning, evidence guardrails, ranking, and safe caching to reduce latency and token usage while producing concise, source-grounded answers across policy, industry, academic, and mixed queries.

## Positioning

This repository provides a runnable search skill pipeline, not a standalone MCP server. The main entry point is `skill.main.run_query()`, which accepts a natural-language query and returns a structured JSON-like result.

The pipeline is designed for:

- policy and regulation questions
- industry and market snapshot questions
- academic and benchmark discovery questions
- mixed questions that require cross-domain evidence

## Capabilities

- Local-first retrieval that avoids unnecessary model calls when evidence is already strong enough.
- Budget-controlled query planning that expands only into a small set of high-value subqueries.
- Evidence guardrails that reject weaker generations when they drop key entities, time/version details, or source support.
- Intent-aware ranking with source deduplication and domain preference.
- Lightweight in-process caching for repeated queries.
- Competition-style evaluation harness and regression tests for routing, ranking, fallback, and output quality.

## Repository Layout

```text
project-root/
├─ README.md
├─ SETUP.md
├─ LICENSE
├─ skill/
│  ├─ SKILL.md
│  ├─ main.py
│  ├─ router.py
│  ├─ planner.py
│  ├─ ranker.py
│  └─ sources/
├─ scripts/
│  ├─ run_competition_eval.py
│  └─ run_shadow_benchmark.py
├─ tests/
└─ ref/
   ├─ competition_eval_cases.json
   └─ competition_eval_report.json
```

## Latest Checked Eval

From `ref/competition_eval_report.json`:

- competition eval: `12/12`
- avg latency: `5268.59 ms`
- p95 latency: `8641.19 ms`
- keyword coverage: `91.67%`
- intent accuracy: `100.00%`

## Requirements

- Python `3.11+`
- `TAVILY_API_KEY` for real web-backed retrieval
- optional `MINIMAX_KEY` or `MINIMAX_API_KEY` for guarded model synthesis

Without `TAVILY_API_KEY`, the pipeline still runs but will return weaker local fallback results because no live search results are available.

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Create a `.env` file in the repository root:

```dotenv
TAVILY_API_KEY=your_tavily_key
MINIMAX_KEY=your_minimax_key
MINIMAX_MODEL=MiniMax-M2.7
MINIMAX_TIMEOUT_SECONDS=2
WASC_ENABLE_CACHE=1
```

`MINIMAX_KEY` is optional. If it is not set, the pipeline stays local-first and skips model synthesis.

## Run A Single Query

Example:

```bash
python -c "from skill.main import run_query; import json; result = run_query('个人信息出境认证办法 2025 年修订了哪些条款？'); print(json.dumps(result, ensure_ascii=False, indent=2))"
```

Other example queries:

- `Vision Pro 当前销量预测如何？`
- `retrieval reranking recent benchmark papers`
- `AI Act 对开源模型和产业落地影响`

## Output Contract

The pipeline returns a structured result with fixed keys:

```json
{
  "summary": "short grounded answer",
  "key_points": [
    "supporting point 1",
    "supporting point 2"
  ],
  "sources": [
    {
      "title": "source title",
      "url": "https://example.com"
    }
  ],
  "time_or_version": "2025",
  "uncertainties": [
    "what remains uncertain"
  ]
}
```

## Run Tests

```bash
pytest -q
```

## Reproduce The Evaluation

Run the full evaluation set:

```bash
python scripts/run_competition_eval.py --dataset ref/competition_eval_cases.json --output ref/competition_eval_report.json
```

Quick smoke run on the first 3 cases:

```bash
python scripts/run_competition_eval.py --max-cases 3 --print-cases
```

Evaluation inputs are stored in `ref/competition_eval_cases.json`, and a checked report is included in `ref/competition_eval_report.json`.

## Notes On MCP Integration

This repository does not expose an MCP server or tool schema by itself. If you want to use it inside an MCP host, wrap `run_query()` behind your own server entrypoint and tool contract.
