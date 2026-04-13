## Setup

### Prerequisites

- Python `3.11+`
- network access for Tavily-backed retrieval
- `TAVILY_API_KEY` for live search
- optional `MINIMAX_KEY` or `MINIMAX_API_KEY` for guarded synthesis

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Environment Variables

Create `.env` in the repository root:

```dotenv
TAVILY_API_KEY=your_tavily_key
MINIMAX_KEY=your_minimax_key
MINIMAX_MODEL=MiniMax-M2.7
MINIMAX_BASE_URL=https://api.minimaxi.com/v1
MINIMAX_TIMEOUT_SECONDS=2
WASC_ENABLE_CACHE=1
```

Notes:

- `TAVILY_API_KEY` is required for real search results.
- `MINIMAX_KEY` is optional. If omitted, the pipeline still runs and returns local-first results only.
- `MINIMAX_API_KEY` is also accepted as an alternative to `MINIMAX_KEY`.
- `WASC_ENABLE_CACHE=0` disables the in-process cache.

### Run A Single Query

```bash
python -c "from skill.main import run_query; import json; result = run_query('个人信息出境认证办法 2025 年修订了哪些条款？'); print(json.dumps(result, ensure_ascii=False, indent=2))"
```

### Run Tests

```bash
pytest -q
```

### Reproduce Evaluation

Full evaluation:

```bash
python scripts/run_competition_eval.py --dataset ref/competition_eval_cases.json --output ref/competition_eval_report.json
```

Quick smoke run:

```bash
python scripts/run_competition_eval.py --max-cases 3 --print-cases
```

### Test Samples And Reproduction Inputs

- evaluation dataset: `ref/competition_eval_cases.json`
- checked report: `ref/competition_eval_report.json`
- regression tests: `tests/`
- example query categories:
  - policy: `个人信息出境认证办法 2025 年修订了哪些条款？`
  - industry: `Vision Pro 当前销量预测如何？`
  - academic: `retrieval reranking recent benchmark papers`
  - mixed: `AI Act 对开源模型和产业落地影响`

### MCP Integration Status

This repository does not include a standalone MCP server implementation. It provides the core skill pipeline that can be wrapped by an external MCP host if needed.
