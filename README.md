# WASC Search Skill

Search and answer-generation pipeline focused on speed, evidence quality, and stable outputs.

It combines local-first retrieval, budget-controlled query planning, evidence guardrails, ranking, and safe caching to reduce latency and token usage while producing concise, source-grounded answers across policy, industry, academic, and mixed queries.

## Highlights

- Local-first pipeline that avoids unnecessary model calls when evidence is already strong enough.
- Budget-controlled query planner that expands only into a small set of high-value subqueries.
- Evidence guardrails that reject weaker generations when they drop key entities, time/version details, or source support.
- Competition-style evaluation harness and regression tests for routing, ranking, fallback, and output quality.

## Latest Checked Eval

From `ref/competition_eval_report.json`:

- competition eval: `12/12`
- avg latency: `4963.96 ms`
- keyword coverage: `91.67%`
- intent accuracy: `100.00%`

## Competition Eval

Run the competition-style evaluation set:

```bash
python scripts/run_competition_eval.py --dataset ref/competition_eval_cases.json --output ref/competition_eval_report.json
```

Quick smoke run on the first 3 cases:

```bash
python scripts/run_competition_eval.py --max-cases 3 --print-cases
```

## Demo Video

Generate the Bilibili demo video and cover image:

```bash
python scripts/generate_bilibili_demo.py
```

Outputs:

- `artifacts/bilibili_demo/wasc_bilibili_demo.mp4`
- `artifacts/bilibili_demo/wasc_bilibili_cover.png`
