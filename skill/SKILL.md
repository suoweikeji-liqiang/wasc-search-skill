# SKILL

## Name

WASC Search Skill

## Purpose

Answer search-oriented natural-language queries with concise, source-grounded structured output.

## Input

- one natural-language query string

## Output

- `summary`: short grounded answer
- `key_points`: concise supporting points
- `sources`: top supporting sources with title and URL
- `time_or_version`: extracted date, year, or version when available
- `uncertainties`: explicit gaps or limits in the evidence

## Runtime Behavior

- local-first retrieval and synthesis
- intent-aware query planning for policy, industry, academic, and mixed questions
- ranking and source deduplication
- guarded optional MiniMax synthesis when local evidence is insufficient
- lightweight cache for repeated queries

## Environment

- requires `TAVILY_API_KEY` for live retrieval
- optionally uses `MINIMAX_KEY` or `MINIMAX_API_KEY` for model synthesis

## Main Entry Point

- `skill.main.run_query(query: str) -> dict`
