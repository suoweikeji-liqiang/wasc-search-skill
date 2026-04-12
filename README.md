# WASC Search Skill

First runnable benchmark-oriented search skill baseline.

## Competition Eval

Run the competition-style evaluation set:

```bash
python scripts/run_competition_eval.py --dataset ref/competition_eval_cases.json --output ref/competition_eval_report.json
```

Quick smoke run on first 3 cases:

```bash
python scripts/run_competition_eval.py --max-cases 3 --print-cases
```
