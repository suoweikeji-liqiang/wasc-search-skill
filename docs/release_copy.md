# Release Copy

## GitHub

### Repository Name

`wasc-search-skill`

### Tagline

Fast, source-grounded search and answer pipeline with local-first retrieval and stable outputs.

### Short Description

WASC Search Skill is a search and answer-generation pipeline focused on speed, evidence quality, and stable outputs. It combines local-first retrieval, budget-controlled query planning, evidence guardrails, ranking, and safe caching to reduce latency, lower token usage, and produce concise, source-grounded answers across policy, industry, academic, and mixed queries.

### Social Preview Line

Faster, steadier, and more source-grounded search answers without turning every query into a full model generation.

## Bilibili

### Recommended Title

我把搜索问答压进了 5 秒级响应，还保住了 12/12

### Alternate Titles

- local-first 搜索优化实战：更快、更稳、更省 token
- 不靠堆模型，怎么把搜索问答做得更快更稳
- 一个更像工程解的搜索问答系统：速度、证据和稳定性一起抓

### Description

这条视频展示了一个搜索问答管线的实机演示。

核心不是“每题都调大模型硬写”，而是先做 local-first 检索、预算受控 query planner、evidence guardrail、排序和安全缓存，在保证答案更短、更稳、更可追溯的同时，把延迟和 token 消耗压下来。

仓库地址：
https://github.com/suoweikeji-liqiang/wasc-search-skill

主要结果：
- competition eval: 12/12
- avg latency: 4963.96 ms
- keyword coverage: 91.67%
- intent accuracy: 100.00%

### Tags

- AI
- 搜索
- RAG
- 信息检索
- 大模型应用
- 工程优化
- Python
- GitHub
- 开源

### Pinned Comment

仓库我放出来了：
https://github.com/suoweikeji-liqiang/wasc-search-skill

这条视频主要讲 4 个点：local-first、query planner、evidence guardrail、cache。
如果你更关心可复现结果，可以直接看 README、评测脚本和 `ref/competition_eval_report.json`。

### Cover Text

- 更快、更稳、更省 token
- 5 秒级响应
- 12/12 实测
