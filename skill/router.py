import re
from typing import Literal

POLICY_KEYWORDS: set[str] = {"政策", "法规", "办法", "条例", "通知", "标准", "guideline", "act"}
ACADEMIC_KEYWORDS: set[str] = {"论文", "paper", "arxiv", "综述", "研究", "citation", "benchmark"}
INDUSTRY_KEYWORDS: set[str] = {"公司", "市场", "销量", "行业", "产业", "供应链", "融资", "产品"}

Intent = Literal["policy", "academic", "industry", "mixed"]


def _matches_keyword(query: str, keyword: str) -> bool:
    if keyword.isascii() and keyword.isalpha():
        return re.search(rf"\b{re.escape(keyword)}\b", query) is not None
    return keyword in query


def classify_query(query: str) -> Intent:
    lowered: str = query.lower()
    has_policy: bool = any(_matches_keyword(lowered, keyword) for keyword in POLICY_KEYWORDS)
    has_academic: bool = any(_matches_keyword(lowered, keyword) for keyword in ACADEMIC_KEYWORDS)
    has_industry: bool = any(_matches_keyword(lowered, keyword) for keyword in INDUSTRY_KEYWORDS)

    matched: list[bool] = [has_policy, has_academic, has_industry]
    if sum(matched) > 1:
        return "mixed"
    if has_policy:
        return "policy"
    if has_academic:
        return "academic"
    if has_industry:
        return "industry"
    return "mixed"
