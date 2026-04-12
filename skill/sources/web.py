from skill.sources.base import SourceQuery

POLICY_CHANGE_TERMS = ("变化", "调整", "修订", "新增", "更新")


def _is_policy_change_query(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in POLICY_CHANGE_TERMS)


def build_web_queries(query: str, intent: str) -> list[SourceQuery]:
    if intent == "policy":
        base_query = f"site:gov.cn {query}"
        if _is_policy_change_query(query):
            return [base_query, f"{base_query} 修订 调整 变化"]
        return [base_query]
    return [query]
