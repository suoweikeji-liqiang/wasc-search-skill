from skill.sources.base import SourceQuery


def build_web_queries(query: str, intent: str) -> list[SourceQuery]:
    if intent == "policy":
        return [f"site:gov.cn {query}", query]
    return [query]
