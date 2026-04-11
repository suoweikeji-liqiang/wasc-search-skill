from skill.sources.base import SourceQuery


def build_academic_queries(query: str) -> list[SourceQuery]:
    return [query, f"{query} arxiv", f"{query} semantic scholar"]
