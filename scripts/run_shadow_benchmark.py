import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skill.main import run_query

QUERIES: list[str] = [
    "2025年数据出境安全评估办法有哪些变化？",
    "Vision Pro 当前销量预测如何？",
    "RAG chunking 最新论文综述",
]


def main() -> None:
    for query in QUERIES:
        print(query)
        print(run_query(query))


if __name__ == "__main__":
    main()
