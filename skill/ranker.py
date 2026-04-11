from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    if " " in text:
        return text.split()
    return list(text)


def select_top_chunks(query: str, chunks: list[str], limit: int = 3) -> list[str]:
    if not chunks or limit <= 0 or not query.strip():
        return []

    chunk_tokens = [
        (chunk, _tokenize(chunk))
        for chunk in chunks
    ]
    non_empty_chunk_tokens = [
        (chunk, tokens)
        for chunk, tokens in chunk_tokens
        if tokens
    ]
    if not non_empty_chunk_tokens:
        return []

    filtered_chunks = [chunk for chunk, _ in non_empty_chunk_tokens]
    tokenized_chunks = [tokens for _, tokens in non_empty_chunk_tokens]
    tokenized_query = _tokenize(query)
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25.get_top_n(
        tokenized_query,
        filtered_chunks,
        n=min(limit, len(filtered_chunks)),
    )

