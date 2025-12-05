from services.query_service import (
    execute_code_example_query,
    execute_document_query,
    rerank_results,
)


def test_rerank_results_applies_scores(monkeypatch):
    calls = {}

    def fake_rerank(query, docs, top_n):
        calls["payload"] = (query, docs, top_n)
        return [{"index": 1, "score": 0.9}]

    results = [
        {"content": "a", "similarity": 0.1},
        {"content": "b", "similarity": 0.2},
    ]
    reranked = rerank_results("q", results, rerank_fn=fake_rerank)
    assert reranked[0]["content"] == "b"
    assert calls["payload"][0] == "q"


def test_execute_document_query_hybrid_combines_results():
    vector_results = [
        {"id": 1, "url": "https://a", "content": "foo", "metadata": {}, "similarity": 0.4},
        {"id": 2, "url": "https://b", "content": "bar", "metadata": {}, "similarity": 0.3},
    ]
    keyword_results = [
        {"id": 1, "url": "https://a", "chunk_number": 0, "content": "foo", "metadata": {}, "source_id": "s"},
        {"id": 3, "url": "https://c", "chunk_number": 0, "content": "baz", "metadata": {}, "source_id": "s"},
    ]

    payload = execute_document_query(
        supabase_client=None,
        query="foo",
        source=None,
        match_count=3,
        use_hybrid_search=True,
        use_reranking=False,
        reranking_available=False,
        vector_search_fn=lambda **_: vector_results,
        keyword_search_fn=lambda *args, **kwargs: keyword_results,
    )

    assert payload["count"] == 3
    urls = {item["url"] for item in payload["results"]}
    assert "https://a" in urls and "https://c" in urls


def test_execute_code_example_query_handles_vector_only():
    vector_results = [
        {
            "id": 1,
            "url": "https://a",
            "content": "code",
            "summary": "desc",
            "metadata": {},
            "source_id": "s",
            "similarity": 0.8,
        }
    ]

    payload = execute_code_example_query(
        supabase_client=None,
        query="abc",
        source_id=None,
        match_count=1,
        use_hybrid_search=False,
        use_reranking=False,
        reranking_available=False,
        vector_search_fn=lambda **_: vector_results,
    )

    assert payload["results"][0]["code"] == "code"
