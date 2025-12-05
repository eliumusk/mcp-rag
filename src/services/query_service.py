"""Helper functions shared by query-focused MCP tools."""
from typing import Any, Callable, Dict, Iterable, List, Optional

from utils import jina_rerank_documents, search_code_examples, search_documents


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    *,
    content_key: str = "content",
    rerank_fn: Callable[[str, List[str], int], Optional[List[Dict[str, Any]]]] = jina_rerank_documents,
) -> List[Dict[str, Any]]:
    """Apply Jina reranking to a list of semantic results."""

    if not results:
        return results

    try:
        documents = [result.get(content_key, "") or "" for result in results]
        rerank_data = rerank_fn(query, documents, top_n=len(documents)) or []
        score_map = {item.get("index"): item.get("score", 0.0) for item in rerank_data if "index" in item}
        for idx, result in enumerate(results):
            if idx in score_map:
                result["rerank_score"] = float(score_map[idx])
        return sorted(results, key=lambda item: item.get("rerank_score", 0.0), reverse=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Error during reranking: {exc}")
        return results


def _combine_semantic_and_keyword_results(
    vector_results: Iterable[Dict[str, Any]],
    keyword_results: Iterable[Dict[str, Any]],
    match_count: int,
    convert_keyword_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge semantic and keyword matches, preferring overlap."""

    seen_ids = set()
    combined_results: List[Dict[str, Any]] = []

    vector_list = list(vector_results)
    vector_ids = {result.get("id") for result in vector_list if result.get("id")}

    for keyword_item in keyword_results:
        keyword_id = keyword_item.get("id")
        if keyword_id in vector_ids and keyword_id not in seen_ids:
            for vector_item in vector_list:
                if vector_item.get("id") == keyword_id:
                    vector_item["similarity"] = min(1.0, vector_item.get("similarity", 0) * 1.2)
                    combined_results.append(vector_item)
                    seen_ids.add(keyword_id)
                    break

    for vector_item in vector_list:
        vector_id = vector_item.get("id")
        if vector_id and vector_id not in seen_ids and len(combined_results) < match_count:
            combined_results.append(vector_item)
            seen_ids.add(vector_id)

    for keyword_item in keyword_results:
        keyword_id = keyword_item.get("id")
        if keyword_id not in seen_ids and len(combined_results) < match_count:
            combined_results.append(convert_keyword_fn(keyword_item))
            seen_ids.add(keyword_id)

    return combined_results[:match_count]


def _keyword_document_search(
    supabase_client: Any,
    query: str,
    source: Optional[str],
    match_count: int,
) -> List[Dict[str, Any]]:
    keyword_query = (
        supabase_client.from_("crawled_pages")
        .select("id, url, chunk_number, content, metadata, source_id")
        .ilike("content", f"%{query}%")
    )
    if source and source.strip():
        keyword_query = keyword_query.eq("source_id", source)
    response = keyword_query.limit(match_count).execute()
    return response.data if response.data else []


def _keyword_code_search(
    supabase_client: Any,
    query: str,
    source_id: Optional[str],
    match_count: int,
) -> List[Dict[str, Any]]:
    keyword_query = (
        supabase_client.from_("code_examples")
        .select("id, url, chunk_number, content, summary, metadata, source_id")
        .or_(f"content.ilike.%{query}%,summary.ilike.%{query}%")
    )
    if source_id and source_id.strip():
        keyword_query = keyword_query.eq("source_id", source_id)
    response = keyword_query.limit(match_count).execute()
    return response.data if response.data else []


def fetch_sources(supabase_client: Any) -> List[Dict[str, Any]]:
    """Return all sources from Supabase."""

    result = supabase_client.from_("sources").select("*").order("source_id").execute()
    sources = []
    if result.data:
        for item in result.data:
            sources.append(
                {
                    "source_id": item.get("source_id"),
                    "summary": item.get("summary"),
                    "total_words": item.get("total_words"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                }
            )
    return sources


def execute_document_query(
    *,
    supabase_client: Any,
    query: str,
    source: Optional[str],
    match_count: int,
    use_hybrid_search: bool,
    use_reranking: bool,
    reranking_available: bool,
    vector_search_fn: Callable[..., List[Dict[str, Any]]] = search_documents,
    keyword_search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Execute the document RAG query and return formatted payload."""

    filter_metadata = {"source": source} if source and source.strip() else None
    base_match_count = match_count * 2 if use_hybrid_search else match_count

    vector_results = vector_search_fn(
        client=supabase_client,
        query=query,
        match_count=base_match_count,
        filter_metadata=filter_metadata,
    )

    if use_hybrid_search:
        keyword_fn = keyword_search_fn or _keyword_document_search
        keyword_results = keyword_fn(
            supabase_client,
            query,
            source,
            match_count * 2,
        )
        results = _combine_semantic_and_keyword_results(
            vector_results,
            keyword_results,
            match_count,
            lambda item: {
                "id": item.get("id"),
                "url": item.get("url"),
                "chunk_number": item.get("chunk_number"),
                "content": item.get("content"),
                "metadata": item.get("metadata"),
                "source_id": item.get("source_id"),
                "similarity": 0.5,
            },
        )
    else:
        results = vector_results[:match_count]

    reranking_applied = use_reranking and reranking_available
    if reranking_applied:
        results = rerank_results(query, results, content_key="content")

    formatted_results = []
    for item in results:
        payload = {
            "url": item.get("url"),
            "content": item.get("content"),
            "metadata": item.get("metadata"),
            "similarity": item.get("similarity"),
        }
        if "rerank_score" in item:
            payload["rerank_score"] = item["rerank_score"]
        formatted_results.append(payload)

    return {
        "query": query,
        "source_filter": source,
        "search_mode": "hybrid" if use_hybrid_search else "vector",
        "reranking_applied": reranking_applied,
        "results": formatted_results,
        "count": len(formatted_results),
    }


def execute_code_example_query(
    *,
    supabase_client: Any,
    query: str,
    source_id: Optional[str],
    match_count: int,
    use_hybrid_search: bool,
    use_reranking: bool,
    reranking_available: bool,
    vector_search_fn: Callable[..., List[Dict[str, Any]]] = search_code_examples,
    keyword_search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Execute the code example query and return formatted payload."""

    filter_metadata = {"source": source_id} if source_id and source_id.strip() else None
    base_match_count = match_count * 2 if use_hybrid_search else match_count

    vector_results = vector_search_fn(
        client=supabase_client,
        query=query,
        match_count=base_match_count,
        filter_metadata=filter_metadata,
    )

    if use_hybrid_search:
        keyword_fn = keyword_search_fn or _keyword_code_search
        keyword_results = keyword_fn(
            supabase_client,
            query,
            source_id,
            match_count * 2,
        )
        results = _combine_semantic_and_keyword_results(
            vector_results,
            keyword_results,
            match_count,
            lambda item: {
                "id": item.get("id"),
                "url": item.get("url"),
                "chunk_number": item.get("chunk_number"),
                "content": item.get("content"),
                "summary": item.get("summary"),
                "metadata": item.get("metadata"),
                "source_id": item.get("source_id"),
                "similarity": 0.5,
            },
        )
    else:
        results = vector_results[:match_count]

    reranking_applied = use_reranking and reranking_available
    if reranking_applied:
        results = rerank_results(query, results, content_key="content")

    formatted_results = []
    for item in results:
        payload = {
            "url": item.get("url"),
            "code": item.get("content"),
            "summary": item.get("summary"),
            "metadata": item.get("metadata"),
            "source_id": item.get("source_id"),
            "similarity": item.get("similarity"),
        }
        if "rerank_score" in item:
            payload["rerank_score"] = item["rerank_score"]
        formatted_results.append(payload)

    return {
        "query": query,
        "source_filter": source_id,
        "search_mode": "hybrid" if use_hybrid_search else "vector",
        "reranking_applied": reranking_applied,
        "results": formatted_results,
        "count": len(formatted_results),
    }
