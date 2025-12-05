"""Supabase-backed query MCP tools."""

import os

from mcp.server.fastmcp import Context, FastMCP

from services.logger import log_error, log_info
from services.query_service import (
    execute_code_example_query,
    execute_document_query,
    fetch_sources,
)
from services.responses import error_response, success_response


async def get_available_sources(ctx: Context) -> str:
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        sources = fetch_sources(supabase_client)
        log_info("get_available_sources", "completed", count=len(sources))
        return success_response("Fetched available sources", data={"sources": sources, "count": len(sources)})
    except Exception as exc:
        log_error("get_available_sources", "failed", error=str(exc))
        return error_response("SOURCES_FETCH_FAILED", "Failed to fetch sources", details={"reason": str(exc)})


async def perform_rag_query(ctx: Context, query: str, source: str | None = None, match_count: int = 5) -> str:
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        reranking_available = ctx.request_context.lifespan_context.reranking_enabled

        payload = execute_document_query(
            supabase_client=supabase_client,
            query=query,
            source=source,
            match_count=match_count,
            use_hybrid_search=use_hybrid_search,
            use_reranking=use_reranking,
            reranking_available=reranking_available,
        )
        log_info(
            "perform_rag_query",
            "completed",
            query=query,
            source=source,
            results=payload.get("count", 0),
        )
        return success_response("RAG query executed", data=payload)
    except Exception as exc:
        log_error("perform_rag_query", "failed", query=query, error=str(exc))
        return error_response("RAG_QUERY_FAILED", "Failed to execute RAG query", details={"query": query, "reason": str(exc)})


async def search_code_examples(
    ctx: Context,
    query: str,
    source_id: str | None = None,
    match_count: int = 5,
) -> str:
    if os.getenv("USE_AGENTIC_RAG", "false") != "true":
        return error_response(
            "CODE_SEARCH_DISABLED",
            "Code example extraction is disabled. Perform a normal RAG search.",
        )

    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        reranking_available = ctx.request_context.lifespan_context.reranking_enabled

        payload = execute_code_example_query(
            supabase_client=supabase_client,
            query=query,
            source_id=source_id,
            match_count=match_count,
            use_hybrid_search=use_hybrid_search,
            use_reranking=use_reranking,
            reranking_available=reranking_available,
        )
        log_info(
            "search_code_examples",
            "completed",
            query=query,
            source=source_id,
            results=payload.get("count", 0),
        )
        return success_response("Code example query executed", data=payload)
    except Exception as exc:
        log_error("search_code_examples", "failed", query=query, error=str(exc))
        return error_response(
            "CODE_QUERY_FAILED",
            "Failed to search code examples",
            details={"query": query, "reason": str(exc)},
        )


def register_query_tools(mcp: FastMCP) -> None:
    mcp.tool()(get_available_sources)
    mcp.tool()(perform_rag_query)
    mcp.tool()(search_code_examples)
