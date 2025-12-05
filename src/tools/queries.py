"""Supabase-backed query MCP tools."""

import json
import os

from mcp.server.fastmcp import Context, FastMCP

from services.query_service import (
    execute_code_example_query,
    execute_document_query,
    fetch_sources,
)


async def get_available_sources(ctx: Context) -> str:
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        sources = fetch_sources(supabase_client)
        return json.dumps({"success": True, "sources": sources, "count": len(sources)}, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


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
        return json.dumps(payload, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "query": query, "error": str(exc)}, indent=2)


async def search_code_examples(
    ctx: Context,
    query: str,
    source_id: str | None = None,
    match_count: int = 5,
) -> str:
    if os.getenv("USE_AGENTIC_RAG", "false") != "true":
        return json.dumps(
            {
                "success": False,
                "error": "Code example extraction is disabled. Perform a normal RAG search.",
            },
            indent=2,
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
        return json.dumps(payload, indent=2)
    except Exception as exc:
        return json.dumps({"success": False, "query": query, "error": str(exc)}, indent=2)


def register_query_tools(mcp: FastMCP) -> None:
    mcp.tool()(get_available_sources)
    mcp.tool()(perform_rag_query)
    mcp.tool()(search_code_examples)
