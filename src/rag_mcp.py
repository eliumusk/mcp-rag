"""
MCP server for web crawling with Crawl4AI.
"""
from pathlib import Path

import asyncio
import os
import sys

from mcp.server.fastmcp import FastMCP

# Ensure knowledge_graph modules can be imported directly
knowledge_graphs_path = Path(__file__).resolve().parent.parent / "knowledge_graphs"
if str(knowledge_graphs_path) not in sys.path:
    sys.path.append(str(knowledge_graphs_path))

from server_context import crawl4ai_lifespan
from tools.crawling import register_crawling_tools
from tools.ingestion import register_ingestion_tools
from tools.knowledge_graph import register_knowledge_graph_tools
from tools.queries import register_query_tools


mcp = FastMCP(
    "rag-mcp",
    description="MCP server for RAG and web crawling with Jina",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051"),
)

register_crawling_tools(mcp)
register_ingestion_tools(mcp)
register_query_tools(mcp)
register_knowledge_graph_tools(mcp)


async def main() -> None:
    transport = os.getenv("TRANSPORT", "sse")
    if transport == "sse":
        await mcp.run_sse_async()
    else:
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
