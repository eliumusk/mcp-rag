"""Server lifecycle utilities for the Crawl4AI MCP instance."""


from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import os
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from supabase import Client

project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

from utils import get_supabase_client

# Ensure knowledge_graphs is importable when running from src/
knowledge_graphs_path = Path(__file__).resolve().parent.parent / "knowledge_graphs"
if str(knowledge_graphs_path) not in sys.path:
    sys.path.append(str(knowledge_graphs_path))

from knowledge_graph_validator import KnowledgeGraphValidator
from parse_repo_into_neo4j import DirectNeo4jExtractor


def format_neo4j_error(error: Exception) -> str:
    """Format Neo4j connection errors for user-friendly messages."""
    error_str = str(error).lower()
    if "authentication" in error_str or "unauthorized" in error_str:
        return "Neo4j authentication failed. Check NEO4J_USER and NEO4J_PASSWORD."
    if "connection" in error_str or "refused" in error_str or "timeout" in error_str:
        return "Cannot connect to Neo4j. Check NEO4J_URI and ensure Neo4j is running."
    if "database" in error_str:
        return "Neo4j database error. Check if the database exists and is accessible."
    return f"Neo4j error: {error}"


@dataclass
class Crawl4AIContext:
    """Context shared across MCP tools."""

    supabase_client: Client
    tenant_id: str
    reranking_enabled: bool = False
    knowledge_validator: Optional[Any] = None
    repo_extractor: Optional[Any] = None


def _should_enable_reranking() -> bool:
    if os.getenv("USE_RERANKING", "false") != "true":
        return False
    if not os.getenv("JINA_API_KEY"):
        print("USE_RERANKING=true but JINA_API_KEY is missing. Disable reranking or set the API key.")
        return False
    return True


def _get_env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


async def _initialize_knowledge_graph_components() -> tuple[Optional[Any], Optional[Any]]:
    if os.getenv("USE_KNOWLEDGE_GRAPH", "false") != "true":
        print("Knowledge graph functionality disabled - set USE_KNOWLEDGE_GRAPH=true to enable")
        return None, None

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = _get_env("NEO4J_USER", "NEO4J_USERNAME")
    neo4j_password = _get_env("NEO4J_PASSWORD", "NEO4J_PASS", "NEO4J_SECRET")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("Neo4j credentials not configured - knowledge graph tools will be unavailable")
        return None, None

    try:
        print("Initializing knowledge graph components...")
        knowledge_validator = KnowledgeGraphValidator(neo4j_uri, neo4j_user, neo4j_password)
        await knowledge_validator.initialize()
        print("✓ Knowledge graph validator initialized")

        repo_extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)
        await repo_extractor.initialize()
        print("✓ Repository extractor initialized")
        return knowledge_validator, repo_extractor
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to initialize Neo4j components: {format_neo4j_error(exc)}")
        return None, None


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """Manage crawler and database clients across the MCP lifespan."""

    supabase_client = get_supabase_client()
    tenant_id = os.getenv("TENANT_ID", "default")
    reranking_enabled = _should_enable_reranking()
    knowledge_validator, repo_extractor = await _initialize_knowledge_graph_components()

    try:
        yield Crawl4AIContext(
            supabase_client=supabase_client,
            tenant_id=tenant_id,
            reranking_enabled=reranking_enabled,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor,
        )
    finally:
        if knowledge_validator:
            try:
                await knowledge_validator.close()
                print("✓ Knowledge graph validator closed")
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Error closing knowledge validator: {exc}")
        if repo_extractor:
            try:
                await repo_extractor.close()
                print("✓ Repository extractor closed")
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Error closing repository extractor: {exc}")
