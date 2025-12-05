# 所有回答用中文回复
# Repository Guidelines

## Project Structure & Module Organization
- Core MCP source lives in `src/`. `rag_mcp.py` wires FastMCP tools, Jina-based fetchers, Supabase, and optional Neo4j components; `utils.py` centralizes Supabase/RAG helpers shared by the tools.
- Knowledge-graph utilities (`knowledge_graphs/`) house the Neo4j extractors, script analyzers, and hallucination reporters that are imported dynamically when `USE_KNOWLEDGE_GRAPH=true`.
- Infrastructure artifacts: `Dockerfile` builds a minimal SSE-ready image, `uv.lock` pins Python dependencies, and `crawled_pages.sql` demonstrates the Supabase schema. Place temporary assets under `tmp/` (gitignored) and keep new modules inside `src/` or `knowledge_graphs/` as appropriate.

## Build, Test, and Development Commands
- `uv pip install -e .` — install editable dependencies against Python 3.12 in the active `uv` virtualenv.
- `uv run src/rag_mcp.py` — launch the MCP server locally (SSE by default); export `.env` variables such as `JINA_API_KEY`, `LLM_API_KEY`, `LLM_API_BASE`, `SUPABASE_URL`, and optional `USE_KNOWLEDGE_GRAPH=true` beforehand.
- `docker build -t mcp/crawl4ai .` — produce a container image with the server entrypoint.
- `docker run --rm -p 8051:8051 --env-file .env mcp/crawl4ai` — start the server in Docker for clients that expect `http://localhost:8051/sse`.

## Coding Style & Naming Conventions
- Follow standard Python formatting: 4-space indentation, double-quoted docstrings, and module-level constants in `SCREAMING_SNAKE_CASE`.
- Use descriptive, typed function signatures and async context managers when dealing with crawler or database lifecycles. Favor `snake_case` for functions/variables and `CamelCase` for classes/dataclasses.
- Keep imports grouped (stdlib → third-party → local) as seen in `src/rag_mcp.py`, and guard optional dependencies with feature flags instead of brittle `try/except pass` blocks.

## Testing Guidelines
- Add new tests under `tests/` using `pytest`. Mirror filenames after their targets (e.g., `test_utils.py`) and focus on deterministic units by mocking Supabase, HTTP calls, and Neo4j clients.
- Aim for coverage of every new branch you introduce, especially around async tool handlers, data transformations, and error formatting helpers.
- Run `uv run pytest` before submitting changes; integrate regression checks into CI when available.

## Commit & Pull Request Guidelines
- Write imperative, concise commit subjects (<72 chars) similar to the existing history (“Add knowledge graph functionality”). Include context or migration notes in the body when touching env vars or schemas.
- Reference related issues in the PR description, summarize behavioral changes, document new environment flags, and attach logs or screenshots that prove crawls, RAG queries, or knowledge-graph workflows succeed.
- Ensure PRs leave the repo runnable: keep `.env.example` current, update README snippets when commands change, and describe validation steps so reviewers can reproduce them quickly.

## Security & Configuration Tips
- Never commit actual API keys, Supabase credentials, or Neo4j passwords. Store secrets in `.env` (already ignored) and describe required keys via placeholder names.
- When adding new tools, enforce parameter validation similar to `validate_github_url` to prevent accidental external calls or misconfigured crawls.
