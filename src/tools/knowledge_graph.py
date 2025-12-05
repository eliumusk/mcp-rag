"""Knowledge graph focused MCP tools."""

import json
import os
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context, FastMCP

from ai_script_analyzer import AIScriptAnalyzer
from hallucination_reporter import HallucinationReporter


def validate_script_path(script_path: str) -> Dict[str, Any]:
    if not script_path or not isinstance(script_path, str):
        return {"valid": False, "error": "Script path is required"}
    if not os.path.exists(script_path):
        return {"valid": False, "error": f"Script not found: {script_path}"}
    if not script_path.endswith(".py"):
        return {"valid": False, "error": "Only Python (.py) files are supported"}
    try:
        with open(script_path, "r", encoding="utf-8") as handle:
            handle.read(1)
        return {"valid": True}
    except Exception as exc:
        return {"valid": False, "error": f"Cannot read script file: {exc}"}


def validate_github_url(repo_url: str) -> Dict[str, Any]:
    if not repo_url or not isinstance(repo_url, str):
        return {"valid": False, "error": "Repository URL is required"}
    repo_url = repo_url.strip()
    if not ("github.com" in repo_url.lower() or repo_url.endswith(".git")):
        return {"valid": False, "error": "Please provide a valid GitHub repository URL"}
    if not (repo_url.startswith("https://") or repo_url.startswith("git@")):
        return {"valid": False, "error": "Repository URL must start with https:// or git@"}
    return {"valid": True, "repo_name": repo_url.split("/")[-1].replace(".git", "")}


async def check_ai_script_hallucinations(ctx: Context, script_path: str) -> str:
    try:
        if os.getenv("USE_KNOWLEDGE_GRAPH", "false") != "true":
            return json.dumps(
                {
                    "success": False,
                    "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment.",
                },
                indent=2,
            )

        knowledge_validator = ctx.request_context.lifespan_context.knowledge_validator
        if not knowledge_validator:
            return json.dumps(
                {
                    "success": False,
                    "error": "Knowledge graph validator not available. Check Neo4j configuration in environment variables.",
                },
                indent=2,
            )

        validation = validate_script_path(script_path)
        if not validation["valid"]:
            return json.dumps({"success": False, "script_path": script_path, "error": validation["error"]}, indent=2)

        analyzer = AIScriptAnalyzer()
        analysis_result = analyzer.analyze_script(script_path)
        if analysis_result.errors:
            print(f"Analysis warnings for {script_path}: {analysis_result.errors}")

        validation_result = await knowledge_validator.validate_script(analysis_result)
        reporter = HallucinationReporter()
        report = reporter.generate_comprehensive_report(validation_result)

        return json.dumps(
            {
                "success": True,
                "script_path": script_path,
                "overall_confidence": validation_result.overall_confidence,
                "validation_summary": report["validation_summary"],
                "hallucinations_detected": report["hallucinations_detected"],
                "recommendations": report["recommendations"],
                "analysis_metadata": report["analysis_metadata"],
                "libraries_analyzed": report.get("libraries_analyzed", []),
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"success": False, "script_path": script_path, "error": f"Analysis failed: {exc}"}, indent=2)


async def query_knowledge_graph(ctx: Context, command: str) -> str:
    try:
        if os.getenv("USE_KNOWLEDGE_GRAPH", "false") != "true":
            return json.dumps(
                {
                    "success": False,
                    "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment.",
                },
                indent=2,
            )

        repo_extractor = ctx.request_context.lifespan_context.repo_extractor
        if not repo_extractor or not repo_extractor.driver:
            return json.dumps(
                {
                    "success": False,
                    "error": "Neo4j connection not available. Check Neo4j configuration in environment variables.",
                },
                indent=2,
            )

        cleaned_command = command.strip()
        if not cleaned_command:
            return json.dumps(
                {
                    "success": False,
                    "command": "",
                    "error": "Command cannot be empty. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>",
                },
                indent=2,
            )

        parts = cleaned_command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        async with repo_extractor.driver.session() as session:
            if cmd == "repos":
                return await _handle_repos_command(session, cleaned_command)
            if cmd == "explore":
                if not args:
                    return json.dumps(
                        {
                            "success": False,
                            "command": cleaned_command,
                            "error": "Repository name required. Usage: explore <repo_name>",
                        },
                        indent=2,
                    )
                return await _handle_explore_command(session, cleaned_command, args[0])
            if cmd == "classes":
                repo_name = args[0] if args else None
                return await _handle_classes_command(session, cleaned_command, repo_name)
            if cmd == "class":
                if not args:
                    return json.dumps(
                        {
                            "success": False,
                            "command": cleaned_command,
                            "error": "Class name required. Usage: class <class_name>",
                        },
                        indent=2,
                    )
                return await _handle_class_command(session, cleaned_command, args[0])
            if cmd == "method":
                if not args:
                    return json.dumps(
                        {
                            "success": False,
                            "command": cleaned_command,
                            "error": "Method name required. Usage: method <method_name> [class_name]",
                        },
                        indent=2,
                    )
                method_name = args[0]
                class_name = args[1] if len(args) > 1 else None
                return await _handle_method_command(session, cleaned_command, method_name, class_name)
            if cmd == "query":
                if not args:
                    return json.dumps(
                        {
                            "success": False,
                            "command": cleaned_command,
                            "error": "Cypher query required. Usage: query <cypher_query>",
                        },
                        indent=2,
                    )
                cypher_query = " ".join(args)
                return await _handle_query_command(session, cleaned_command, cypher_query)
            return json.dumps(
                {
                    "success": False,
                    "command": cleaned_command,
                    "error": "Unknown command '{cmd}'. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>",
                },
                indent=2,
            )
    except Exception as exc:
        return json.dumps({"success": False, "command": command, "error": f"Query execution failed: {exc}"}, indent=2)


async def _handle_repos_command(session: Any, command: str) -> str:
    query = "MATCH (r:Repository) RETURN r.name as name ORDER BY r.name"
    result = await session.run(query)
    repos: list[str] = []
    async for record in result:
        repos.append(record["name"])
    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {"repositories": repos},
            "metadata": {"total_results": len(repos), "limited": False},
        },
        indent=2,
    )


async def _handle_explore_command(session: Any, command: str, repo_name: str) -> str:
    repo_check_query = "MATCH (r:Repository {name: $repo_name}) RETURN r.name as name"
    result = await session.run(repo_check_query, repo_name=repo_name)
    repo_record = await result.single()
    if not repo_record:
        return json.dumps(
            {"success": False, "command": command, "error": f"Repository '{repo_name}' not found in knowledge graph"},
            indent=2,
        )

    files_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
    RETURN count(f) as file_count
    """
    result = await session.run(files_query, repo_name=repo_name)
    file_count = (await result.single())["file_count"]

    classes_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
    RETURN count(DISTINCT c) as class_count
    """
    result = await session.run(classes_query, repo_name=repo_name)
    class_count = (await result.single())["class_count"]

    functions_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
    RETURN count(DISTINCT func) as function_count
    """
    result = await session.run(functions_query, repo_name=repo_name)
    function_count = (await result.single())["function_count"]

    methods_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
    RETURN count(DISTINCT m) as method_count
    """
    result = await session.run(methods_query, repo_name=repo_name)
    method_count = (await result.single())["method_count"]

    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {
                "repository": repo_name,
                "statistics": {
                    "files": file_count,
                    "classes": class_count,
                    "functions": function_count,
                    "methods": method_count,
                },
            },
            "metadata": {"total_results": 1, "limited": False},
        },
        indent=2,
    )


async def _handle_classes_command(session: Any, command: str, repo_name: Optional[str]) -> str:
    limit = 20
    if repo_name:
        query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, repo_name=repo_name, limit=limit)
    else:
        query = """
        MATCH (c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, limit=limit)

    classes = []
    async for record in result:
        classes.append({"name": record["name"], "full_name": record["full_name"]})

    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {"classes": classes, "repository_filter": repo_name},
            "metadata": {"total_results": len(classes), "limited": len(classes) >= limit},
        },
        indent=2,
    )


async def _handle_class_command(session: Any, command: str, class_name: str) -> str:
    class_query = """
    MATCH (c:Class)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN c.name as name, c.full_name as full_name
    LIMIT 1
    """
    result = await session.run(class_query, class_name=class_name)
    class_record = await result.single()
    if not class_record:
        return json.dumps(
            {"success": False, "command": command, "error": f"Class '{class_name}' not found in knowledge graph"},
            indent=2,
        )

    actual_name = class_record["name"]
    full_name = class_record["full_name"]

    methods_query = """
    MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN m.name as name, m.params_list as params_list, m.params_detailed as params_detailed, m.return_type as return_type
    ORDER BY m.name
    """
    result = await session.run(methods_query, class_name=class_name)
    methods = []
    async for record in result:
        params_to_use = record["params_detailed"] or record["params_list"] or []
        methods.append({"name": record["name"], "parameters": params_to_use, "return_type": record["return_type"] or "Any"})

    attributes_query = """
    MATCH (c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN a.name as name, a.type as type
    ORDER BY a.name
    """
    result = await session.run(attributes_query, class_name=class_name)
    attributes = []
    async for record in result:
        attributes.append({"name": record["name"], "type": record["type"] or "Any"})

    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {"class": {"name": actual_name, "full_name": full_name, "methods": methods, "attributes": attributes}},
            "metadata": {
                "total_results": 1,
                "methods_count": len(methods),
                "attributes_count": len(attributes),
                "limited": False,
            },
        },
        indent=2,
    )


async def _handle_method_command(
    session: Any,
    command: str,
    method_name: str,
    class_name: Optional[str],
) -> str:
    if class_name:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE (c.name = $class_name OR c.full_name = $class_name)
          AND m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list,
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        """
        result = await session.run(query, class_name=class_name, method_name=method_name)
    else:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list,
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        ORDER BY c.name
        LIMIT 20
        """
        result = await session.run(query, method_name=method_name)

    methods = []
    async for record in result:
        params_to_use = record["params_detailed"] or record["params_list"] or []
        methods.append(
            {
                "class_name": record["class_name"],
                "class_full_name": record["class_full_name"],
                "method_name": record["method_name"],
                "parameters": params_to_use,
                "return_type": record["return_type"] or "Any",
                "legacy_args": record["args"] or [],
            }
        )

    if not methods:
        suffix = f" in class '{class_name}'" if class_name else ""
        return json.dumps(
            {"success": False, "command": command, "error": f"Method '{method_name}'{suffix} not found"},
            indent=2,
        )

    return json.dumps(
        {
            "success": True,
            "command": command,
            "data": {"methods": methods, "class_filter": class_name},
            "metadata": {"total_results": len(methods), "limited": len(methods) >= 20 and not class_name},
        },
        indent=2,
    )


async def _handle_query_command(session: Any, command: str, cypher_query: str) -> str:
    try:
        result = await session.run(cypher_query)
        records = []
        count = 0
        async for record in result:
            records.append(dict(record))
            count += 1
            if count >= 20:
                break
        return json.dumps(
            {
                "success": True,
                "command": command,
                "data": {"query": cypher_query, "results": records},
                "metadata": {"total_results": len(records), "limited": len(records) >= 20},
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "command": command,
                "error": f"Cypher query error: {exc}",
                "data": {"query": cypher_query},
            },
            indent=2,
        )


async def parse_github_repository(ctx: Context, repo_url: str) -> str:
    try:
        if os.getenv("USE_KNOWLEDGE_GRAPH", "false") != "true":
            return json.dumps(
                {
                    "success": False,
                    "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment.",
                },
                indent=2,
            )

        repo_extractor = ctx.request_context.lifespan_context.repo_extractor
        if not repo_extractor:
            return json.dumps(
                {
                    "success": False,
                    "error": "Repository extractor not available. Check Neo4j configuration in environment variables.",
                },
                indent=2,
            )

        validation = validate_github_url(repo_url)
        if not validation["valid"]:
            return json.dumps({"success": False, "repo_url": repo_url, "error": validation["error"]}, indent=2)

        repo_name = validation["repo_name"]
        print(f"Starting repository analysis for: {repo_name}")
        await repo_extractor.analyze_repository(repo_url)
        print(f"Repository analysis completed for: {repo_name}")

        async with repo_extractor.driver.session() as session:
            stats_query = """
            MATCH (r:Repository {name: $repo_name})
            OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
            OPTIONAL MATCH (c)-[:HAS_ATTRIBUTE]->(a:Attribute)
            WITH r,
                 count(DISTINCT f) as files_count,
                 count(DISTINCT c) as classes_count,
                 count(DISTINCT m) as methods_count,
                 count(DISTINCT func) as functions_count,
                 count(DISTINCT a) as attributes_count

            OPTIONAL MATCH (r)-[:CONTAINS]->(sample_f:File)
            WITH r, files_count, classes_count, methods_count, functions_count, attributes_count,
                 collect(DISTINCT sample_f.module_name)[0..5] as sample_modules

            RETURN r.name as repo_name,
                   files_count,
                   classes_count,
                   methods_count,
                   functions_count,
                   attributes_count,
                   sample_modules
            """
            result = await session.run(stats_query, repo_name=repo_name)
            record = await result.single()
            if record:
                stats = {
                    "repository": record["repo_name"],
                    "files_processed": record["files_count"],
                    "classes_created": record["classes_count"],
                    "methods_created": record["methods_count"],
                    "functions_created": record["functions_count"],
                    "attributes_created": record["attributes_count"],
                    "sample_modules": record["sample_modules"] or [],
                }
            else:
                return json.dumps(
                    {
                        "success": False,
                        "repo_url": repo_url,
                        "error": f"Repository '{repo_name}' not found in database after parsing",
                    },
                    indent=2,
                )

        return json.dumps(
            {
                "success": True,
                "repo_url": repo_url,
                "repo_name": repo_name,
                "message": f"Successfully parsed repository '{repo_name}' into knowledge graph",
                "statistics": stats,
                "ready_for_validation": True,
                "next_steps": [
                    "Repository is now available for hallucination detection",
                    f"Use check_ai_script_hallucinations to validate scripts against {repo_name}",
                    "The knowledge graph contains classes, methods, and functions from this repository",
                ],
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"success": False, "repo_url": repo_url, "error": f"Repository parsing failed: {exc}"}, indent=2)


def register_knowledge_graph_tools(mcp: FastMCP) -> None:
    mcp.tool()(check_ai_script_hallucinations)
    mcp.tool()(query_knowledge_graph)
    mcp.tool()(parse_github_repository)
