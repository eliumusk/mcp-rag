"""Local file ingestion MCP tools."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import concurrent.futures
import json
import os

from mcp.server.fastmcp import Context, FastMCP

from text_processing import (
    collect_local_files,
    extract_section_info,
    parse_file_paths_input,
    process_code_example,
    smart_chunk_markdown,
)
from utils import (
    add_code_examples_to_supabase,
    add_documents_to_supabase,
    extract_code_blocks,
    extract_source_summary,
    load_local_document,
    update_source_info,
)


async def ingest_local_files(
    ctx: Context,
    file_paths: str,
    source_id: str | None = None,
    recursive: bool = False,
) -> str:
    """Ingest markdown/text/PDF files into Supabase."""

    supabase_client = ctx.request_context.lifespan_context.supabase_client
    parsed_paths = parse_file_paths_input(file_paths)
    if not parsed_paths:
        return json.dumps(
            {"success": False, "error": "No file paths provided. Pass a JSON array, newline, or comma-separated paths."},
            indent=2,
        )

    files = collect_local_files(parsed_paths, recursive=recursive)
    if not files:
        return json.dumps(
            {"success": False, "error": "No matching files were found for the provided paths."},
            indent=2,
        )

    ingest_time = datetime.utcnow().isoformat()
    enable_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"

    summaries: List[Dict[str, Any]] = []
    errors: List[str] = []
    total_chunks = 0
    total_files = 0

    for file_path in files:
        try:
            content = load_local_document(str(file_path))
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")
            continue

        if not content or not content.strip():
            errors.append(f"{file_path}: File is empty after extraction.")
            continue

        url = f"file://{file_path.as_posix()}"
        derived_source = source_id or file_path.stem
        chunks = smart_chunk_markdown(content)

        if not chunks:
            errors.append(f"{file_path}: No chunks were produced (file may be too small).")
            continue

        urls: List[str] = []
        chunk_numbers: List[int] = []
        contents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        total_word_count = 0

        for idx, chunk in enumerate(chunks):
            urls.append(url)
            chunk_numbers.append(idx)
            contents.append(chunk)
            meta = extract_section_info(chunk)
            meta["chunk_index"] = idx
            meta["file_path"] = str(file_path)
            meta["source"] = derived_source
            meta["ingest_time"] = ingest_time
            metadatas.append(meta)
            total_word_count += meta.get("word_count", 0)

        url_to_full_document = {url: content}
        try:
            source_summary = extract_source_summary(derived_source, content[:5000])
            update_source_info(supabase_client, derived_source, source_summary, total_word_count)
            add_documents_to_supabase(
                supabase_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document,
            )
        except Exception as exc:
            errors.append(f"{file_path}: Failed to insert chunks ({exc})")
            continue

        code_examples_stored = 0
        if enable_code_examples:
            code_blocks = extract_code_blocks(content)
            if code_blocks:
                code_urls: List[str] = []
                code_chunk_numbers: List[int] = []
                code_examples: List[str] = []
                code_summaries: List[str] = []
                code_metadatas: List[Dict[str, Any]] = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    summary_args = [(block["code"], block["context_before"], block["context_after"]) for block in code_blocks]
                    summaries_list = list(executor.map(process_code_example, summary_args))

                for idx, (block, summary) in enumerate(zip(code_blocks, summaries_list)):
                    code_urls.append(url)
                    code_chunk_numbers.append(idx)
                    code_examples.append(block["code"])
                    code_summaries.append(summary)
                    code_metadatas.append(
                        {
                            "chunk_index": idx,
                            "file_path": str(file_path),
                            "source": derived_source,
                            "char_count": len(block["code"]),
                            "word_count": len(block["code"].split()),
                        }
                    )

                add_code_examples_to_supabase(
                    supabase_client,
                    code_urls,
                    code_chunk_numbers,
                    code_examples,
                    code_summaries,
                    code_metadatas,
                )
                code_examples_stored = len(code_examples)

        summaries.append(
            {
                "file": str(file_path),
                "url": url,
                "chunks_stored": len(chunks),
                "code_examples_stored": code_examples_stored,
                "source_id": derived_source,
                "word_count": total_word_count,
            }
        )
        total_chunks += len(chunks)
        total_files += 1

    return json.dumps(
        {
            "success": total_files > 0,
            "files_processed": total_files,
            "chunks_stored": total_chunks,
            "results": summaries,
            "errors": errors,
        },
        indent=2,
    )


def register_ingestion_tools(mcp: FastMCP) -> None:
    """Register ingestion tools against FastMCP."""

    mcp.tool()(ingest_local_files)
