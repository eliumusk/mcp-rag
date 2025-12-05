"""Crawl-related MCP tools and helpers."""

from typing import Any, Dict, List

import asyncio
import concurrent.futures
import json
import os
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree

import requests
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from mcp.server.fastmcp import Context, FastMCP

from text_processing import extract_section_info, process_code_example, smart_chunk_markdown
from utils import (
    add_code_examples_to_supabase,
    add_documents_to_supabase,
    extract_code_blocks,
    extract_source_summary,
    update_source_info,
)


def is_sitemap(url: str) -> bool:
    """Return True if URL appears to reference a sitemap."""

    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path


def is_txt(url: str) -> bool:
    """Return True when URL looks like a plaintext resource."""

    return url.endswith(".txt")


def parse_sitemap(sitemap_url: str) -> List[str]:
    """Parse sitemap URLs into a list of endpoints."""

    resp = requests.get(sitemap_url, timeout=30)
    if resp.status_code != 200:
        return []

    try:
        tree = ElementTree.fromstring(resp.content)
        return [loc.text for loc in tree.findall(".//{*}loc")]
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Error parsing sitemap XML: {exc}")
        return []


async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """Crawl a plain text or markdown file via the crawler."""

    crawl_config = CrawlerRunConfig()
    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{"url": url, "markdown": result.markdown}]
    print(f"Failed to crawl {url}: {result.error_message}")
    return []


async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """Crawl multiple URLs concurrently."""

    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    return [{"url": r.url, "markdown": r.markdown} for r in results if r.success and r.markdown]


async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
) -> List[Dict[str, Any]]:
    """Recursively crawl internal links up to ``max_depth`` levels."""

    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    visited = set()

    def normalize_url(url: str) -> str:
        return urldefrag(url)[0]

    current_urls = {normalize_url(u) for u in start_urls}
    results_all: List[Dict[str, Any]] = []

    for _ in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({"url": result.url, "markdown": result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all


async def crawl_single_page(ctx: Context, url: str) -> str:
    """Crawl a single web page and store its content in Supabase."""

    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)

        if not (result.success and result.markdown):
            return json.dumps({"success": False, "url": url, "error": result.error_message}, indent=2)

        parsed_url = urlparse(url)
        source_id = parsed_url.netloc or parsed_url.path
        chunks = smart_chunk_markdown(result.markdown)

        urls: List[str] = []
        chunk_numbers: List[int] = []
        contents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        total_word_count = 0

        for i, chunk in enumerate(chunks):
            urls.append(url)
            chunk_numbers.append(i)
            contents.append(chunk)

            meta = extract_section_info(chunk)
            meta["chunk_index"] = i
            meta["url"] = url
            meta["source"] = source_id
            meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
            metadatas.append(meta)
            total_word_count += meta.get("word_count", 0)

        url_to_full_document = {url: result.markdown}
        source_summary = extract_source_summary(source_id, result.markdown[:5000])
        update_source_info(supabase_client, source_id, source_summary, total_word_count)
        add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)

        code_examples_stored = 0
        code_blocks = None
        if os.getenv("USE_AGENTIC_RAG", "false") == "true":
            code_blocks = extract_code_blocks(result.markdown)
            if code_blocks:
                code_urls: List[str] = []
                code_chunk_numbers: List[int] = []
                code_examples: List[str] = []
                code_summaries: List[str] = []
                code_metadatas: List[Dict[str, Any]] = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    summary_args = [(block["code"], block["context_before"], block["context_after"]) for block in code_blocks]
                    summaries = list(executor.map(process_code_example, summary_args))

                for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                    code_urls.append(url)
                    code_chunk_numbers.append(i)
                    code_examples.append(block["code"])
                    code_summaries.append(summary)
                    code_metadatas.append(
                        {
                            "chunk_index": i,
                            "url": url,
                            "source": source_id,
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

        return json.dumps(
            {
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "code_examples_stored": code_examples_stored,
                "content_length": len(result.markdown),
                "total_word_count": total_word_count,
                "source_id": source_id,
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", [])),
                },
            },
            indent=2,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
) -> str:
    """Intelligently crawl a URL based on its type and store content in Supabase."""

    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        if is_txt(url):
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({"success": False, "url": url, "error": "No URLs found in sitemap"}, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            crawl_results = await crawl_recursive_internal_links(
                crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent
            )
            crawl_type = "webpage"

        if not crawl_results:
            return json.dumps({"success": False, "url": url, "error": "No content found"}, indent=2)

        urls: List[str] = []
        chunk_numbers: List[int] = []
        contents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        chunk_count = 0
        source_content_map: Dict[str, str] = {}
        source_word_counts: Dict[str, int] = {}

        for doc in crawl_results:
            source_url = doc["url"]
            md = doc["markdown"]
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)

            parsed_url = urlparse(source_url)
            source_id = parsed_url.netloc or parsed_url.path

            if source_id not in source_content_map:
                source_content_map[source_id] = md[:5000]
                source_word_counts[source_id] = 0

            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)

                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = source_id
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

                source_word_counts[source_id] += meta.get("word_count", 0)
                chunk_count += 1

        url_to_full_document = {doc["url"]: doc["markdown"] for doc in crawl_results}

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            source_summary_args = [(source_id, content) for source_id, content in source_content_map.items()]
            source_summaries = list(
                executor.map(lambda args: extract_source_summary(args[0], args[1]), source_summary_args)
            )

        for (source_id, _), summary in zip(source_summary_args, source_summaries):
            word_count = source_word_counts.get(source_id, 0)
            update_source_info(supabase_client, source_id, summary, word_count)

        batch_size = 20
        add_documents_to_supabase(
            supabase_client,
            urls,
            chunk_numbers,
            contents,
            metadatas,
            url_to_full_document,
            batch_size=batch_size,
        )

        code_examples: List[str] = []
        if os.getenv("USE_AGENTIC_RAG", "false") == "true":
            code_urls: List[str] = []
            code_chunk_numbers: List[int] = []
            code_summaries: List[str] = []
            code_metadatas: List[Dict[str, Any]] = []

            for doc in crawl_results:
                source_url = doc["url"]
                md = doc["markdown"]
                code_blocks = extract_code_blocks(md)
                if not code_blocks:
                    continue

                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    summary_args = [(block["code"], block["context_before"], block["context_after"]) for block in code_blocks]
                    summaries = list(executor.map(process_code_example, summary_args))

                parsed_url = urlparse(source_url)
                source_id = parsed_url.netloc or parsed_url.path

                for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                    code_urls.append(source_url)
                    code_chunk_numbers.append(len(code_examples))
                    code_examples.append(block["code"])
                    code_summaries.append(summary)
                    code_metadatas.append(
                        {
                            "chunk_index": len(code_examples) - 1,
                            "url": source_url,
                            "source": source_id,
                            "char_count": len(block["code"]),
                            "word_count": len(block["code"].split()),
                        }
                    )

            if code_examples:
                add_code_examples_to_supabase(
                    supabase_client,
                    code_urls,
                    code_chunk_numbers,
                    code_examples,
                    code_summaries,
                    code_metadatas,
                    batch_size=batch_size,
                )

        return json.dumps(
            {
                "success": True,
                "url": url,
                "crawl_type": crawl_type,
                "pages_crawled": len(crawl_results),
                "chunks_stored": chunk_count,
                "code_examples_stored": len(code_examples),
                "sources_updated": len(source_content_map),
                "urls_crawled": [doc["url"] for doc in crawl_results][:5]
                + (["..."] if len(crawl_results) > 5 else []),
            },
            indent=2,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


def register_crawling_tools(mcp: FastMCP) -> None:
    """Register crawl tools against the provided FastMCP instance."""

    mcp.tool()(crawl_single_page)
    mcp.tool()(smart_crawl_url)
