"""Crawl-related MCP tools implemented via Jina's lightweight fetch proxy."""

from typing import Any, Dict, List, Set

import asyncio
import concurrent.futures
import json
import os
import re
from urllib.parse import urljoin, urlparse, urldefrag
from xml.etree import ElementTree

import requests
from mcp.server.fastmcp import Context, FastMCP

from text_processing import extract_section_info, process_code_example, smart_chunk_markdown
from utils import (
    add_code_examples_to_supabase,
    add_documents_to_supabase,
    extract_code_blocks,
    extract_source_summary,
    update_source_info,
)


JINA_PROXY_BASE = os.getenv("JINA_RAG_BASE", "https://r.jina.ai")
JINA_API_KEY = os.getenv("JINA_API_KEY")
REQUEST_TIMEOUT = float(os.getenv("JINA_FETCH_TIMEOUT", "60"))
HTML_USER_AGENT = (
    os.getenv("CRAWLER_USER_AGENT")
    or "Mozilla/5.0 (compatible; Crawl4AI-RAG/1.0; +https://github.com/eliumusk/mcp-crawl4ai-rag)"
)


def is_sitemap(url: str) -> bool:
    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path


def is_txt(url: str) -> bool:
    return url.endswith(".txt")


def parse_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url, headers={"User-Agent": HTML_USER_AGENT}, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        return []

    try:
        tree = ElementTree.fromstring(resp.content)
        return [loc.text for loc in tree.findall(".//{*}loc") if loc.text]
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Error parsing sitemap XML: {exc}")
        return []


def _build_proxy_url(target_url: str) -> str:
    base = JINA_PROXY_BASE.rstrip("/")
    return f"{base}/{target_url}"


def fetch_with_jina(target_url: str) -> str:
    proxy_url = _build_proxy_url(target_url)
    headers = {"User-Agent": HTML_USER_AGENT}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"

    resp = requests.get(proxy_url, headers=headers, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Jina fetch failed for {target_url}: HTTP {resp.status_code}")
    return resp.text


async def fetch_markdown(target_url: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_with_jina, target_url)


def fetch_html_for_links(target_url: str) -> str:
    resp = requests.get(target_url, headers={"User-Agent": HTML_USER_AGENT}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def extract_internal_links(base_url: str, html: str) -> Set[str]:
    links = set()
    base = urlparse(base_url)
    for href in re.findall(r'href=["\'](.*?)["\']', html, flags=re.IGNORECASE):
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme.startswith("http") and parsed.netloc == base.netloc:
            links.add(absolute)
    return links


async def crawl_markdown_file(url: str) -> List[Dict[str, Any]]:
    try:
        markdown = await fetch_markdown(url)
        return [{"url": url, "markdown": markdown}]
    except Exception as exc:
        print(f"Failed to fetch {url}: {exc}")
        return []


async def crawl_batch(urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[Dict[str, Any]] = []

    async def fetch_one(target_url: str) -> None:
        async with semaphore:
            try:
                markdown = await fetch_markdown(target_url)
                if markdown:
                    results.append({"url": target_url, "markdown": markdown})
            except Exception as exc:
                print(f"Failed to fetch {target_url}: {exc}")

    await asyncio.gather(*(fetch_one(url) for url in urls))
    return results


async def crawl_recursive_internal_links(
    start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    visited = set()
    current_urls = {urldefrag(url)[0] for url in start_urls}
    results: List[Dict[str, Any]] = []

    for _ in range(max_depth):
        urls_to_fetch = [url for url in current_urls if url not in visited]
        if not urls_to_fetch:
            break

        batch_results = await crawl_batch(urls_to_fetch, max_concurrent=max_concurrent)
        results.extend(batch_results)
        new_urls: Set[str] = set()

        for url in urls_to_fetch:
            visited.add(url)
            try:
                html = fetch_html_for_links(url)
                new_urls |= extract_internal_links(url, html)
            except Exception as exc:
                print(f"Failed to extract links from {url}: {exc}")

        current_urls = new_urls

    return results


async def crawl_single_page(ctx: Context, url: str) -> str:
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        markdown = await fetch_markdown(url)
        if not markdown:
            return json.dumps({"success": False, "url": url, "error": "No content returned"}, indent=2)

        parsed_url = urlparse(url)
        source_id = parsed_url.netloc or parsed_url.path
        chunks = smart_chunk_markdown(markdown)

        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        total_word_count = 0

        for i, chunk in enumerate(chunks):
            urls.append(url)
            chunk_numbers.append(i)
            contents.append(chunk)
            meta = extract_section_info(chunk)
            meta["chunk_index"] = i
            meta["url"] = url
            meta["source"] = source_id
            meta["crawl_time"] = "jina_fetch"
            metadatas.append(meta)
            total_word_count += meta.get("word_count", 0)

        url_to_full_document = {url: markdown}
        source_summary = extract_source_summary(source_id, markdown[:5000])
        update_source_info(supabase_client, source_id, source_summary, total_word_count)
        add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)

        code_examples_stored = 0
        code_blocks = extract_code_blocks(markdown)
        if os.getenv("USE_AGENTIC_RAG", "false") == "true" and code_blocks:
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []

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
                "content_length": len(markdown),
                "total_word_count": total_word_count,
                "source_id": source_id,
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def smart_crawl_url(
    ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000
) -> str:
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        if is_txt(url):
            crawl_results = await crawl_markdown_file(url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({"success": False, "url": url, "error": "No URLs found in sitemap"}, indent=2)
            crawl_results = await crawl_batch(sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            crawl_results = await crawl_recursive_internal_links([url], max_depth=max_depth, max_concurrent=max_concurrent)
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
                meta["crawl_time"] = "jina_fetch"
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
            code_urls = []
            code_chunk_numbers = []
            code_summaries = []
            code_metadatas = []

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

                for block, summary in zip(code_blocks, summaries):
                    code_chunk_numbers.append(len(code_examples))
                    code_examples.append(block["code"])
                    code_summaries.append(summary)
                    code_urls.append(source_url)
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
    except Exception as exc:
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


def register_crawling_tools(mcp: FastMCP) -> None:
    mcp.tool()(crawl_single_page)
    mcp.tool()(smart_crawl_url)
