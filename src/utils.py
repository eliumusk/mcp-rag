"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
try:
    from supabase import create_client, Client
except ImportError:  # pragma: no cover - allows tests to run without Supabase installed
    create_client = None
    Client = Any  # type: ignore
from urllib.parse import urlparse
from pathlib import Path
import requests
import re
import time
import hashlib
from datetime import datetime, timezone, timedelta

JINA_API_URL = os.getenv("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
JINA_API_KEY = os.getenv("JINA_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3")
JINA_EMBEDDING_TASK = os.getenv("JINA_EMBEDDING_TASK", "text-matching")
JINA_RERANK_URL = os.getenv("JINA_RERANK_URL", "https://api.jina.ai/v1/rerank")
JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v3")
DEFAULT_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE")

EMBEDDING_VERSION = os.getenv("EMBEDDING_VERSION")
EMBEDDING_IDENTIFIER = EMBEDDING_VERSION or EMBEDDING_MODEL
EMBEDDING_CACHE_ENABLED = os.getenv("EMBEDDING_CACHE_ENABLED", "true") == "true"
EMBEDDING_CACHE_TTL_SECONDS = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60)))


def utcnow_iso() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def compute_content_hash(text: str) -> str:
    """Create a stable SHA256 hash for caching embeddings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _is_cache_entry_stale(refreshed_at: Optional[str]) -> bool:
    if EMBEDDING_CACHE_TTL_SECONDS <= 0:
        return False
    timestamp = _parse_iso_timestamp(refreshed_at)
    if not timestamp:
        return True
    age = datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)
    return age > timedelta(seconds=EMBEDDING_CACHE_TTL_SECONDS)


def fetch_embedding_cache(
    client: Client,
    tenant_id: str,
    content_hashes: List[str],
    model_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Fetch cached embeddings from Supabase."""
    if not EMBEDDING_CACHE_ENABLED or not content_hashes:
        return {}
    unique_hashes = list({h for h in content_hashes if h})
    if not unique_hashes:
        return {}
    try:
        response = (
            client.table("embedding_cache")
            .select("content_hash, embedding, refreshed_at, needs_refresh")
            .eq("tenant_id", tenant_id)
            .eq("model_name", model_name)
            .in_("content_hash", unique_hashes)
            .execute()
        )
        rows = response.data or []
        return {row["content_hash"]: row for row in rows}
    except Exception as exc:
        print(f"Error fetching embedding cache: {exc}")
        return {}


def store_embeddings_in_cache(
    client: Client,
    records: List[Dict[str, Any]],
) -> None:
    """Persist new embeddings to the cache table."""
    if not EMBEDDING_CACHE_ENABLED or not records:
        return
    try:
        client.table("embedding_cache").upsert(records).execute()
    except Exception as exc:
        print(f"Error storing embeddings in cache: {exc}")


def flag_cache_entry_for_refresh(
    client: Client,
    tenant_id: str,
    content_hash: str,
    model_name: str,
) -> None:
    """Mark cache entries as needing refresh for asynchronous processing."""
    if not EMBEDDING_CACHE_ENABLED:
        return
    try:
        (
            client.table("embedding_cache")
            .update({"needs_refresh": True})
            .eq("tenant_id", tenant_id)
            .eq("content_hash", content_hash)
            .eq("model_name", model_name)
            .execute()
        )
    except Exception as exc:
        print(f"Error flagging cache entry for refresh: {exc}")


def get_embeddings_with_cache(
    client: Client,
    texts: List[str],
    content_hashes: List[str],
    tenant_id: str,
    model_name: str,
    cache_context: str,
) -> List[List[float]]:
    """
    Return embeddings for texts, preferring cached values when available.
    """
    if not texts:
        return []

    if len(texts) != len(content_hashes):
        raise ValueError("texts and content_hashes must be the same length")

    if not EMBEDDING_CACHE_ENABLED:
        return create_embeddings_batch(texts)

    cache_map = fetch_embedding_cache(client, tenant_id, content_hashes, model_name)
    embeddings: List[Optional[List[float]]] = [None] * len(texts)
    missing_texts: List[str] = []
    missing_indices: List[int] = []

    for idx, content_hash in enumerate(content_hashes):
        cache_entry = cache_map.get(content_hash)
        if cache_entry and cache_entry.get("embedding"):
            if _is_cache_entry_stale(cache_entry.get("refreshed_at")) and not cache_entry.get("needs_refresh"):
                flag_cache_entry_for_refresh(client, tenant_id, content_hash, model_name)
            embeddings[idx] = cache_entry["embedding"]
        else:
            missing_texts.append(texts[idx])
            missing_indices.append(idx)

    hit_count = len(texts) - len(missing_texts)
    print(
        f"[EmbeddingCache] context={cache_context} tenant={tenant_id} model={model_name} "
        f"hits={hit_count} misses={len(missing_texts)}",
        flush=True,
    )

    if missing_texts:
        fresh_embeddings = create_embeddings_batch(missing_texts)
        now_iso = utcnow_iso()
        cache_records: List[Dict[str, Any]] = []
        for index, embedding in zip(missing_indices, fresh_embeddings):
            embeddings[index] = embedding
            cache_records.append(
                {
                    "tenant_id": tenant_id,
                    "content_hash": content_hashes[index],
                    "model_name": model_name,
                    "embedding": embedding,
                    "refreshed_at": now_iso,
                    "needs_refresh": False,
                    "metadata": {"context": cache_context},
                }
            )
        store_embeddings_in_cache(client, cache_records)

    # Replace any remaining None entries with zero vectors to avoid downstream errors
    return [embedding if embedding is not None else [0.0] * DEFAULT_EMBEDDING_DIM for embedding in embeddings]

def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")

    if create_client is None:
        raise ImportError("supabase-py is required to create a Supabase client. Install the dependencies first.")
    
    return create_client(url, key)

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call using Jina AI.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY must be set to create embeddings.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    for retry in range(max_retries):
        try:
            payload = {
                "model": EMBEDDING_MODEL,
                "task": JINA_EMBEDDING_TASK,
                "input": texts
            }
            response = requests.post(
                JINA_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            embeddings = [item["embedding"] for item in data.get("data", [])]
            
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} embeddings but received {len(embeddings)} from Jina API."
                )
            return embeddings
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0
                
                for i, text in enumerate(texts):
                    try:
                        single_payload = {
                            "model": EMBEDDING_MODEL,
                            "task": JINA_EMBEDDING_TASK,
                            "input": [text]
                        }
                        single_response = requests.post(
                            JINA_API_URL,
                            headers=headers,
                            json=single_payload,
                            timeout=60
                        )
                        single_response.raise_for_status()
                        single_data = single_response.json()
                        embeddings.append(single_data["data"][0]["embedding"])
                        successful_count += 1
                    except Exception as individual_error:
                        print(f"Failed to create embedding for text {i}: {individual_error}")
                        embeddings.append([0.0] * DEFAULT_EMBEDDING_DIM)
                
                print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                return embeddings

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using the configured embedding provider.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * DEFAULT_EMBEDDING_DIM
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * DEFAULT_EMBEDDING_DIM


def call_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 200
) -> str:
    """
    Call the configured chat completion endpoint (OpenRouter compatible).
    """
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY must be set for LLM calls.")
    
    if not LLM_API_BASE:
        raise ValueError("LLM_API_BASE must be set to call the configured LLM.")
    
    model_choice = model or os.getenv("MODEL_CHOICE")
    if not model_choice:
        raise ValueError("MODEL_CHOICE must be set to call the LLM.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    
    payload = {
        "model": model_choice,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = requests.post(
        f"{LLM_API_BASE.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    data = response.json()
    
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise ValueError(f"Unexpected response format from LLM: {data}")


def jina_rerank_documents(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Jina's rerank API.
    """
    if not documents:
        return []
    
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY must be set to use reranking.")
    
    payload = {
        "model": model or JINA_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n or len(documents),
        "return_documents": False
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    
    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"Error calling Jina rerank API: {e}")
        return []


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pypdf is required to process PDF files. Install it with `pip install pypdf`."
        ) from e
    
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages).strip()


def load_local_document(file_path: str) -> str:
    """
    Load local file content, supporting Markdown, text, and PDF files.
    """
    path_obj = Path(file_path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = path_obj.suffix.lower()
    if suffix in {".md", ".markdown", ".txt", ".rst", ".html"}:
        with open(path_obj, "r", encoding="utf-8") as f:
            return f.read()
    elif suffix == ".pdf":
        return extract_text_from_pdf(path_obj)
    else:
        # Attempt to read as UTF-8 text
        with open(path_obj, "r", encoding="utf-8") as f:
            return f.read()

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        context = call_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            model=model_choice,
            temperature=0.3,
            max_tokens=200
        )
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

def add_documents_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20,
    tenant_id: Optional[str] = None,
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    tenant = tenant_id or os.getenv("TENANT_ID", "default")

    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Delete existing records for these URLs in a single operation
    try:
        if unique_urls:
            # Use the .in_() filter to delete all records with matching URLs
            client.table("crawled_pages").delete().eq("tenant_id", tenant).in_("url", unique_urls).execute()
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        for url in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("tenant_id", tenant).eq("url", url).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")
                # Continue with the next URL even if one fails
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        
        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))
            
            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                                for idx, arg in enumerate(process_args)}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])
            
            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}")
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents
        
        # Create or reuse embeddings using the cache
        batch_hashes = [compute_content_hash(text) for text in contextual_contents]
        batch_embeddings = get_embeddings_with_cache(
            client=client,
            texts=contextual_contents,
            content_hashes=batch_hashes,
            tenant_id=tenant,
            model_name=EMBEDDING_IDENTIFIER,
            cache_context="documents",
        )
        
        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(contextual_contents[j])
            
            # Extract source_id, allowing metadata override
            meta_source = batch_metadatas[j].get("source") if isinstance(batch_metadatas[j], dict) else None
            parsed_url = urlparse(batch_urls[j])
            source_id = meta_source or parsed_url.netloc or parsed_url.path
            
            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": contextual_contents[j],  # Store original content
                "metadata": {
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                },
                "source_id": source_id,  # Add source_id field
                "embedding": batch_embeddings[j],  # Use embedding from contextual content
                "content_hash": batch_hashes[j],
                "embedding_model": EMBEDDING_IDENTIFIER,
                "embedding_cached_at": utcnow_iso(),
                "tenant_id": tenant,
            }
            
            batch_data.append(data)
        
        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                client.table("crawled_pages").insert(batch_data).execute()
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into Supabase (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.table("crawled_pages").insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {record['url']}: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")

def search_documents(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)
    tenant = tenant_id or os.getenv("TENANT_ID", "default")
    
    # Execute the search using the match_crawled_pages function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {
            'tenant_filter': tenant,
            'query_embedding': query_embedding,
            'match_count': match_count,
        }
        
        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params['filter'] = filter_metadata  # Pass the dictionary directly, not JSON-encoded
        
        result = client.rpc('match_crawled_pages', params).execute()
        
        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def extract_code_blocks(markdown_content: str, min_length: int = 100) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    
    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)
        
    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []
    
    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")
    
    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3
    
    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]
        
        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]
        
        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and not ' ' in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()
        
        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue
        
        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 100)
        context_before = markdown_content[context_start:start_pos].strip()
        
        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 100)
        context_after = markdown_content[end_pos + 3:context_end].strip()
        
        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })
        
        # Move to next pair (skip the closing backtick we just processed)
        i += 2
    
    return code_blocks


def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
    
    try:
        summary = call_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            model=model_choice,
            temperature=0.3,
            max_tokens=100
        )
        
        return summary.strip()
    
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


def add_code_examples_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20,
    tenant_id: Optional[str] = None,
):
    """
    Add code examples to the Supabase code_examples table in batches.
    
    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return
        
    tenant = tenant_id or os.getenv("TENANT_ID", "default")
        
    # Delete existing records for these URLs
    unique_urls = list(set(urls))
    for url in unique_urls:
        try:
            client.table('code_examples').delete().eq('tenant_id', tenant).eq('url', url).execute()
        except Exception as e:
            print(f"Error deleting existing code examples for {url}: {e}")
    
    # Process in batches
    total_items = len(urls)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_texts = []
        
        # Create combined texts for embedding (code + summary)
        for j in range(i, batch_end):
            combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
            batch_texts.append(combined_text)
        
        batch_hashes = [compute_content_hash(text) for text in batch_texts]
        embeddings = get_embeddings_with_cache(
            client=client,
            texts=batch_texts,
            content_hashes=batch_hashes,
            tenant_id=tenant,
            model_name=EMBEDDING_IDENTIFIER,
            cache_context="code_examples",
        )
        
        # Prepare batch data
        batch_data = []
        for j, embedding in enumerate(embeddings):
            idx = i + j
            
            # Extract source_id from URL
            parsed_url = urlparse(urls[idx])
            meta_source = None
            try:
                meta_source = metadatas[idx].get("source")
            except Exception:
                meta_source = None
            source_id = meta_source or parsed_url.netloc or parsed_url.path
            
            batch_data.append({
                'url': urls[idx],
                'chunk_number': chunk_numbers[idx],
                'content': code_examples[idx],
                'summary': summaries[idx],
                'metadata': metadatas[idx],  # Store as JSON object, not string
                'source_id': source_id,
                'embedding': embedding,
                'content_hash': batch_hashes[j],
                'embedding_model': EMBEDDING_IDENTIFIER,
                'embedding_cached_at': utcnow_iso(),
                'tenant_id': tenant
            })
        
        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                client.table('code_examples').insert(batch_data).execute()
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into Supabase (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.table('code_examples').insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {record['url']}: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")
        print(f"Inserted batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} code examples")


def update_source_info(client: Client, source_id: str, summary: str, word_count: int, tenant_id: Optional[str] = None):
    """
    Update or insert source information in the sources table.
    
    Args:
        client: Supabase client
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    try:
        tenant = tenant_id or os.getenv("TENANT_ID", "default")
        # Try to update existing source
        result = client.table('sources').update({
            'summary': summary,
            'total_word_count': word_count,
            'updated_at': 'now()'
        }).eq('source_id', source_id).eq('tenant_id', tenant).execute()
        
        # If no rows were updated, insert new source
        if not result.data:
            client.table('sources').insert({
                'source_id': source_id,
                'summary': summary,
                'total_word_count': word_count,
                'tenant_id': tenant
            }).execute()
            print(f"Created new source: {source_id}")
        else:
            print(f"Updated source: {source_id}")
            
    except Exception as e:
        print(f"Error updating source {source_id}: {e}")


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.
    
    This function relies on the configured LLM (OpenRouter-compatible) to summarize the source content.
    
    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"
    
    if not content or len(content.strip()) == 0:
        return default_summary
    
    # Get the model choice from environment variables
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content
    
    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
    
    try:
        summary = call_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                {"role": "user", "content": prompt}
            ],
            model=model_choice,
            temperature=0.3,
            max_tokens=150
        )
        
        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    except Exception as e:
        print(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
        return default_summary


def search_code_examples(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for code examples in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results
        
    Returns:
        List of matching code examples
    """
    # Create a more descriptive query for better embedding match
    # Since code examples are embedded with their summaries, we should make the query more descriptive
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    
    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)
    tenant = tenant_id or os.getenv("TENANT_ID", "default")
    
    # Execute the search using the match_code_examples function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {
            'tenant_filter': tenant,
            'query_embedding': query_embedding,
            'match_count': match_count,
        }
        
        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params['filter'] = filter_metadata
            
        # Add source filter if provided
        if source_id:
            params['source_filter'] = source_id
        
        result = client.rpc('match_code_examples', params).execute()
        
        return result.data
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []
