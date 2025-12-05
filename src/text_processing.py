"""Shared helpers for chunking, metadata extraction, and file expansion."""
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import glob
import json
import re

from utils import generate_code_example_summary


SUPPORTED_LOCAL_EXTENSIONS = {".md", ".markdown", ".txt", ".rst", ".html", ".pdf"}


def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks while respecting basic structural boundaries."""

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extract light-weight metadata about a markdown chunk."""

    headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
    header_str = "; ".join([f"{h[0]} {h[1]}" for h in headers]) if headers else ""
    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split()),
    }


def process_code_example(args: Sequence[str]) -> str:
    """Process a single code example to generate its summary."""

    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)


def parse_file_paths_input(file_paths: str) -> List[str]:
    """Parse newline/comma separated or JSON encoded file paths."""

    if not file_paths:
        return []

    cleaned_paths: List[str] = []
    try:
        parsed = json.loads(file_paths)
        if isinstance(parsed, str):
            cleaned_paths.append(parsed)
        elif isinstance(parsed, list):
            cleaned_paths.extend([str(item) for item in parsed])
    except json.JSONDecodeError:
        for part in re.split(r"[\n,]+", file_paths):
            part = part.strip()
            if part:
                cleaned_paths.append(part)

    return cleaned_paths


def collect_local_files(raw_paths: List[str], recursive: bool = False) -> List[Path]:
    """Expand provided raw paths (files/dirs/globs) into concrete files."""

    collected: List[Path] = []
    seen = set()

    for raw in raw_paths:
        if not raw:
            continue
        has_wildcard = any(char in raw for char in ("*", "?", "[", "]"))
        if has_wildcard:
            candidates = [Path(p).expanduser() for p in glob.glob(raw, recursive=recursive)]
        else:
            candidates = [Path(raw).expanduser()]

        for candidate in candidates:
            if not candidate.exists():
                continue

            if candidate.is_file():
                if candidate.suffix.lower() in SUPPORTED_LOCAL_EXTENSIONS or not candidate.suffix:
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        collected.append(resolved)
                        seen.add(resolved)
            elif candidate.is_dir():
                iterator = candidate.rglob("*") if recursive else candidate.glob("*")
                for child in iterator:
                    if child.is_file():
                        suffix = child.suffix.lower()
                        if suffix in SUPPORTED_LOCAL_EXTENSIONS or not suffix:
                            resolved_child = child.resolve()
                            if resolved_child not in seen:
                                collected.append(resolved_child)
                                seen.add(resolved_child)

    return collected
