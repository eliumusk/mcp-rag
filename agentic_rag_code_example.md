# search_code_examples 本地示例

本文档用于演示如何通过 `ingest_local_files` 摄入本地 Markdown 文件，再使用 `search_code_examples` 工具检索长代码示例。

## 背景

- 业务需要：为 AI coding agent 提供 FastAPI + Supabase 的查询接口示例。
- 工具链：`ingest_local_files` → Supabase 向量库 → `search_code_examples`。
- 配置提醒：确保 `.env` 中的 `USE_AGENTIC_RAG=true`，否则代码示例不会被单独抽取。

## 代码片段

下面的代码演示了如何包装 Supabase 查询为一个异步的代码示例检索 API。内容超过 300 个字符，方便 `search_code_examples` 抽取。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException


@dataclass
class CodeRequest:
    query: str
    match_count: int = 4
    source_id: Optional[str] = None


class CodeExampleGateway:
    """轻量封装 Supabase 向量查询逻辑，用于 search_code_examples 测试。"""

    def __init__(self, supabase_client):
        self._client = supabase_client

    def fetch(self, request: CodeRequest) -> List[Dict[str, Any]]:
        payload = {
            "query": request.query,
            "match_count": request.match_count,
            "source_id": request.source_id,
        }
        response = self._client.rpc("search_code_examples", payload).execute()
        if getattr(response, "error", None):
            raise RuntimeError(f"search_code_examples failed: {response.error}")
        return response.data or []


router = APIRouter(prefix="/code-search", tags=["code-examples"])


@router.post("/")
async def find_examples(body: CodeRequest, gateway: CodeExampleGateway):
    examples = gateway.fetch(body)
    if not examples:
        raise HTTPException(status_code=404, detail="no matching examples")
    return {
        "query": body.query,
        "hits": [
            {
                "url": item.get("url"),
                "summary": item.get("summary"),
                "similarity": item.get("similarity"),
            }
            for item in examples
        ],
    }

@router.post("/")
async def find_examples(body: CodeRequest, gateway: CodeExampleGateway):
    examples = gateway.fetch(body)
    if not examples:
        raise HTTPException(status_code=404, detail="no matching examples")
    return {
        "query": body.query,
        "hits": [
            {
                "url": item.get("url"),
                "summary": item.get("summary"),
                "similarity": item.get("similarity"),
            }
            for item in examples
        ],
    }
@router.post("/")
async def find_examples(body: CodeRequest, gateway: CodeExampleGateway):
    examples = gateway.fetch(body)
    if not examples:
        raise HTTPException(status_code=404, detail="no matching examples")
    return {
        "query": body.query,
        "hits": [
            {
                "url": item.get("url"),
                "summary": item.get("summary"),
                "similarity": item.get("similarity"),
            }
            for item in examples
        ],
    }
```

## 使用流程

1. 将本文件路径（`tmp/agentic_rag_code_example.md`）传给 `ingest_local_files`，例如：
   ```json
   {
     "tool": "ingest_local_files",
     "file_paths": ["tmp/agentic_rag_code_example.md"],
     "recursive": false
   }
   ```
2. 等待摄入完成后，调用 `search_code_examples`：
   ```json
   {
     "tool": "search_code_examples",
     "query": "FastAPI Supabase search_code_examples",
     "match_count": 3
   }
   ```
3. 返回结果应包含上方代码片段的 URL、summary 与 similarity，验证工具链是否工作正常。
