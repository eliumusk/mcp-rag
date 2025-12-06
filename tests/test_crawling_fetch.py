import asyncio

import pytest

from tools import crawling


def _run(coro):
    return asyncio.run(coro)


def test_fetch_content_with_fallback_uses_direct(monkeypatch):
    calls: list[str] = []

    class DummyResponse:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("error")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None):
            calls.append(url)
            if url.startswith(crawling.JINA_PROXY_BASE.rstrip("/")):
                return DummyResponse(422, "")
            return DummyResponse(200, "direct content")

    monkeypatch.setattr(crawling, "httpx", type("H", (), {"AsyncClient": DummyClient}))

    content, links = _run(crawling.fetch_content_with_fallback("https://example.com/doc"))
    assert content == "direct content"
    assert links == []
    assert len(calls) == 2


def test_fetch_content_with_fallback_raises_when_direct_fails(monkeypatch):
    class DummyResponse:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            raise RuntimeError("boom")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None):
            return DummyResponse(500, "")

    monkeypatch.setattr(crawling, "httpx", type("H", (), {"AsyncClient": DummyClient}))

    with pytest.raises(RuntimeError):
        _run(crawling.fetch_content_with_fallback("https://example.com/doc2"))


def test_parse_links_summary_extracts_urls():
    text = """Hello world

Links/Buttons:
- [Example](https://example.com/a)
- [](https://example.com/b)
- - https://example.com/c
"""
    body, links = crawling._parse_links_summary(text)
    assert "Links/Buttons" not in body
    assert links == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]


def test_parse_links_summary_handles_absence():
    body, links = crawling._parse_links_summary("No markers here")
    assert body == "No markers here"
    assert links == []
