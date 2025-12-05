from types import SimpleNamespace

import pytest

from tools import crawling


def test_fetch_content_with_fallback_uses_direct(monkeypatch):
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url.startswith(crawling.JINA_PROXY_BASE.rstrip("/")):
            return SimpleNamespace(
                status_code=422,
                text="",
                content=b"",
                raise_for_status=lambda: None,
            )
        return SimpleNamespace(
            status_code=200,
            text="direct content",
            content=b"direct content",
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(crawling, "requests", SimpleNamespace(get=fake_get))

    content, links = crawling.fetch_content_with_fallback("https://example.com/doc")
    assert content == "direct content"
    assert links == []
    assert len(calls) == 2  # proxy + direct fallback


def test_fetch_content_with_fallback_raises_when_direct_fails(monkeypatch):
    def fake_get(url, **kwargs):
        return SimpleNamespace(
            status_code=500,
            text="",
            content=b"",
            raise_for_status=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    monkeypatch.setattr(crawling, "requests", SimpleNamespace(get=fake_get))

    with pytest.raises(RuntimeError):
        crawling.fetch_content_with_fallback("https://example.com/doc2")


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
