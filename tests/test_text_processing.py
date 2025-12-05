from text_processing import (
    collect_local_files,
    parse_file_paths_input,
    smart_chunk_markdown,
)


def test_parse_file_paths_input_handles_json_and_csv():
    assert parse_file_paths_input('["a.md", "b.md"]') == ["a.md", "b.md"]
    assert parse_file_paths_input("foo.md,bar.md") == ["foo.md", "bar.md"]


def test_smart_chunk_markdown_respects_boundaries():
    text = """# Title\n\nParagraph one. Paragraph two.\n\n```python\nprint('hi')\n```"""
    chunks = smart_chunk_markdown(text, chunk_size=30)
    # Ensure final chunk is the code block and text remains grouped
    assert len(chunks) >= 2
    assert "Title" in chunks[0]
    assert chunks[-1].startswith("```python")


def test_collect_local_files(tmp_path):
    file_a = tmp_path / "doc.md"
    file_a.write_text("content")
    file_b = tmp_path / "notes.txt"
    file_b.write_text("content")
    # Unsupported extension should be skipped
    (tmp_path / "skip.bin").write_text("binary")

    files = collect_local_files([str(tmp_path / "*.md"), str(tmp_path / "notes.txt")])
    paths = {str(path) for path in files}
    assert str(file_a.resolve()) in paths
    assert str(file_b.resolve()) in paths
