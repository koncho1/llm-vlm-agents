import pytest

from agent_project.evaluate import evaluate_retrieval, load_questions
from agent_project.retrieval import (
    DOCS_DIR,
    TOP_K,
    DocumentIndex,
    chunk_markdown,
    index_is_current,
    load_chunks,
)

DOC = """# Shoe Guide

## Suede

Keep suede dry.

Brush it when dry.

## Leather

Polish leather.
"""


def test_chunks_keep_source_title_and_heading() -> None:
    chunks = chunk_markdown("shoes", DOC)
    assert [c["id"] for c in chunks] == ["shoes#1", "shoes#2"]
    assert chunks[0]["source"] == "shoes"
    assert chunks[0]["title"] == "Shoe Guide"
    assert chunks[0]["heading"] == "Suede"
    assert chunks[0]["text"] == "Keep suede dry. Brush it when dry."


def test_long_sections_split_on_paragraphs() -> None:
    long_doc = "# T\n\n## H\n\n" + "\n\n".join(["word " * 100] * 3)
    chunks = chunk_markdown("long", long_doc)
    assert len(chunks) > 1
    assert all(c["heading"] == "H" for c in chunks)


def test_every_doc_chunk_has_a_unique_id() -> None:
    ids = [c["id"] for c in load_chunks(DOCS_DIR)]
    assert len(ids) == len(set(ids))


def test_committed_index_matches_the_documents() -> None:
    # Fails when a document changed without `uv run outfit-index`.
    assert index_is_current()


def test_stale_index_is_refused(tmp_path) -> None:
    (tmp_path / "doc.md").write_text(DOC, encoding="utf-8")
    with pytest.raises(RuntimeError, match="outfit-index"):
        DocumentIndex.load(docs_dir=tmp_path)


@pytest.fixture(scope="module")
def index() -> DocumentIndex:
    return DocumentIndex.load()


def test_search_returns_scored_chunks_with_sources(index) -> None:
    results = index.search("How do I wash a wool sweater?")
    assert len(results) == TOP_K
    assert results[0]["id"] == "fabric-care#1"
    assert results[0]["score"] >= results[-1]["score"]


def test_retrieval_finds_every_answerable_eval_question_in_top_k(index) -> None:
    rows = evaluate_retrieval(index, load_questions())
    misses = [r["id"] for r in rows if r["rank"] is not None and r["rank"] > TOP_K]
    assert misses == []
