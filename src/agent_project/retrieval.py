"""Local document index: markdown sections -> FastEmbed vectors -> cosine search."""

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from agent_project.tools import DATA_DIR

ROOT_DIR = DATA_DIR.parent
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = ROOT_DIR / ".cache" / "fastembed"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_CHUNK_CHARS = 900
TOP_K = 4


def chunk_markdown(source: str, text: str) -> list[dict[str, Any]]:
    """One chunk per `## ` section; long sections split on paragraph breaks."""
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source
    chunks: list[dict[str, Any]] = []
    for section in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = section.partition("\n")
        paragraphs = [" ".join(p.split()) for p in body.split("\n\n") if p.strip()]
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) > MAX_CHUNK_CHARS:
                chunks.append(_chunk(source, title, heading, current, len(chunks)))
                current = ""
            current = f"{current} {paragraph}".strip()
        if current:
            chunks.append(_chunk(source, title, heading, current, len(chunks)))
    return chunks


def _chunk(source: str, title: str, heading: str, text: str, n: int) -> dict:
    return {
        "id": f"{source}#{n + 1}",
        "source": source,
        "title": title,
        "heading": heading.strip(),
        "text": text,
    }


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[dict[str, Any]]:
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
    return chunks


def doc_hashes(docs_dir: Path = DOCS_DIR) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(docs_dir.glob("*.md"))
    }


@lru_cache(maxsize=1)
def _embedder():
    # Windows without Developer Mode cannot symlink; the cache still works.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from fastembed import TextEmbedding

    return TextEmbedding(MODEL_NAME, cache_dir=str(CACHE_DIR))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def build_index(
    docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR
) -> dict[str, Any]:
    chunks = load_chunks(docs_dir)
    passages = [f"{c['title']}: {c['heading']}\n{c['text']}" for c in chunks]
    vectors = _normalize(np.array(list(_embedder().passage_embed(passages))))
    manifest = {
        "model": MODEL_NAME,
        "chunk_count": len(chunks),
        "docs": doc_hashes(docs_dir),
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", vectors.astype(np.float32))
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def index_is_current(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR) -> bool:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest["model"] == MODEL_NAME and manifest["docs"] == doc_hashes(docs_dir)


class DocumentIndex:
    def __init__(self, chunks: list[dict[str, Any]], vectors: np.ndarray) -> None:
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def load(
        cls, docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR
    ) -> "DocumentIndex":
        if not index_is_current(docs_dir, index_dir):
            raise RuntimeError(
                "The document index is missing or out of date. "
                "Run `uv run outfit-index` to rebuild it."
            )
        chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        return cls(chunks, np.load(index_dir / "embeddings.npy"))

    def search(self, query: str, k: int = TOP_K) -> list[dict[str, Any]]:
        query_vector = _normalize(np.array(list(_embedder().query_embed([query]))))
        scores = self.vectors @ query_vector[0]
        best = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in best]


def main() -> None:
    manifest = build_index()
    print(
        f"Indexed {manifest['chunk_count']} chunks from {len(manifest['docs'])} "
        f"documents with {manifest['model']} into {INDEX_DIR}"
    )


if __name__ == "__main__":
    main()
