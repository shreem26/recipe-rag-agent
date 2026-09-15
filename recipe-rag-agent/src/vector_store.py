"""
vector_store.py
----------------
A small, dependency-light vector store for recipe chunks.

Design choice: this uses TF-IDF + cosine similarity (scikit-learn) rather
than a neural embedding model. For a recipe knowledge base, queries and
documents share a lot of literal vocabulary ("sugar-free", "chocolate",
"gluten-free", ingredient names), so TF-IDF retrieval works well while
staying 100% offline -- no model weights to download, no GPU, instant
startup, fully deterministic and easy to unit test.

Swapping in a semantic embedding backend (e.g. Chroma's default embedding
function, sentence-transformers, or the Voyage/OpenAI embeddings APIs) is
a drop-in change: keep the same add_chunks()/search() interface and swap
out _rebuild_index()/search()'s internals.

The index is persisted to disk (as a pickle of chunk records) so it
survives restarts; the TF-IDF matrix itself is rebuilt in memory on load,
which keeps persistence robust across scikit-learn versions.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import RecipeChunk

DEFAULT_DB_PATH = "vector_index"


class RecipeVectorStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._data_path = self.db_path / "chunks.pkl"

        self._chunks: Dict[str, dict] = {}
        self._ids: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

        self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        if self._data_path.exists():
            with open(self._data_path, "rb") as f:
                self._chunks = pickle.load(f)
            self._rebuild_index()

    def _persist(self) -> None:
        with open(self._data_path, "wb") as f:
            pickle.dump(self._chunks, f)

    def reset(self) -> None:
        """Wipe the index so a fresh one can be built."""
        self._chunks = {}
        self._ids = []
        self._vectorizer = None
        self._matrix = None
        if self._data_path.exists():
            self._data_path.unlink()

    # -- indexing ----------------------------------------------------------

    def _rebuild_index(self) -> None:
        if not self._chunks:
            self._vectorizer = None
            self._matrix = None
            self._ids = []
            return

        self._ids = list(self._chunks.keys())
        texts = [self._chunks[i]["text"] for i in self._ids]

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def count(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: List[RecipeChunk], batch_size: int = 100) -> None:
        if not chunks:
            return
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = {
                "text": chunk.text,
                "source": chunk.source,
                "title": chunk.title,
            }
        self._rebuild_index()
        self._persist()

    # -- search --------------------------------------------------------

    def search(self, query: str, k: int = 5, where: Optional[dict] = None) -> List[dict]:
        """
        Return the top-k chunks most relevant to `query`, each as a dict
        with text / source / title / distance (lower distance = closer).
        """
        if not self._chunks or self._vectorizer is None:
            return []

        k = min(k, len(self._chunks))
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix)[0]

        ranked = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )[:k]

        hits = []
        for idx in ranked:
            score = similarities[idx]
            if score <= 0:
                continue
            chunk_id = self._ids[idx]
            record = self._chunks[chunk_id]
            hits.append(
                {
                    "text": record["text"],
                    "source": record["source"],
                    "title": record["title"],
                    "distance": round(1 - float(score), 4),
                }
            )
        return hits
