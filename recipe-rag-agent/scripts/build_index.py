"""
Rebuild the recipe vector index from a folder of documents.

Usage:
    python scripts/build_index.py                     # indexes data/sample_recipes
    python scripts/build_index.py path/to/my/recipes   # indexes a custom folder
    python scripts/build_index.py --reset              # wipes the index first
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_documents  # noqa: E402
from src.ingest import load_directory  # noqa: E402
from src.vector_store import RecipeVectorStore  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reset = "--reset" in sys.argv[1:]
    source_dir = args[0] if args else "data/sample_recipes"

    print(f"[build_index] Loading documents from: {source_dir}")
    docs = load_directory(source_dir)
    print(f"[build_index] Loaded {len(docs)} document(s)")

    chunks = chunk_documents(docs)
    print(f"[build_index] Produced {len(chunks)} chunk(s)")

    store = RecipeVectorStore()
    if reset:
        print("[build_index] Resetting existing index...")
        store.reset()

    store.add_chunks(chunks)
    print(f"[build_index] Index now contains {store.count()} chunk(s). Done.")


if __name__ == "__main__":
    main()
