"""
chunker.py
----------
Splits raw ingested documents into indexable chunks.

Recipe documents have natural structure (a title, then "Ingredients",
then "Instructions"/"Directions"). We try to detect that structure so a
whole recipe stays together as one chunk -- that matters a lot for RAG
quality here, because a recipe with its ingredient list separated from
its steps is much less useful to retrieve.

If a document doesn't look recipe-structured (e.g. a long blog post with
narrative text), we fall back to a sliding-window text splitter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .ingest import RawDocument

# Matches lines like "Ingredients", "INGREDIENTS:", "What you'll need"
INGREDIENTS_HEADER = re.compile(
    r"^\s*(ingredients|what you.?ll need|you will need)\s*:?\s*$", re.IGNORECASE
)

MAX_CHUNK_CHARS = 3000
FALLBACK_CHUNK_CHARS = 1200
FALLBACK_OVERLAP = 150


@dataclass
class RecipeChunk:
    chunk_id: str
    text: str
    source: str
    title: str
    metadata: dict = field(default_factory=dict)


def _guess_title(block: str, fallback: str) -> str:
    for line in block.splitlines():
        line = line.strip().strip("#*").strip()
        if line and not INGREDIENTS_HEADER.match(line) and len(line) < 90:
            return line
    return fallback


def _split_on_recipe_boundaries(text: str) -> List[str]:
    """
    Split a multi-recipe document (e.g. a cookbook export) into blocks,
    using "Ingredients" headers as anchors. Each block runs from just
    before one title-before-Ingredients to just before the next.
    """
    lines = text.splitlines()
    ingredient_line_idxs = [
        i for i, line in enumerate(lines) if INGREDIENTS_HEADER.match(line)
    ]

    if len(ingredient_line_idxs) <= 1:
        return [text]

    # Walk backwards from each "Ingredients" line to find its title line
    # (first non-blank line above it that isn't itself a header keyword).
    boundaries = []
    for idx in ingredient_line_idxs:
        title_idx = idx
        for back in range(idx - 1, -1, -1):
            if lines[back].strip():
                title_idx = back
            else:
                if title_idx != idx:
                    break
        boundaries.append(title_idx)

    boundaries = sorted(set(boundaries))
    blocks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        block = "\n".join(lines[start:end]).strip()
        if block:
            blocks.append(block)
    return blocks


def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_document(doc: RawDocument) -> List[RecipeChunk]:
    """Turn one RawDocument into one or more RecipeChunks."""
    blocks = _split_on_recipe_boundaries(doc.text)

    chunks: List[RecipeChunk] = []
    for block_idx, block in enumerate(blocks):
        title = _guess_title(block, fallback=f"{doc.source} (part {block_idx + 1})")

        if len(block) <= MAX_CHUNK_CHARS:
            pieces = [block]
        else:
            pieces = _sliding_window(block, FALLBACK_CHUNK_CHARS, FALLBACK_OVERLAP)

        for piece_idx, piece in enumerate(pieces):
            chunk_id = f"{doc.source}::{block_idx}::{piece_idx}"
            chunks.append(
                RecipeChunk(
                    chunk_id=chunk_id,
                    text=piece,
                    source=doc.source,
                    title=title,
                    metadata={"source": doc.source, "title": title},
                )
            )
    return chunks


def chunk_documents(docs: List[RawDocument]) -> List[RecipeChunk]:
    all_chunks: List[RecipeChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
