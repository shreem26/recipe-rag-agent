"""
ingest.py
---------
Loads raw recipe documents (cookbooks, blogs, user notes) from disk.

Supported formats: .txt, .md, .pdf, .docx

Each loaded document becomes a `RawDocument`: the full text plus light
metadata (source filename). Splitting a document into individual recipes
happens later, in chunker.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@dataclass
class RawDocument:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _load_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def load_document(path: Path) -> RawDocument:
    """Load a single file into a RawDocument, based on its extension."""
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        text = _load_txt(path)
    elif ext == ".pdf":
        text = _load_pdf(path)
    elif ext == ".docx":
        text = _load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return RawDocument(text=text, source=path.name, metadata={"path": str(path)})


def load_directory(directory: str | Path) -> List[RawDocument]:
    """
    Walk a directory (recursively) and load every supported recipe document.
    Unsupported / unreadable files are skipped with a warning, so a single
    bad file never breaks the whole ingestion run.
    """
    directory = Path(directory)
    documents: List[RawDocument] = []

    if not directory.exists():
        return documents

    for root, _dirs, files in os.walk(directory):
        for filename in sorted(files):
            path = Path(root) / filename
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                documents.append(load_document(path))
            except Exception as exc:  # noqa: BLE001 - keep ingestion resilient
                print(f"[ingest] Skipping {path}: {exc}")

    return documents
