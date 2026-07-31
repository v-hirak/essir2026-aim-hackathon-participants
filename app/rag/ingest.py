"""Load a PDF into the vector store.

    parse PDF (data/in) -> pages -> [chunk] -> embeddings -> Qdrant

By default there is no chunking (one vector per page) and embeddings come from a local
sentence-transformers model. Both are yours to improve (see chunking.py and embeddings.py).
"""

from __future__ import annotations

import uuid
from pathlib import Path

# from pypdf import PdfReader
from qdrant_client import models

import re
import unicodedata
import pymupdf

from langchain_core.documents import Document

from ..config import get_settings
from ..models import IngestResponse
from ..vectorstore.qdrant_store import get_store
from .chunking import chunk_pages
from .embeddings import get_embedder

# A fixed namespace so re-ingesting the same document overwrites its points
# (idempotent ids) instead of duplicating them.
_NAMESPACE = uuid.UUID("6f0d9b1e-3b7a-4c2e-9a1d-000000000000")


def _find_pdf(filename: str | None) -> Path:
    in_dir = Path(get_settings().in_dir)
    if filename:
        path = in_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"no such PDF: {path}")
        return path
    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no *.pdf found in {in_dir}/ — put your document there first")
    return pdfs[0]

def clean_block(text: str) -> str:
    """
    Normalize PDF text while preserving list structure.
    """

    # Convert ligatures: ﬁ -> fi, ﬂ -> fl, etc.
    text = unicodedata.normalize("NFKC", text)

    lines = [line.rstrip() for line in text.splitlines()]

    # Numbered lists, bullets, lettered lists
    list_pattern = re.compile(
        r"^\s*(\d+[\.\)]|[a-zA-Z][\.\)]|[-*•])\s+"
    )

    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned.append("\n")
            continue

        if list_pattern.match(stripped):
            cleaned.append("\n" + stripped)
            continue

        # Join wrapped lines within a paragraph
        if cleaned and not cleaned[-1].endswith("\n"):
            cleaned.append(" " + stripped)
        else:
            cleaned.append(stripped)

    text = "".join(cleaned)

    # Clean up excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pages(path: Path) -> list[str]:
    """Per-page text via pypdf.

    TODO(level-1): pypdf is fine for clean digital PDFs and poor on complex layout
      (two columns, tables, ligatures, math). If your citations won't match the
      document, your extractor is usually why. Try pdfplumber, PyMuPDF, Docling,
      GROBID or Marker and keep whichever reads your document best.
    """
    # reader = PdfReader(str(path))
    # return [(page.extract_text() or "") for page in reader.pages]

    # ---------------------------------------------------
    # Read PDF and create one LangChain Document per block
    # ---------------------------------------------------

    pdf = pymupdf.open(str(path))

    documents = []

    for page_num, page in enumerate(pdf):
        blocks = page.get_text("blocks")

        for block_num, block in enumerate(blocks):
            text = clean_block(block[4])

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path,
                        "page": page_num + 1,
                        "block": block_num,
                    },
                )
            )

    return documents


def ingest(filename: str | None = None, reset: bool = False) -> IngestResponse:
    settings = get_settings()
    embedder = get_embedder()
    store = get_store()

    path = _find_pdf(filename)
    pages = extract_pages(path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(f"{path.name} produced no text — is it a scanned/image PDF?")

    # Embed in batches. is_query=False marks these as documents ("passage:" for e5).
    vectors: list[list[float]] = []
    batch = 32
    for i in range(0, len(chunks), batch):
        texts = [c.text for c in chunks[i : i + batch]]
        vectors.extend(embedder.embed(texts, is_query=False))

    store.ensure_collection(dim=len(vectors[0]), reset=reset)

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, f"{path.name}:{c.index}")),
            vector=vec,
            payload={"text": c.text, "page": c.page, "source": path.name},
        )
        for c, vec in zip(chunks, vectors)
    ]
    store.upsert(points)

    return IngestResponse(
        document=path.name,
        pages=len(pages),
        chunks=len(chunks),
        collection=settings.qdrant_collection,
    )
