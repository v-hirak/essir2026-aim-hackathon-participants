"""Split each PDF page into overlapping, page-aware retrieval units."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import get_settings


@dataclass
class Chunk:
    text: str
    page: int      # 1-indexed
    index: int     # position within the document


def chunk_pages(pages: list[str]) -> list[Chunk]:
    """Split on a nearby paragraph, sentence, or word boundary with overlap."""
    settings = get_settings()
    chunks: list[Chunk] = []
    index = 0

    for page, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        start = 0
        while text and start < len(text):
            end = min(start + settings.chunk_size, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind("\n\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind(" ", start, end),
                )
                if boundary > start + settings.chunk_size // 2:
                    end = boundary + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, page=page, index=index))
                index += 1
            if end == len(text):
                break
            start = max(end - settings.chunk_overlap, start + 1)

    return chunks
