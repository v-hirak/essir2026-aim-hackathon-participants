"""Turn PDF pages into the units you index.

**Chunking is OFF by default.** Out of the box each page becomes exactly one vector — no
splitting, no overlap. That is the simplest thing that runs, and it is deliberately weak:
a whole page is often too long for the embedding model (it gets truncated) and too coarse
to retrieve precisely.

Implementing real chunking is one of the first things that will improve your Level-1 scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..config import get_settings


@dataclass
class Chunk:
    text: str
    page: int      # 1-indexed
    index: int     # position within the document
    block: int     # block within a page, 0-indexed


def chunk_pages(pages: list[str]) -> list[Chunk]:
    """Default: one chunk per page (no chunking).

    TODO(level-1): THIS IS WHERE CHUNKING GOES, and right now there is none. Split each
      page into retrievable units — by tokens, by sentences/paragraphs, or structure-aware
      (headings, tables). Keep the correct `page` on each piece so citations still line up.
      Good chunking is usually the single biggest win for retrieval quality.
    TODO(level-3): a flat chunk loses where it sits in the document. Section titles, or a
      small/large ("parent") hierarchy, help a lot with whole-document questions.

    The settings `chunk_size` / `chunk_overlap` exist for when you implement this — they are
    unused by the baseline.
    """
    # chunks: list[Chunk] = []
    # idx = 0
    # for page_no, text in enumerate(pages, start=1):
    #     text = text.strip()
    #     if not text:
    #         continue
    #     chunks.append(Chunk(text=text, page=page_no, index=idx))
    #     idx += 1
    # return chunks

    # ---------------------------------------------------
    # Split only large blocks
    # ---------------------------------------------------

    s = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        # separators=[
        #     "\n\n",
        #     "\n",
        #     ". ",
        #     " ",
        #     "",
        # ],
        chunk_size=s.CHUNK_SIZE,
        chunk_overlap=s.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(pages)

    final_chunks: list[Chunk] = []
    idx = 0
    for chunk in chunks:
        final_chunks.append(Chunk(text=chunk.page_content, page=chunk.metadata["page"], bloc=chunk.metadat["block"], index=idx))
        idx += 1

    return final_chunks

    # print(f"Documents: {len(documents)}")
    # print(f"Chunks: {len(chunks)}")

    # # Inspect a few chunks
    # for chunk in chunks[:3]:
    #     print("=" * 80)
    #     print(chunk.metadata)
    #     print(chunk.page_content[:500])
