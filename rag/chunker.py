"""
Text chunking for RAG ingestion, splits documents into overlapping chunks suitable for embedding.

Sentence-aware sliding window. 
A naive 800-char split would cut mid-sentence constantly, breaking semantic units. 
We split at sentence boundaries and then fill chunks sentence by sentence. 
When a chunk reaches capacity, the last N sentences (totalling ~150 chars) are carried over 
 as overlap to the next chunk, ensuring a question whose answer spans a boundary retrieves 
 at least one chunk containing the full context.

strategy for chunking:-
We use sentence-aware chunking with a sliding window:
  1. Split text into sentences (using a simple sentence boundary regex)
  2. Accumulate sentences into a chunk until the character limit is reached
  3. Each new chunk overlaps with the previous by 'overlap_chars' characters
     This ensures context isn't lost at chunk boundaries
Why sentence-aware rather than naive character splits:
  - Naive splits cut mid-sentence, breaking semantic units
  - Sentence boundaries are natural semantic boundaries
  - Overlap ensures a query whose answer spans a boundary still retrieves it

chunk metadata:-
Each chunk carries:
  - chunk_index     : position within the document
  - total_chunks    : total number of chunks from this document
  - char_start      : character offset in the original text (for debugging)
  - document_id     : ID of the parent document
  - source_path     : original file path
  - title           : document title
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    total_chunks: int       # filled in after all chunks are created
    document_id: str
    char_start: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Unique ID for this chunk within the vector store."""
        return f"{self.document_id}::chunk{self.chunk_index}"


# pre-compiled sentence boundary pattern,
# splits on: . ! ? followed by whitespace (not mid-abbreviation like "U.S.")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    document_id: str,
    chunk_chars: int = 800,
    overlap_chars: int = 150,
    metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks.

    Args:
        text:          The full document text to chunk.
        document_id:   Parent document ID (used to build chunk IDs).
        chunk_chars:   Target maximum characters per chunk.
        overlap_chars: Characters of overlap between consecutive chunks.
        metadata:      Extra metadata copied into every chunk.

    Returns:
        List of TextChunk objects, ordered by position.
    """
    text = text.strip()
    if not text:
        return []

    base_meta = metadata or {}

    # split into sentences
    sentences = _SENTENCE_END.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    # build chunks by accumulating sentences
    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_chars = 0
    char_offset = 0
    chunk_start_offset = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1    # +1 for space separator

        if current_chars + sentence_len > chunk_chars and current_sentences:
            # flush the current chunk
            chunk_text_str = " ".join(current_sentences)
            chunks.append(TextChunk(
                text=chunk_text_str,
                chunk_index=len(chunks),
                total_chunks=0,             # filled in below
                document_id=document_id,
                char_start=chunk_start_offset,
                metadata=dict(base_meta),
            ))

            # compute overlap: take sentences from the end of current chunk
            # that together amount to ~overlap_chars characters
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > overlap_chars:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1

            current_sentences = overlap_sentences + [sentence]
            current_chars = sum(len(s) + 1 for s in current_sentences)
            chunk_start_offset = char_offset - overlap_len
        else:
            current_sentences.append(sentence)
            current_chars += sentence_len

        char_offset += sentence_len

    # flush the final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(TextChunk(
            text=chunk_text_str,
            chunk_index=len(chunks),
            total_chunks=0,
            document_id=document_id,
            char_start=chunk_start_offset,
            metadata=dict(base_meta),
        ))

    # fill in total_chunks
    total = len(chunks)
    for c in chunks:
        c.total_chunks = total

    return chunks