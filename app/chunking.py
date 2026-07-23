"""Document parsing + chunking.

Primary path: Docling (layout-aware — preserves tables, headings, reading
order → much better chunks). Falls back to PyMuPDF page-text extraction if
Docling isn't available or fails on a file, so the pipeline still runs.
"""
from dataclasses import dataclass

from .config import settings


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class TextChunk:
    page: int
    chunk_index: int
    text: str


def _parse_with_docling(path: str) -> list[PageText]:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(path)
    doc = result.document

    # Group text items by their page number using provenance info.
    pages: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            continue
        page_no = 1
        prov = getattr(item, "prov", None)
        if prov:
            page_no = getattr(prov[0], "page_no", 1)
        pages.setdefault(page_no, []).append(text)

    return [PageText(page=p, text="\n".join(t)) for p, t in sorted(pages.items())]


def _parse_with_pymupdf(path: str) -> list[PageText]:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append(PageText(page=i, text=page.get_text("text")))
    return pages


def parse_document(path: str) -> list[PageText]:
    try:
        pages = _parse_with_docling(path)
        if pages:
            return pages
    except Exception:
        pass
    return _parse_with_pymupdf(path)


def chunk_pages(pages: list[PageText]) -> list[TextChunk]:
    """Sliding-window chunking over words, per page, so every chunk keeps a
    single source page number for citations."""
    size = settings.chunk_size_words
    overlap = settings.chunk_overlap_words
    step = max(1, size - overlap)

    chunks: list[TextChunk] = []
    idx = 0
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        for start in range(0, len(words), step):
            window = words[start : start + size]
            if not window:
                continue
            chunks.append(
                TextChunk(page=page.page, chunk_index=idx, text=" ".join(window))
            )
            idx += 1
            if start + size >= len(words):
                break
    return chunks
