"""Loads product documentation into searchable chunks.

Layout (shared with the evals, see README):

    products/<product-id>/docs/*.md

Every markdown file is split into chunks at its headings (``#``, ``##``, ``###``). A chunk id looks like
``product-one/pricing.md#plans`` (product / file # slugified heading); text before the first ``##`` heading
becomes the ``overview`` chunk. Edit a file and the next search sees it, no restart needed.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# In docker-compose ./products is mounted at /products; locally it is <repo>/products.
PRODUCTS_DIR = Path(os.environ.get("PRODUCTS_DIR") or Path(__file__).resolve().parent.parent / "products")

Chunk = Dict[str, Any]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _doc_files() -> List[Tuple[str, Path]]:
    files = []
    if PRODUCTS_DIR.is_dir():
        for product_dir in sorted(p for p in PRODUCTS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")):
            for path in sorted((product_dir / "docs").glob("*.md")):
                files.append((product_dir.name, path))
    return files


def signature() -> Tuple:
    """Cheap fingerprint of the docs on disk; changes whenever a file is added, removed or edited."""
    return tuple((product, path.name, path.stat().st_mtime_ns) for product, path in _doc_files())


def _split(markdown: str, fallback_title: str) -> Tuple[str, List[Tuple[str, str]]]:
    """-> (document title, [(section heading, body), ...])"""
    title, section, lines, parts = fallback_title, "Overview", [], []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            parts.append((section, body))

    for line in markdown.splitlines():
        match = re.match(r"^(#{1,3})\s+(.*\S)\s*$", line)
        if not match:
            lines.append(line)
            continue
        flush()
        lines.clear()
        if len(match.group(1)) == 1:
            title, section = match.group(2), "Overview"
        else:
            section = match.group(2)
    flush()
    return title, parts


def load_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    for product, path in _doc_files():
        title, parts = _split(path.read_text(encoding="utf-8"), path.stem)
        seen: Dict[str, int] = {}
        for section, body in parts:
            anchor = slug(section)
            seen[anchor] = seen.get(anchor, 0) + 1
            if seen[anchor] > 1:
                anchor = "%s-%d" % (anchor, seen[anchor])
            chunks.append(
                {
                    "id": "%s/%s#%s" % (product, path.name, anchor),
                    "product": product,
                    "doc": path.name,
                    "doc_title": title,
                    "section": section,
                    "text": body,
                }
            )
    return chunks
