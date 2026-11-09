"""
The text recipe used to embed a catalog service.

Matches how the original WoService vectors were built (WoSemanticSearch seeding,
verified exact against the stored vectors):
  titleEmb <- title
  descEmb  <- f"{description}. {keywords}"   (description alone when there are no keywords)

Every path that embeds catalog text must use this, so query-time matching sees one
consistent catalog.
"""

from __future__ import annotations


def catalog_texts(title: str | None, description: str | None, keywords: str | None) -> tuple[str, str]:
    """Return (title_text, description_text) to embed for a service."""
    t = (title or "").strip()
    d = (description or "").strip()
    k = (keywords or "").strip()
    return t, (f"{d}. {k}" if k else d)
