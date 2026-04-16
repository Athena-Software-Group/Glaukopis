"""Common data structures for athena_scrape."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(slots=True)
class UrlRecord:
    """Structured representation of a URL discovered during collection."""

    url_id: str
    url: str
    source_type: str
    source_link: str
    collected_at: str
    published: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        url: str,
        *,
        source_type: str,
        source_link: str,
        published: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "UrlRecord":
        """Create a :class:`UrlRecord` with a deterministic identifier."""

        url_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        collected_at = datetime.now(tz=timezone.utc).isoformat()
        return cls(
            url_id=url_id,
            url=url,
            source_type=source_type,
            source_link=source_link,
            collected_at=collected_at,
            published=published,
            metadata=metadata or {},
        )


@dataclass(slots=True)
class ContentRecord:
    """Fetched page content that can be post-processed."""

    url_id: str
    url: str
    source_type: str
    fetched_at: str
    status: int
    content_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessedRecord:
    """Structured artifact emitted by the MCQ post-processing stage."""

    url_id: str
    url: str
    source_type: str
    payload: Dict[str, Any]
    processed_at: str


