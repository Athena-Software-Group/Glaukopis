"""Parser for CISA advisory content."""
from __future__ import annotations

from typing import Any, Dict, Optional

import bs4


SECTION_HEADING_TAGS = {"h1", "h2", "h3"}


def parse_cisa(html: str) -> Optional[Dict[str, Any]]:
    soup = bs4.BeautifulSoup(html, "html.parser")

    title_node = soup.find("h1") or soup.find("title")
    if not title_node:
        return None
    title = title_node.get_text(" ", strip=True)

    lead = soup.find("p")
    summary = lead.get_text(" ", strip=True) if lead else None

    sections: Dict[str, list[str]] = {}
    current_heading: Optional[str] = None

    for node in soup.find_all(SECTION_HEADING_TAGS.union({"p", "ul", "ol"})):
        if node.name in SECTION_HEADING_TAGS:
            current_heading = node.get_text(" ", strip=True)
            sections.setdefault(current_heading, [])
            continue
        if not current_heading:
            current_heading = "Summary"
            sections.setdefault(current_heading, [])
        if node.name == "p":
            text = node.get_text(" ", strip=True)
            if text:
                sections[current_heading].append(text)
        elif node.name in {"ul", "ol"}:
            for li in node.find_all("li"):
                text = li.get_text(" ", strip=True)
                if text:
                    sections[current_heading].append(text)

    return {
        "title": title,
        "summary": summary,
        "sections": sections,
    }


