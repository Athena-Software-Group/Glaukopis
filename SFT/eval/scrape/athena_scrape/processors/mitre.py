"""Parsers for MITRE ATT&CK content."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import bs4


def _find_labelled_value(soup: bs4.BeautifulSoup, label: str) -> Optional[str]:
    node = soup.find("span", class_="h5", string=re.compile(label, re.I))
    if not node:
        return None
    sibling = node.find_next(string=False)
    if sibling and hasattr(sibling, "get_text"):
        return sibling.get_text(" ", strip=True)
    link = node.find_next("a")
    return link.get_text(" ", strip=True) if link else None


def parse_mitre_attack(html: str) -> Optional[Dict[str, Any]]:
    """Extract salient MITRE ATT&CK attributes from HTML."""

    soup = bs4.BeautifulSoup(html, "html.parser")

    header = soup.find("h1")
    if not header:
        return None

    raw_heading = header.get_text(" ", strip=True)
    parts = [p.strip() for p in re.split(r"\s+\|\s+", raw_heading) if p.strip()]
    identifier: Optional[str]
    name: Optional[str]
    if len(parts) == 2:
        identifier, name = parts
    else:
        id_span = header.find("span", string=re.compile(r"^[TSMG]\d+", re.I))
        identifier = id_span.get_text(strip=True) if id_span else None
        name = raw_heading

    data: Dict[str, Any] = {
        "id": identifier,
        "name": name,
    }

    tactic = _find_labelled_value(soup, "Tactic")
    platform = _find_labelled_value(soup, "Platform")
    description_node = soup.find("div", class_="description-body")
    description = description_node.get_text(" ", strip=True) if description_node else None

    data.update(
        {
            "tactic": tactic,
            "platform": platform,
            "description": description,
        }
    )

    sections: Dict[str, Any] = {}
    for node in soup.find_all(["h2", "h3", "h4"]):
        heading = node.get_text(" ", strip=True)
        if heading.lower().startswith("references"):
            break
        content: list[str] = []
        for sibling in node.next_siblings:
            if getattr(sibling, "name", None) in {"h2", "h3", "h4"}:
                break
            if getattr(sibling, "name", None) == "p":
                text = sibling.get_text(" ", strip=True)
                if text:
                    content.append(text)
            elif getattr(sibling, "name", None) in {"ul", "ol"}:
                for li in sibling.find_all("li"):
                    li_text = li.get_text(" ", strip=True)
                    if li_text:
                        content.append(li_text)
        if content:
            sections[heading] = content

    if sections:
        data["sections"] = sections

    label = soup.find("span", string=re.compile("Last Modified", re.I))
    if label:
        next_text = label.find_next(string=True)
        if isinstance(next_text, str):
            data["last_modified"] = next_text.strip()

    return data


