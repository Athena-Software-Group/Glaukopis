"""Markdown renderers for MCQ content."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import bs4


def _clean_text(text: Optional[str]) -> str:
    return text.strip() if isinstance(text, str) else ""


def _markdown_from_sections(title: str, sections: Dict[str, str]) -> str:
    lines = [f"# {title.strip()}" if title else "#"]
    for heading, body in sections.items():
        body = body.strip()
        if not body:
            continue
        lines.append(f"## {heading.strip()}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_mitre_attack(html: str) -> str:
    soup = bs4.BeautifulSoup(html, "html.parser")
    header = soup.find("h1")
    title = header.get_text(" ", strip=True) if header else "MITRE ATT&CK Entry"

    sections: Dict[str, str] = {}

    labels = {
        "ID": "ID",
        "Tactic": "Tactic",
        "Platform": "Platform",
    }
    for key, heading in labels.items():
        span = soup.find("span", string=lambda s: s and heading in s)
        if span:
            value = span.find_next("a")
            if value:
                sections[key] = value.get_text(" ", strip=True)
            else:
                sibling = span.find_next(string=True)
                sections[key] = (sibling or "").strip()

    description = soup.find("div", class_="description-body")
    if description:
        sections["Description"] = description.get_text("\n", strip=True)

    for node in soup.find_all(["h2", "h3"]):
        heading = node.get_text(" ", strip=True)
        if heading.lower().startswith("references"):
            break
        content_parts = []
        for sibling in node.find_next_siblings():
            if sibling.name in {"h2", "h3"}:
                break
            if sibling.name == "p":
                content_parts.append(sibling.get_text(" ", strip=True))
            elif sibling.name in {"ul", "ol"}:
                for li in sibling.find_all("li"):
                    content_parts.append(f"- {li.get_text(' ', strip=True)}")
        if content_parts:
            sections[heading] = "\n".join(content_parts)

    return _markdown_from_sections(title, sections)


def render_cwe_entry(html: str) -> str:
    soup = bs4.BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "CWE Entry"

    def extract_section(section_id: str) -> str:
        node = soup.find("div", id=section_id)
        if not node:
            return ""
        block = node.find_next("div", class_="expandblock")
        if not block:
            return node.get_text("\n", strip=True)
        detail = block.find("div", class_="detail")
        return detail.get_text("\n", strip=True) if detail else block.get_text("\n", strip=True)

    sections = {
        "Description": extract_section("Description"),
        "Modes Of Introduction": extract_section("Modes_Of_Introduction"),
        "Common Consequences": extract_section("Common_Consequences"),
        "Potential Mitigations": extract_section("Potential_Mitigations"),
    }

    related = []
    related_div = soup.find("div", id="Related_Attack_Patterns")
    if related_div:
        table = related_div.find_next("table")
        if table:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    related.append(f"- **{cells[0].get_text(strip=True)}:** {cells[1].get_text(strip=True)}")
    if related:
        sections["Related Attack Patterns"] = "\n".join(related)

    return _markdown_from_sections(title, sections)


def render_capec_entry(html: str) -> str:
    soup = bs4.BeautifulSoup(html, "html.parser")
    heading = soup.find("h2") or soup.find("h1")
    title = heading.get_text(strip=True) if heading else "CAPEC Entry"

    def extract_text(section_id: str) -> str:
        node = soup.find("div", id=section_id)
        if not node:
            return ""
        block = node.find_next("div", class_="expandblock") or node
        detail = block.find("div", class_="detail")
        return detail.get_text("\n", strip=True) if detail else block.get_text("\n", strip=True)

    sections = {
        "Description": extract_text("Description"),
        "Likelihood Of Attack": extract_text("Likelihood_Of_Attack"),
        "Typical Severity": extract_text("Typical_Severity"),
        "Execution Flow": extract_text("Execution_Flow"),
        "Prerequisites": extract_text("Prerequisites"),
        "Skills Required": extract_text("Skills_Required"),
    }

    mitigations = extract_text("Mitigations")
    if mitigations:
        sections["Mitigations"] = mitigations

    related = []
    root = soup.find("div", id="Related_Weaknesses")
    if root:
        table = root.find("table", id="Detail")
        if table:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    related.append(f"- **{cells[0].get_text(strip=True)}:** {cells[1].get_text(strip=True)}")
    if related:
        sections["Related Weaknesses"] = "\n".join(related)

    return _markdown_from_sections(title, sections)


def render_cisa_advisory(html: str) -> str:
    soup = bs4.BeautifulSoup(html, "html.parser")
    title_node = soup.find("h1") or soup.find("title")
    title = title_node.get_text(" ", strip=True) if title_node else "CISA Advisory"

    sections: Dict[str, str] = {}
    intro = soup.find("p")
    if intro:
        sections["Summary"] = intro.get_text(" ", strip=True)

    for node in soup.find_all(["h2", "h3"]):
        heading = node.get_text(" ", strip=True)
        if not heading:
            continue
        content_parts = []
        for sibling in node.find_next_siblings():
            if sibling.name in {"h2", "h3"}:
                break
            if sibling.name == "p":
                content_parts.append(sibling.get_text(" ", strip=True))
            elif sibling.name in {"ul", "ol"}:
                for li in sibling.find_all("li"):
                    content_parts.append(f"- {li.get_text(' ', strip=True)}")
        if content_parts:
            sections[heading] = "\n".join(content_parts)

    return _markdown_from_sections(title, sections)


RENDERERS: Dict[str, Callable[[str], str]] = {
    "mitre_attack": render_mitre_attack,
    "cwe_catalog": render_cwe_entry,
    "capec_catalog": render_capec_entry,
    "cisa_ics": render_cisa_advisory,
    "cisa_csa": render_cisa_advisory,
}


def render_markdown(source_type: str, html: str, fallback_text: str) -> str:
    renderer = RENDERERS.get(source_type)
    if renderer:
        try:
            rendered = renderer(html)
            if rendered and len([line for line in rendered.strip().splitlines() if line.strip()]) > 1:
                return rendered if rendered.endswith("\n") else rendered + "\n"
        except Exception:
            pass
    clean = fallback_text.strip()
    if clean:
        return clean + "\n"
    return bs4.BeautifulSoup(html or "", "html.parser").get_text("\n", strip=True) + "\n"



