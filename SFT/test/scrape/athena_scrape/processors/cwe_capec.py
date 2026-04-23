"""Parsing helpers for CWE and CAPEC HTML pages."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import bs4


def _extract_expandblock_text(root: bs4.BeautifulSoup, section_id: str) -> Optional[str]:
    node = root.find("div", id=section_id)
    if not node:
        return None
    block = node.find_next("div", class_="expandblock")
    if not block:
        return None
    detail = block.find("div", class_="detail")
    if not detail:
        return block.get_text("\n", strip=True)
    return detail.get_text("\n", strip=True)


def parse_cwe(html: str) -> Optional[Dict[str, Any]]:
    soup = bs4.BeautifulSoup(html, "html.parser")
    title = soup.find("title")
    if not title:
        return None

    title_text = title.get_text(strip=True)
    match = re.search(r"CWE-(\d+)\s*:(.*)", title_text)
    cwe_id = match.group(1) if match else None
    name = match.group(2).strip() if match else title_text

    description = _extract_expandblock_text(soup, "Description")

    def parse_table(section_id: str, headers_expected: Optional[int] = None) -> List[Dict[str, str]]:
        node = soup.find("div", id=section_id)
        if not node:
            return []
        table = node.find_next("table")
        if not table:
            return []
        header_cells = table.find_all("th")
        headers = [cell.get_text(strip=True) for cell in header_cells]
        items: List[Dict[str, str]] = []
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue
            entry: Dict[str, str] = {}
            for idx, cell in enumerate(cells):
                key = headers[idx] if idx < len(headers) else f"col_{idx}"
                entry[key] = cell.get_text(" ", strip=True)
            items.append(entry)
        if headers_expected and len(headers) != headers_expected:
            return []
        return items

    modes = parse_table("Modes_Of_Introduction")
    consequences = parse_table("Common_Consequences")

    mitigations: List[Dict[str, str]] = []
    mitigation_root = soup.find("div", id="Potential_Mitigations")
    if mitigation_root:
        for section in mitigation_root.find_all("div", class_="expandblock"):
            phase = section.find_previous("p", class_="subheading")
            detail = section.find("div", class_="indent")
            mitigations.append(
                {
                    "phase": phase.get_text(strip=True) if phase else "",
                    "detail": detail.get_text(" ", strip=True) if detail else section.get_text(" ", strip=True),
                }
            )

    related_patterns: List[Dict[str, str]] = []
    patterns_table = soup.find("div", id="Related_Attack_Patterns")
    if patterns_table:
        table = patterns_table.find_next("table")
        if table:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    related_patterns.append(
                        {
                            "capec_id": cells[0].get_text(strip=True),
                            "name": cells[1].get_text(strip=True),
                        }
                    )

    return {
        "id": f"CWE-{cwe_id}" if cwe_id else None,
        "name": name,
        "description": description,
        "modes_of_introduction": modes,
        "common_consequences": consequences,
        "potential_mitigations": mitigations,
        "related_attack_patterns": related_patterns,
    }


def parse_capec(html: str) -> Optional[Dict[str, Any]]:
    soup = bs4.BeautifulSoup(html, "html.parser")
    heading = soup.find("h2") or soup.find("h1")
    if not heading:
        return None

    heading_text = heading.get_text(strip=True)
    match = re.search(r"CAPEC-(\d+)", heading_text)
    capec_id = match.group(1) if match else None

    description = _extract_expandblock_text(soup, "Description")
    likelihood = _extract_expandblock_text(soup, "Likelihood_Of_Attack")
    severity = _extract_expandblock_text(soup, "Typical_Severity")
    prerequisites = _extract_expandblock_text(soup, "Prerequisites")
    skills = _extract_expandblock_text(soup, "Skills_Required")

    execution_flow_root = soup.find("div", id="Execution_Flow")
    execution_steps: List[str] = []
    if execution_flow_root:
        for td in execution_flow_root.find_all("td"):
            text = td.get_text(" ", strip=True)
            if text:
                execution_steps.append(text)

    consequences = []
    consequences_root = soup.find("div", id="Consequences")
    if consequences_root:
        table = consequences_root.find("table", id="Detail")
        if table:
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if not cells:
                    continue
                entry = {}
                for idx, cell in enumerate(cells):
                    key = headers[idx] if idx < len(headers) else f"col_{idx}"
                    entry[key] = cell.get_text(" ", strip=True)
                consequences.append(entry)

    mitigations = []
    mitigations_root = soup.find("div", id="Mitigations")
    if mitigations_root:
        for td in mitigations_root.find_all("td"):
            text = td.get_text(" ", strip=True)
            if text:
                mitigations.append(text)

    related_weaknesses = []
    related_root = soup.find("div", id="Related_Weaknesses")
    if related_root:
        table = related_root.find("table", id="Detail")
        if table:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    related_weaknesses.append(
                        {
                            "cwe_id": cells[0].get_text(strip=True),
                            "weakness_name": cells[1].get_text(strip=True),
                        }
                    )

    return {
        "id": f"CAPEC-{capec_id}" if capec_id else None,
        "name": heading_text,
        "description": description,
        "likelihood_of_attack": likelihood,
        "typical_severity": severity,
        "execution_flow": execution_steps,
        "prerequisites": prerequisites,
        "skills_required": skills,
        "consequences": consequences,
        "mitigations": mitigations,
        "related_weaknesses": related_weaknesses,
    }


