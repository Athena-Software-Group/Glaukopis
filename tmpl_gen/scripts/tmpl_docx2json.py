#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 13:11:51 2026

@author: icardei


Extracts templates written in a Word docx file OR a JSON file to a file in JSON tmpl_gen format.
The output JSON file can be used as input for iftgen.py to generate an IFT dataset.

Usage examples:

    python3 tmpl_docx2json.py -i templates-aligned-2025-02.docx -o templates-aligned-2025-02.json --count_limit 3000
    python3 tmpl_docx2json.py -i templates.json -o templates-out.json --count_limit 3000

Here is an example for a template written in a Word document or plain text file:

"1.1 Instruction: You are a CTI expert that gives precise and concise answers. Question: What course of action should be taken to mitigate the attack pattern {ap:attack-pattern.name} ({ap.id})? Answer: To mitigate {ap.name}, organizations should implement the following course of action: {coa:ap.mitigates<course-of-action.name}. Description: {coa.description}.
Summary: {ap.mitigates<course-of-action|description}
Schema: course-of-action - mitigates - attack-pattern"

Docx/txt format:
  line 1: tmpl ID: xxx.yyy   string
  line 1: Instruction: <instruction text> Question: <question text> Answer: <answer text>
  OPTIONAL line 2: Summary: <summary template code>
  OPTIONAL next line: Schema: <relationship 1>[, <relationship 2>]*

JSON input format (array of objects):
  [
    {
      "id": "M.1",
      "instruction": "<instruction text>",
      "question": "<question text>",
      "answer": "<answer text>",
      "summary": "<optional summary>",
      "schema": "<optional schema>",
      "count_limit": <optional int>
    },
    ...
  ]

NOTE:
    It only accepts as input DOCX files, NOT older .DOC files.


TODO: Generalize the input format, like .txt, .doc..,
"""

import docx
import json
import argparse
import re

def get_text_lines(filename:str) -> list[str]:
    """
    Returns the lines of text from a source file.
    NOTE: does not work wth .doc format.
    """
    if filename.endswith(".docx") or filename.endswith(".DOCX"):
        doc = docx.Document(filename)
        lst_lines = list(doc.paragraphs)
    else:
        # treat like a text file:
        with open(filename, "r") as fin:
            lst_lines = list(fin)
    return lst_lines


def extract_templates_from_json(args) -> list[dict]:
    """
    Reads templates from a structured JSON input file.
    Each entry must have: id, instruction, question, answer.
    Optional fields: summary, schema, count_limit.
    """
    with open(args.input, "r") as fin:
        lst_input = json.load(fin)

    if not isinstance(lst_input, list):
        raise ValueError("JSON input must be a top-level array of template objects.")

    lst_tmpls = list()
    for i, entry in enumerate(lst_input):
        missing = [f for f in ("id", "instruction", "question", "answer") if f not in entry]
        if missing:
            print(f"\n@@@ Entry {i} missing required field(s): {missing} — skipping.\n")
            continue

        t_id    = str(entry["id"]).strip()
        t_instr = str(entry["instruction"]).strip()
        t_q     = str(entry["question"]).strip()
        t_a     = str(entry["answer"]).strip()

        t_text = f"Instruction: {t_instr}\n\nQuestion: {t_q}\n\nAnswer: {t_a}"
        tmpl = {
            "shortname":   t_id,
            "comment":     t_id,
            "text":        t_text,
            "source_file": args.input,
            "source_line": i + 1,
        }
        # Only carry forward count_limit when the source JSON has declared one
        # explicitly; otherwise leave it unset so the parser's resolution
        # chain can fall through to --count_max / default_count_limit.
        if "count_limit" in entry and entry["count_limit"] is not None:
            try:
                tmpl["count_limit"] = int(entry["count_limit"])
            except (TypeError, ValueError):
                print(f"@@@ Entry {i}: invalid count_limit {entry['count_limit']!r} - ignored")

        for prefix in ("summary", "schema", "source"):
            if prefix in entry and entry[prefix]:
                tmpl[prefix] = str(entry[prefix]).strip()

        lst_tmpls.append(tmpl)
        print(f"    Loaded template {t_id}")

    return lst_tmpls


def extract_templates_from_txt(args) -> list[dict]:
    """
    Reads templates from a plain-text file.
    Expected format per template block (blocks separated by blank lines):

      ID Instruction: <text>
      Question: <text>
      [A) ...  B) ...  C) ...  D) ...]   (optional MCQ options, appended to question)
      Answer: <text>
      [{force ...}]    (optional constraint annotations — skipped)
      [Summary: <text>]
      [Schema: <text>]
    """
    with open(args.input, "r") as fin:
        lines = [l.rstrip("\n") for l in fin]

    re_id_instr = re.compile(r"^(\S+)\s+Instruction:\s*(.+)$")

    lst_tmpls = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        m = re_id_instr.match(line)
        if not m:
            i += 1
            continue

        src_line = i + 1
        t_id = m.group(1)
        t_instr = m.group(2).strip()
        i += 1

        # Collect Question (skip any intervening blank lines)
        t_q = None
        while i < len(lines):
            l = lines[i].strip()
            if re_id_instr.match(l):
                break
            if l.startswith("Question:"):
                q_parts = [l[len("Question:"):].strip()]
                i += 1
                # Append MCQ option lines (A), B), ... that immediately follow
                while i < len(lines) and re.match(r"^[A-Z]\)", lines[i].strip()):
                    q_parts.append(lines[i].strip())
                    i += 1
                t_q = "\n".join(q_parts)
                break
            i += 1

        if t_q is None:
            print(f"\n@@@ Template {t_id} at line {src_line}: missing Question — skipping.\n")
            continue

        # Collect Answer.  After the initial "Answer: <text>" line we accept
        # additional "Answer: <text>" continuation lines (the AB.RMS.3*
        # convention: a long human rationale followed by a strict-format
        # final-answer directive on its own "Answer: ..." line).  Blank
        # lines between continuations are preserved.  Any other non-blank,
        # non-sentinel line (e.g. a section banner like "EPSS Templates"
        # or "Section 5 - ...") terminates the answer -- it is treated as
        # stray prose belonging to the file's outer structure, not the
        # template body.  Stop also at: next template's ID line, a
        # {force ...} constraint, or Summary/Schema/Sample/Shuffle/Count/Source.
        _ANS_SENTINEL_PREFIXES = ("{force", "Summary: ", "Schema: ",
                                  "Sample: ", "Shuffle: ", "Count: ",
                                  "Source: ")
        t_a = None
        ans_lines = []
        while i < len(lines):
            l = lines[i].strip()
            if re_id_instr.match(l):
                break
            if t_a is None:
                if l.startswith("Answer:"):
                    t_a = l[len("Answer:"):].strip()
                    ans_lines.append(t_a)
                    i += 1
                    continue
                i += 1
                continue
            if l.startswith(_ANS_SENTINEL_PREFIXES):
                break
            if l == "":
                ans_lines.append("")
                i += 1
                continue
            if l.startswith("Answer:"):
                # Preserve the literal "Answer:" prefix on continuation
                # lines: AthenaBench scorers (athena-rms / athena-ate /
                # athena-rcm / athena-vsp / athena-taa) match a final
                # "Answer: <ID>" sentinel in the model output.
                ans_lines.append(l)
                i += 1
                continue
            # Stray non-Answer prose between templates (section banner /
            # horizontal rule / commentary) -- end the answer here.
            break

        if t_a is None:
            print(f"\n@@@ Template {t_id} at line {src_line}: missing Answer — skipping.\n")
            continue

        while ans_lines and ans_lines[-1] == "":
            ans_lines.pop()
        t_a = "\n".join(ans_lines)

        t_text = f"Instruction: {t_instr}\n\nQuestion: {t_q}\n\nAnswer: {t_a}"
        tmpl = {
            "shortname":   t_id,
            "comment":     t_id,
            "text":        t_text,
            "source_file": args.input,
            "source_line": src_line,
        }

        # Collect optional Summary/Schema/Sample/Shuffle/Count; skip {force}
        # lines and blank lines. Count: N is authoritative per-template author
        # intent (it is consulted by tmpl_parser.process_template ahead of the
        # CLI --count_max, which acts as an operator's broad cap for templates
        # that do not declare a Count: of their own). When no Count: directive
        # is present we intentionally omit "count_limit" from the template so
        # the downstream resolution chain can distinguish author-declared from
        # unspecified.
        while i < len(lines):
            l = lines[i].strip()
            if re_id_instr.match(l):
                break  # leave for outer loop
            if l == "" or l.startswith("{force"):
                i += 1
                continue
            for prefix in ("Summary", "Schema", "Sample", "Shuffle", "Count",
                           "Per_primary_grouping", "Source"):
                if l.startswith(f"{prefix}: "):
                    val = l[len(f"{prefix}: "):]
                    if prefix == "Count":
                        try:
                            tmpl["count_limit"] = int(val.strip())
                        except ValueError:
                            print(f"@@@ Template {t_id}: invalid Count '{val}' - ignored")
                    elif prefix == "Per_primary_grouping":
                        # Per-template override of the gencfg per_primary_grouping
                        # default. Used to opt high-fan-out constraint templates
                        # (e.g. AB.TAA.NEG.1: rel != grp sharing 3 patterns) out
                        # of the per-primary CALL-subquery form, which under
                        # tight constraints collapses yield to ~10% of the cap.
                        sval = val.strip().lower()
                        if sval in ("true", "1", "yes"):
                            tmpl["per_primary_grouping"] = True
                        elif sval in ("false", "0", "no"):
                            tmpl["per_primary_grouping"] = False
                        else:
                            print(f"@@@ Template {t_id}: invalid Per_primary_grouping '{val}' - ignored")
                    elif prefix == "Source":
                        # v13 licence-allowlist gate input. Stored as a free-form
                        # tag (e.g. "athena-cti-db-internal", "mitre-attack",
                        # "misp-galaxy-threat-actor"). Absent => downstream
                        # to_alpaca defaults to "athena-cti-db-internal".
                        tmpl["source"] = val.strip()
                    else:
                        tmpl[prefix.lower()] = val
                    break
            i += 1

        lst_tmpls.append(tmpl)
        print(f"    Loaded template {t_id}")

    return lst_tmpls


def extract_templates(args):
    # Route to JSON handler when input is a JSON file
    if args.input.lower().endswith(".json"):
        lst_tmpls = extract_templates_from_json(args)
        with open(args.out, "w") as fout:
            fout.write(json.dumps(lst_tmpls, indent=4))
        print(f"\nWrote {len(lst_tmpls)} templates to file {args.out}")
        return

    # Route to plain-text handler for .txt files
    if args.input.lower().endswith(".txt"):
        lst_tmpls = extract_templates_from_txt(args)
        with open(args.out, "w") as fout:
            fout.write(json.dumps(lst_tmpls, indent=4))
        print(f"\nWrote {len(lst_tmpls)} templates to file {args.out}")
        return

    lst_para = get_text_lines(args.input)
    lst_tmpls = list()
    re_pattern = r"^(.*) Instruction: (.*) Question: (.*) Answer: (.*)$"

    i = 0
    while i < len(lst_para):
        para = lst_para[i]
        # paragraphs from docx have a .text attribute; plain-text lines are strings
        line = para.text.strip() if hasattr(para, "text") else para.strip()
        i += 1
        # print(i, ":", line)
        if len(line) < 50:   # skip titles, headings, etc.
            print(f"--- Skip short line {i}: {line}\n")
            continue

        re_match = re.search(re_pattern, line)
        if not re_match:
            print(f"\n*** Skip no match line {i}: {line}\n")
            continue

        groups = list(re_match.groups())
        if len(groups) != 4:
            print(f"\n@@@ line {i} Template format error. Line:\n{line}\nGroups: {groups}\n")
            continue
        t_id, t_instr, t_q, t_a = groups
        t_text = f"Instruction: {t_instr}\n\nQuestion: {t_q}\n\nAnswer: {t_a}"
        tmpl = {"shortname": t_id, "comment": t_id, "text": t_text,
                "source_file": args.input, "source_line": i,
                "count_limit": args.count_limit}

        lst_prefixes = ["Summary", "Schema"]
        for prefix in lst_prefixes:
            if i == len(lst_para):
                break
            next_para = lst_para[i]
            next_line = next_para.text.strip() if hasattr(next_para, "text") else next_para.strip()
            ppref = f"{prefix}: "
            if next_line.startswith(ppref):
                tmpl[prefix.lower()] = next_line.split(ppref)[1]
                i += 1
        lst_tmpls.append(tmpl)

    with open(args.out, "w") as fout:
        json_txt = json.dumps(lst_tmpls, indent=4)
        fout.write(json_txt)

    print(f"\nWrote {len(lst_tmpls)} templates to file {args.out}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Extract templates from a Word doc/x file or a JSON file to a JSON file to be used as input to ifgen.
Usage examples:

    python3 tmpl_docx2json.py -i templates-aligned-2025-02.docx -o templates-aligned-2025-02.json --count_limit 3000
    python3 tmpl_docx2json.py -i templates.json -o templates-out.json --count_limit 3000
"""
    )
    
    parser.add_argument(
         "--input",
         "-i",
         type=str,
         required=True,
#         default="",
         help="Input file: Word document (.docx), plain-text (.txt), or structured JSON array (.json)"
    )
    
    parser.add_argument(
         "--count_limit",
         "-c",
         type=int,
         required=False,
         default=3000,
         help="Count limit parameter for triples generated from one template."
    )
    
    
    parser.add_argument(
         "--out",
         "-o",
         type=str,
         required=True,
#         default="",
         help="Output file, in JSON format."
    )
    
    args = parser.parse_args()

    extract_templates(args)


if __name__ == "__main__":
    main()
    
