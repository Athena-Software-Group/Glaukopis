import re
from pipelines.data_loader import load_pickle_file


class cti_postprocessing:
    def format_rcm(self, text):
        cwe_pattern = r'CWE-\d+'
        matches = re.findall(cwe_pattern, text)
        if matches:
            return matches[-1], True
        else:
            return text, False

    def format_vsp(self, text):
        cvss_pattern = (
            r'AV:[A-Za-z]+/AC:[A-Za-z]+/PR:[A-Za-z]+/UI:[A-Za-z]+/'
            r'S:[A-Za-z]+/C:[A-Za-z]+/I:[A-Za-z]+/A:[A-Za-z]+'
        )
        matches = re.findall(cvss_pattern, text)
        if matches:
            return matches[-1], True
        else:
            return text, False

    def format_mcq(self, text):
        last_line = text.split('\n')[-1].rstrip()
        if last_line.startswith(('A)', 'B)', 'C)', 'D)')):
            return last_line[0]
        if last_line.endswith(('A', 'B', 'C', 'D')):
            return last_line[-1]
        if last_line.endswith('**'):
            return last_line[-3]
        if len(last_line) == 0:
            last_line = text.split('\n')[-2].rstrip()
            if last_line.startswith(('A)', 'B)', 'C)', 'D)')):
                return last_line[0]
            if last_line.endswith(('A', 'B', 'C', 'D')):
                return last_line[-1]
            if last_line.endswith('**'):
                return last_line[-3]
        return ' '.join(text.split('\n'))

    def format_taa(self, prediction):
        if not prediction or not isinstance(prediction, str):
            return "X", False

        pred = prediction.strip()

        try:
            alias_dict = load_pickle_file('benchmark_data/cti_bench/cti_taa/alias_dict.pickle')
            alias_to_canonical = {}
            for main_actor, aliases in alias_dict.items():
                alias_to_canonical[main_actor.lower()] = main_actor
                for alias in aliases:
                    alias_to_canonical[alias.lower()] = main_actor
        except FileNotFoundError:
            return "X", False

        pred_lower = pred.lower()
        if pred_lower in alias_to_canonical:
            return alias_to_canonical[pred_lower], True

        patterns = [
            r'\b[A-Za-z][A-Za-z0-9\s\-_\.]*[A-Za-z0-9]\b',
            r'\bAPT[\-\s]?\d+\b',
            r'\bAPT[\-\s]?[A-Z]\d+\b',
            r'\b[A-Za-z]+\s+[A-Za-z]+\b',
        ]
        candidates = []
        for pattern in patterns:
            matches = re.findall(pattern, pred, re.IGNORECASE)
            candidates.extend(matches)
        for candidate in candidates:
            candidate_clean = candidate.strip().lower()
            if candidate_clean in alias_to_canonical:
                return alias_to_canonical[candidate_clean], True

        common_prefixes = [
            "threat actor:", "attributed to", "group:", "actor:", "the threat actor",
            "responsible:", "is", "group name:", "attacker:", "malware family:"
        ]
        for prefix in common_prefixes:
            if prefix in pred_lower:
                after_prefix = pred_lower.split(prefix, 1)[1].strip()
                words = after_prefix.split()[:3]
                for i in range(len(words), 0, -1):
                    candidate = ' '.join(words[:i]).strip(' .,')
                    if candidate in alias_to_canonical:
                        return alias_to_canonical[candidate], True

        delimiters = ['.', ',', ':', ';', '\n', '(', ')', '[', ']', '"', "'"]
        parts = [pred]
        for delimiter in delimiters:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(delimiter))
            parts = new_parts
        for part in parts:
            part_clean = part.strip().lower()
            if part_clean in alias_to_canonical:
                return alias_to_canonical[part_clean], True

        return "X", False

    def format_ate(self, text):
        if not text or not isinstance(text, str):
            return "X", False

        text = text.strip()
        technique_pattern = r'T\d{4}'
        matches = re.findall(technique_pattern, text)

        if matches:
            unique_techniques = []
            seen = set()
            for technique in matches:
                if technique not in seen:
                    unique_techniques.append(technique)
                    seen.add(technique)
            formatted_output = ', '.join(unique_techniques)
            return formatted_output, True

        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if ',' in line and re.search(technique_pattern, line):
                line_matches = re.findall(technique_pattern, line)
                if line_matches:
                    unique_techniques = []
                    seen = set()
                    for technique in line_matches:
                        if technique not in seen:
                            unique_techniques.append(technique)
                            seen.add(technique)
                    formatted_output = ', '.join(unique_techniques)
                    return formatted_output, True

        return "X", False

    def extract_cve_id(self, text):
        if not text or not isinstance(text, str):
            return "NOT_FOUND", False

        cve_pattern = r"\bCVE-\d{4}-\d{4,7}\b"
        matches = re.findall(cve_pattern, text, re.IGNORECASE)

        if matches:
            return matches[0].upper(), True
        else:
            return "NOT_FOUND", False
    
    def format_cybermetric(self, response_text: str):
        """
        Extract the MCQ answer (A/B/C/D) from CyberMetric LLM response.
        Returns tuple: (answer, success_flag)
        """
        if response_text and response_text.strip():
            match = re.search(r"ANSWER:?\s*([A-D])\b", response_text, re.IGNORECASE)
            if match:
                return match.group(1).upper(), True
        return "X", False
