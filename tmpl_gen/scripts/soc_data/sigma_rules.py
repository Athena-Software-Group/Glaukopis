"""Sigma-rule structural and semantic knowledge for SOC.SIGMA.GEN.1.
Covers the YAML schema (logsource, detection, condition, modifiers),
common detection field semantics across logsources, and the ATT&CK-tag
convention. Curated from sigmahq.io and the SigmaHQ rules repo.
"""

from __future__ import annotations

from collections.abc import Callable

from . import pick_distractors

# Top-level Sigma fields and their definitions
SIGMA_FIELDS: list[tuple[str, str]] = [
    ("title",       "a short human-readable name for the rule"),
    ("id",          "a globally unique UUIDv4 identifier for the rule"),
    ("status",      "the maturity level of the rule (e.g., experimental, test, stable, deprecated)"),
    ("description", "a free-text explanation of what the rule detects and why"),
    ("references",  "a list of URLs supporting the detection logic (blog posts, vendor advisories, CVEs)"),
    ("author",      "the rule author and any co-authors, used for attribution"),
    ("date",        "the rule creation date in YYYY-MM-DD format"),
    ("modified",    "the most recent modification date of the rule"),
    ("tags",        "a list of MITRE ATT&CK technique/tactic identifiers and other classification labels"),
    ("logsource",   "the data source the rule applies to (product, category, service)"),
    ("detection",   "the search-identifier definitions and the boolean condition expression"),
    ("condition",   "a boolean expression combining named search identifiers within the detection block"),
    ("falsepositives", "a list of known benign scenarios that may match the rule"),
    ("level",       "the severity level (informational, low, medium, high, critical)"),
    ("fields",      "an optional list of fields the analyst should pivot on after the alert fires"),
]

# Sigma logsource categories (category, definition, typical product)
LOGSOURCE_CATS: list[tuple[str, str, str]] = [
    ("process_creation", "events emitted whenever a new OS process is created", "Sysmon EID 1 / Windows Security 4688"),
    ("file_event",       "events emitted whenever a file is created, modified, or deleted", "Sysmon EID 11/23/26"),
    ("network_connection", "events emitted whenever a process opens an outbound network connection", "Sysmon EID 3"),
    ("registry_event",   "events emitted on Windows registry create/set/delete operations", "Sysmon EID 12/13/14"),
    ("dns_query",        "events emitted whenever a process performs a DNS lookup", "Sysmon EID 22"),
    ("image_load",       "events emitted whenever a process loads a DLL into its address space", "Sysmon EID 7"),
    ("driver_load",      "events emitted whenever a kernel-mode driver is loaded", "Sysmon EID 6"),
    ("pipe_created",     "events emitted whenever a Windows named pipe is created", "Sysmon EID 17"),
    ("process_access",   "events emitted whenever one process opens a handle to another", "Sysmon EID 10"),
    ("create_remote_thread", "events emitted whenever a thread is created in a remote process", "Sysmon EID 8"),
    ("ps_classic_start", "PowerShell engine start events captured via classic PS logging", "Windows PowerShell EID 400"),
    ("ps_module",        "PowerShell module loading and method invocation logs", "Windows PowerShell/Operational EID 4103"),
    ("ps_script",        "PowerShell script-block deobfuscated content logs", "Windows PowerShell/Operational EID 4104"),
    ("webserver",        "HTTP access logs from a web server", "nginx, Apache, IIS"),
    ("proxy",            "HTTP request/response logs from a forward web proxy", "Bluecoat, Squid, ZIA"),
    ("firewall",         "L3/L4 network filter accept/drop events", "Palo Alto, Cisco ASA, pf, iptables"),
    ("auditd",           "Linux kernel audit log records", "auditd"),
    ("sshd",             "OpenSSH daemon authentication and session events", "sshd"),
]

# Common detection field names and what they hold
DETECTION_FIELDS: list[tuple[str, str, str]] = [
    ("Image",         "process_creation",     "the absolute filesystem path of the executed binary"),
    ("OriginalFileName", "process_creation",  "the OriginalFileName resource from the executable's PE metadata"),
    ("CommandLine",   "process_creation",     "the full command-line string passed when the process was launched"),
    ("ParentImage",   "process_creation",     "the absolute filesystem path of the parent process executable"),
    ("ParentCommandLine", "process_creation", "the full command-line string of the parent process"),
    ("User",          "process_creation",     "the user account under which the process was launched"),
    ("IntegrityLevel", "process_creation",    "the Windows integrity level the process is running at (Low/Medium/High/System)"),
    ("Hashes",        "process_creation",     "comma-separated hashes (MD5, SHA1, SHA256, IMPHASH) computed over the executable image"),
    ("TargetFilename", "file_event",          "the absolute path of the file being created, modified, or deleted"),
    ("DestinationIp", "network_connection",   "the remote IPv4/IPv6 address of the outbound TCP/UDP connection"),
    ("DestinationPort", "network_connection", "the remote TCP/UDP port of the outbound connection"),
    ("DestinationHostname", "network_connection", "the resolved destination hostname when DNS data is available"),
    ("Initiated",     "network_connection",   "boolean indicating the local process initiated the connection (egress)"),
    ("TargetObject",  "registry_event",       "the full registry path being created, modified, or deleted"),
    ("Details",       "registry_event",       "the new value being written to the registry path"),
    ("QueryName",     "dns_query",            "the FQDN being resolved by the local process"),
    ("QueryResults",  "dns_query",            "the IP addresses returned by the resolver"),
    ("ImageLoaded",   "image_load",           "the absolute path of the DLL being loaded into the process"),
    ("PipeName",      "pipe_created",         "the name of the Windows named pipe being created"),
    ("ScriptBlockText", "ps_script",          "the deobfuscated PowerShell script block content"),
]

# Sigma value-modifier tokens
SIGMA_MODIFIERS: list[tuple[str, str]] = [
    ("contains",   "true if the field value contains the supplied substring (case-insensitive by default)"),
    ("startswith", "true if the field value starts with the supplied prefix"),
    ("endswith",   "true if the field value ends with the supplied suffix"),
    ("re",         "true if the supplied regular expression matches the field value"),
    ("re|i",       "true if the supplied regular expression matches the field value, ignoring case"),
    ("base64",     "the value is base64-encoded before comparison so it matches encoded representations"),
    ("base64offset", "expands the value into the three offset-shifted base64 representations to defeat boundary obfuscation"),
    ("utf16le",    "the value is encoded as UTF-16 little-endian before comparison (PowerShell -enc convention)"),
    ("wide",       "alias for utf16le; matches Unicode-encoded values"),
    ("cidr",       "interprets the supplied value as a CIDR range and matches IPs within it"),
    ("all",        "applied to a list, requires every list element to match (logical AND)"),
    ("expand",     "instructs the rule processor to expand placeholder values from the rule's placeholders block"),
]

# Common Sigma condition expressions
SIGMA_CONDITIONS: list[tuple[str, str]] = [
    ("selection",                 "match all named selections defined in the detection block"),
    ("selection and not filter",  "match the selection but exclude rows that also match the filter (whitelist pattern)"),
    ("1 of selection_*",          "match if any single selection_* identifier matches (logical OR over a glob)"),
    ("all of selection_*",        "match only if every selection_* identifier matches (logical AND over a glob)"),
    ("selection | count() > 5",   "aggregate over matching events and trigger only when the count exceeds the threshold"),
    ("selection | count() by User > 10", "aggregate matches grouped by the User field and trigger when any group exceeds 10"),
    ("selection | near alert within 10m", "trigger only when 'selection' and 'alert' both match within a 10-minute window"),
]

# Sigma rule severity levels
SIGMA_LEVELS: list[tuple[str, str]] = [
    ("informational", "the rule reports an event of interest with no expectation of malicious intent"),
    ("low",      "the rule indicates suspicious behaviour that requires aggregate context to triage"),
    ("medium",   "the rule fires on behaviour that warrants analyst review even in isolation"),
    ("high",     "the rule fires on behaviour strongly indicative of malicious activity"),
    ("critical", "the rule fires on behaviour that almost certainly represents a confirmed compromise"),
]


def generate(rng, target: int, instruction: str, shortname: str,
             make_mcq: Callable) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < target:
        rows.extend(_one_pass(rng, instruction, shortname, make_mcq))
    return rows[:target]


def _one_pass(rng, instruction: str, shortname: str,
              make_mcq: Callable) -> list[dict]:
    rows: list[dict] = []

    # Pattern 1 -- Sigma top-level fields
    field_names = [n for n, _ in SIGMA_FIELDS]
    field_defs = [d for _, d in SIGMA_FIELDS]
    for name, defn in SIGMA_FIELDS:
        for phr in [f"In a Sigma rule YAML, which top-level field holds {defn}?",
                    f"Which Sigma rule field is described as: {defn}?",
                    f"Which top-level Sigma field corresponds to: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, field_names, name),
                f"In a Sigma rule, the '{name}' field holds {defn}.",
                shortname, instruction))
        for phr in [f"What does the Sigma rule '{name}' field hold?",
                    f"How is the Sigma '{name}' field defined?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, field_defs, defn),
                f"In a Sigma rule, the '{name}' field holds {defn}.",
                shortname, instruction))

    # Pattern 2 -- logsource categories
    cat_names = [c for c, _, _ in LOGSOURCE_CATS]
    cat_defs = [d for _, d, _ in LOGSOURCE_CATS]
    cat_prods = [p for _, _, p in LOGSOURCE_CATS]
    for cat, defn, prod in LOGSOURCE_CATS:
        for phr in [f"Which Sigma logsource category covers: {defn}?",
                    f"Which logsource.category value corresponds to: {defn}?",
                    f"Which Sigma category does the following describe: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, cat, pick_distractors(rng, cat_names, cat),
                f"The Sigma logsource category '{cat}' covers {defn}.",
                shortname, instruction))
        for phr in [f"Which product/source typically populates the Sigma '{cat}' logsource category?",
                    f"Which data source most commonly drives the Sigma '{cat}' category?"]:
            rows.append(make_mcq(
                rng, phr, prod, pick_distractors(rng, cat_prods, prod),
                f"The Sigma '{cat}' category is typically populated by {prod}.",
                shortname, instruction))
        for phr in [f"What does the Sigma logsource category '{cat}' contain?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, cat_defs, defn),
                f"The Sigma logsource category '{cat}' covers {defn}.",
                shortname, instruction))

    # Pattern 3 -- detection field semantics
    det_names = [n for n, _, _ in DETECTION_FIELDS]
    det_defs = [d for _, _, d in DETECTION_FIELDS]
    for fname, cat, defn in DETECTION_FIELDS:
        for phr in [f"In a Sigma rule's '{cat}' detection block, which field holds {defn}?",
                    f"Which Sigma detection field corresponds to: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, fname, pick_distractors(rng, det_names, fname),
                f"In a Sigma '{cat}' rule, the '{fname}' field holds {defn}.",
                shortname, instruction))
        for phr in [f"What does the Sigma detection field '{fname}' hold?",
                    f"In a Sigma '{cat}' rule, what is captured in '{fname}'?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, det_defs, defn),
                f"In a Sigma '{cat}' rule, the '{fname}' field holds {defn}.",
                shortname, instruction))

    # Pattern 4 -- modifier semantics
    mod_names = [m for m, _ in SIGMA_MODIFIERS]
    mod_defs = [d for _, d in SIGMA_MODIFIERS]
    for mod, defn in SIGMA_MODIFIERS:
        for phr in [f"What does the Sigma value-modifier '{mod}' do?",
                    f"In a Sigma rule, the '|{mod}' modifier applied to a field means what?",
                    f"Which Sigma modifier behaviour does '{mod}' describe?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, mod_defs, defn),
                f"The Sigma value-modifier '{mod}': {defn}.",
                shortname, instruction))
        for phr in [f"Which Sigma value-modifier should you use to express: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, mod, pick_distractors(rng, mod_names, mod),
                f"The Sigma value-modifier '{mod}': {defn}.",
                shortname, instruction))

    # Pattern 5 -- condition expressions
    cond_exprs = [c for c, _ in SIGMA_CONDITIONS]
    cond_defs = [d for _, d in SIGMA_CONDITIONS]
    for expr, defn in SIGMA_CONDITIONS:
        for phr in [f"What does the Sigma condition expression '{expr}' mean?",
                    f"Which behaviour does the Sigma condition '{expr}' encode?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, cond_defs, defn),
                f"The Sigma condition '{expr}' means: {defn}.",
                shortname, instruction))
        for phr in [f"Which Sigma condition expression encodes: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, expr, pick_distractors(rng, cond_exprs, expr),
                f"The Sigma condition '{expr}' means: {defn}.",
                shortname, instruction))

    # Pattern 6 -- severity levels
    lvl_names = [n for n, _ in SIGMA_LEVELS]
    lvl_defs = [d for _, d in SIGMA_LEVELS]
    for lvl, defn in SIGMA_LEVELS:
        for phr in [f"Which Sigma severity level fits the description: {defn}?",
                    f"What Sigma 'level' value corresponds to: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, lvl, pick_distractors(rng, lvl_names, lvl),
                f"The Sigma severity level '{lvl}' is defined as: {defn}.",
                shortname, instruction))
        for phr in [f"What does the Sigma severity level '{lvl}' indicate?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, lvl_defs, defn),
                f"The Sigma severity level '{lvl}' is defined as: {defn}.",
                shortname, instruction))

    return rows

