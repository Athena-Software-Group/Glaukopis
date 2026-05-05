"""Alert triage rubric, severity scoring, escalation criteria, and
enrichment-field semantics for SOC.TRIAGE.GEN.1. Curated from
SANS SOC analyst guides, MITRE D3FEND, and common SOAR-vendor
triage runbooks.
"""

from __future__ import annotations

from collections.abc import Callable

from . import pick_distractors

# (severity, definition)
SEVERITY_LEVELS: list[tuple[str, str]] = [
    ("Informational",
     "an event the SOC records for context but expects no analyst action under normal conditions"),
    ("Low",
     "an event that requires aggregate context to be actionable; investigated within 24-72 hours"),
    ("Medium",
     "an event that warrants individual analyst review within the same business day"),
    ("High",
     "an event strongly suggestive of malicious activity that requires response within hours"),
    ("Critical",
     "an event with high confidence of confirmed compromise requiring immediate (within 30 minutes) response"),
]

# (signal, suggests, why)
TP_FP_SIGNALS: list[tuple[str, str, str]] = [
    ("execution from a user's Downloads or Desktop folder",
     "true positive",
     "uncommon location for legitimate enterprise software; typical for downloaded malware"),
    ("execution under the SYSTEM account from a temporary path",
     "true positive",
     "legitimate SYSTEM-context binaries normally execute from System32 or Program Files"),
    ("execution signed by the same enterprise CA used for the user's other binaries",
     "false positive",
     "consistent with enterprise software-distribution chains and gold-image baselines"),
    ("PowerShell launched with -EncodedCommand and -NoProfile -WindowStyle Hidden",
     "true positive",
     "the parameter set is the canonical hallmark of red-team and commodity-malware loaders"),
    ("LSASS handle access from a non-Microsoft binary with PROCESS_VM_READ rights",
     "true positive",
     "Mimikatz-style credential dumping; legitimate AV/EDR uses kernel callbacks instead"),
    ("rundll32 spawned from winword.exe with a path under %TEMP%",
     "true positive",
     "characteristic Office-macro-to-loader chain pattern"),
    ("scheduled-task creation by a domain-admin during a known maintenance window",
     "false positive",
     "consistent with documented change ticket; cross-check with CMDB"),
    ("DNS query for a punycode lookalike of an internal domain",
     "true positive",
     "homograph-style brand-impersonation domain typical of phishing infrastructure"),
    ("SMB write of an executable to a remote ADMIN$ share",
     "true positive",
     "lateral-movement pattern (PsExec-style) that legitimate users do not perform"),
    ("a single failed sudo for a known-typo-prone username",
     "false positive",
     "single low-rate auth failure consistent with human typo, not brute force"),
    ("token theft of a refresh token followed by impossible-travel sign-in",
     "true positive",
     "AITM phishing pattern; geographic improbability rules out legitimate roaming"),
    ("OAuth consent grant for an unverified application requesting Mail.ReadWrite scope",
     "true positive",
     "consent-phishing pattern abusing OAuth to bypass MFA-protected mailboxes"),
]

# (enrichment_field, what_it_tells_you)
ENRICHMENT_FIELDS: list[tuple[str, str]] = [
    ("user.role / user.department",
     "the business context of the user account; helps gauge blast radius of a compromise"),
    ("asset.criticality",
     "the business value of the affected host (e.g., domain controller vs developer laptop)"),
    ("asset.exposure",
     "whether the host is internet-facing or restricted to internal networks"),
    ("source.geo.country",
     "the geolocation of the source IP, used to evaluate impossible-travel and unusual-origin signals"),
    ("source.asn",
     "the autonomous system number of the source IP, identifying VPN/proxy/cloud-provider origins"),
    ("threat.indicator.confidence",
     "the confidence score on a matched IOC, used to weight the alert severity"),
    ("threat.indicator.last_seen",
     "the recency of the matched threat indicator; older IOCs have lower predictive value"),
    ("threat.framework.tactic",
     "the MITRE ATT&CK tactic the alert maps to, used for kill-chain coverage analysis"),
    ("threat.framework.technique",
     "the MITRE ATT&CK technique the alert maps to, used to chain alerts on the same actor"),
    ("vulnerability.id",
     "the CVE identifier exploited or referenced by the alert"),
    ("vulnerability.score (CVSS)",
     "the standardised severity score of the vulnerability, used to prioritise patching"),
    ("event.ingest.delay",
     "the time between event generation at the source and indexing in the SIEM; high values indicate ingestion lag"),
    ("user.risk.score",
     "an aggregate risk score derived from prior alerts, anomalies, and policy violations on the user"),
    ("host.risk.score",
     "an aggregate risk score for the host, derived from prior alerts, vulnerability state, and exposure"),
]

# (criterion, escalates_to)
ESCALATION: list[tuple[str, str]] = [
    ("confirmed credential dumping on a privileged account",
     "Tier 2 + IR-on-call (within 30 minutes)"),
    ("confirmed ransomware encryption activity on production",
     "IR-on-call + executive on-call + legal counsel (immediate)"),
    ("confirmed data-exfiltration of customer PII",
     "IR-on-call + privacy/legal + executive on-call (immediate)"),
    ("suspected business email compromise of an executive mailbox",
     "Tier 2 + IR-on-call + executive sponsor (within 1 hour)"),
    ("confirmed compromise of a domain controller",
     "IR-on-call + Active Directory team + executive on-call (immediate)"),
    ("confirmed compromise of a public-facing application",
     "Tier 2 + application owner + IR-on-call (within 1 hour)"),
    ("a single low-confidence anomaly on a non-critical host",
     "Tier 1 monitoring queue (no escalation; follow-up within 72 hours)"),
    ("repeated failed authentication from a single host within policy threshold",
     "Tier 1 automated response (account lockout) without escalation"),
    ("a confirmed false-positive after analyst review",
     "tuning queue (rule author / detection-engineering team) with no operational escalation"),
    ("regulated data exposure that meets statutory notification thresholds",
     "IR-on-call + privacy/legal + compliance officer (notification clock starts)"),
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

    # Pattern 1 -- severity definitions
    sev_names = [n for n, _ in SEVERITY_LEVELS]
    sev_defs = [d for _, d in SEVERITY_LEVELS]
    for sev, defn in SEVERITY_LEVELS:
        for phr in [f"Which alert severity tier is described as: {defn}?",
                    f"Which severity level corresponds to: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, sev, pick_distractors(rng, sev_names, sev),
                f"The '{sev}' severity tier is defined as: {defn}.",
                shortname, instruction))
        for phr in [f"What does the '{sev}' alert severity indicate?",
                    f"How is the '{sev}' alert severity tier defined in standard SOC playbooks?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, sev_defs, defn),
                f"The '{sev}' severity tier is defined as: {defn}.",
                shortname, instruction))

    # Pattern 2 -- TP/FP signal classification
    classes = ["true positive", "false positive"]
    reasons = [r for _, _, r in TP_FP_SIGNALS]
    for sig, cls, why in TP_FP_SIGNALS:
        for phr in [f"During alert triage, the signal '{sig}' is most likely a:",
                    f"How should an analyst classify the signal: '{sig}'?",
                    f"In SOC triage, '{sig}' most directly indicates which class of alert?"]:
            rows.append(make_mcq(
                rng, phr, cls, pick_distractors(rng, classes, cls),
                f"The signal '{sig}' is typically a {cls}: {why}.",
                shortname, instruction))
        for phr in [f"Why is the signal '{sig}' typically classified as a {cls}?"]:
            rows.append(make_mcq(
                rng, phr, why, pick_distractors(rng, reasons, why),
                f"The signal '{sig}' is typically a {cls}: {why}.",
                shortname, instruction))

    # Pattern 3 -- enrichment fields
    enr_names = [n for n, _ in ENRICHMENT_FIELDS]
    enr_defs = [d for _, d in ENRICHMENT_FIELDS]
    for fname, defn in ENRICHMENT_FIELDS:
        for phr in [f"Which alert-enrichment field provides: {defn}?",
                    f"Which enrichment field corresponds to: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, fname, pick_distractors(rng, enr_names, fname),
                f"The enrichment field '{fname}' provides {defn}.",
                shortname, instruction))
        for phr in [f"What does the enrichment field '{fname}' provide to the analyst?",
                    f"How is the enrichment field '{fname}' used in alert triage?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, enr_defs, defn),
                f"The enrichment field '{fname}' provides {defn}.",
                shortname, instruction))

    # Pattern 4 -- escalation
    esc_targets = list({t for _, t in ESCALATION})
    esc_crits = [c for c, _ in ESCALATION]
    for crit, target in ESCALATION:
        for phr in [f"Per the standard SOC escalation matrix, '{crit}' should be escalated to:",
                    f"Which escalation path applies to: '{crit}'?",
                    f"In SOC triage, the appropriate escalation for '{crit}' is:"]:
            rows.append(make_mcq(
                rng, phr, target, pick_distractors(rng, esc_targets, target),
                f"Per the standard SOC escalation matrix, '{crit}' is escalated to: {target}.",
                shortname, instruction))
        for phr in [f"Which incident criterion warrants escalation to '{target}'?"]:
            rows.append(make_mcq(
                rng, phr, crit, pick_distractors(rng, esc_crits, crit),
                f"Per the standard SOC escalation matrix, '{crit}' is escalated to: {target}.",
                shortname, instruction))

    return rows
