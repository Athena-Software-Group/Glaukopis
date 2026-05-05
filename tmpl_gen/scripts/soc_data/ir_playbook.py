"""Incident-response playbook knowledge for SOC.IR.GEN.1. Curated from
NIST SP 800-61 Rev 2, NIST SP 800-86 (forensic guide), the SANS PICERL
model, and CISA scenario playbooks. Encodes per-scenario actions,
evidence-collection ordering, and common artifact sources.
"""

from __future__ import annotations

from collections.abc import Callable

from . import pick_distractors

# (scenario, phase, action) -- which IR phase action belongs to which scenario
SCENARIO_ACTIONS: list[tuple[str, str, str]] = [
    ("ransomware", "Containment",
     "isolate affected hosts from the network at the switch port to prevent encryption spread"),
    ("ransomware", "Containment",
     "block the ransomware operator's C2 domains/IPs at the perimeter and DNS resolver"),
    ("ransomware", "Containment",
     "disable compromised privileged accounts and force a tenant-wide privileged-credential rotation"),
    ("ransomware", "Eradication",
     "remove the persistence mechanism (scheduled task, service, registry run key) before bringing hosts back"),
    ("ransomware", "Recovery",
     "restore from offline immutable backups and validate integrity before reconnecting to production"),
    ("ransomware", "Detection and Analysis",
     "identify the initial-access vector, the ransomware family, and the encryption time window"),
    ("ransomware", "Post-Incident",
     "produce a lessons-learned report including dwell time, MTTR, and gaps in EDR/backup posture"),

    ("business email compromise (BEC)", "Containment",
     "revoke all sessions and refresh tokens for the compromised mailbox to invalidate stolen tokens"),
    ("business email compromise (BEC)", "Containment",
     "remove malicious mail-flow rules (forward, delete) the attacker created in the victim mailbox"),
    ("business email compromise (BEC)", "Detection and Analysis",
     "review unified audit log for inbox-rule creation, eDiscovery searches, and suspicious sign-ins"),
    ("business email compromise (BEC)", "Detection and Analysis",
     "identify the wire-fraud target, the spoofed counterparty, and the attacker-controlled bank account"),
    ("business email compromise (BEC)", "Eradication",
     "block the attacker's IPs and OAuth-app registrations across the tenant"),
    ("business email compromise (BEC)", "Recovery",
     "rotate the user's password, regenerate any app passwords, and re-enrol MFA factors"),

    ("account takeover", "Containment",
     "revoke all active sessions and refresh tokens to invalidate the attacker's authenticated state"),
    ("account takeover", "Detection and Analysis",
     "review the user's sign-in log for impossible-travel, unfamiliar location, and risky-IP signals"),
    ("account takeover", "Eradication",
     "rotate the user's password and enrol or re-enrol a strong MFA factor (FIDO2 preferred)"),
    ("account takeover", "Eradication",
     "remove any OAuth consents the attacker granted on behalf of the user"),

    ("data exfiltration", "Detection and Analysis",
     "quantify the volume, sensitivity classification, and external destination of the exfiltrated data"),
    ("data exfiltration", "Detection and Analysis",
     "identify the staging directory, archive format, and exfiltration channel (DNS, HTTPS, S3, cloud-share)"),
    ("data exfiltration", "Containment",
     "block egress to the exfiltration destination at the perimeter and disable the involved service tokens"),
    ("data exfiltration", "Eradication",
     "remove the staging tools (rclone, MEGAcmd, custom uploaders) and any persistence on the staging host"),
    ("data exfiltration", "Post-Incident",
     "trigger regulatory notifications within statutory deadlines (GDPR 72h, HIPAA 60d, etc.) for in-scope data"),

    ("web shell", "Detection and Analysis",
     "identify the web-shell file path, parent web-server process, and first-write timestamp"),
    ("web shell", "Containment",
     "remove the web shell, take the application offline if exploitation persists, and patch the underlying CVE"),
    ("web shell", "Eradication",
     "audit the web root for additional implants and reset any credentials accessible from the web-server identity"),
    ("web shell", "Recovery",
     "rebuild the web server from a known-good gold image rather than reusing the compromised host"),

    ("lateral movement", "Detection and Analysis",
     "construct the attacker timeline using authentication logs, EDR process trees, and SMB/RDP session events"),
    ("lateral movement", "Containment",
     "isolate the suspected source and destination hosts and reset any traversed service-account credentials"),
    ("lateral movement", "Eradication",
     "remove any newly-created scheduled tasks, services, or registry-run keys on each traversed host"),

    ("DDoS", "Containment",
     "engage the upstream provider's scrubbing service and apply rate-limits at the L7 reverse proxy"),
    ("DDoS", "Detection and Analysis",
     "identify the attack vector (UDP amplification, SYN flood, HTTP slowloris, application-layer)"),
    ("DDoS", "Recovery",
     "restore service capacity progressively and retain attack telemetry for after-action analysis"),
]

# NIST SP 800-86 evidence collection order of volatility (most to least)
VOLATILITY_ORDER: list[tuple[int, str, str]] = [
    (1, "CPU registers and cache", "the most volatile state, surviving only nanoseconds"),
    (2, "memory contents (RAM)", "lost on power loss, may contain plaintext keys, decrypted payloads, network sockets"),
    (3, "network state and routing tables", "ARP cache, routing table, open sockets, listening ports"),
    (4, "running processes and open files", "process list with command lines, loaded modules, open handles"),
    (5, "temporary file systems", "/tmp, %TEMP%, swap/pagefile content"),
    (6, "non-volatile storage (disk)", "filesystem contents and free-space residue, surviving reboots"),
    (7, "remote logging and monitoring data", "SIEM, syslog, EDR backend telemetry off-host"),
    (8, "physical configuration and network topology", "patch panels, cable runs, hardware inventory"),
    (9, "archival media", "long-term backups and offline cold storage"),
]

# Common forensic artefact sources (artefact, what it tells you)
ARTEFACTS: list[tuple[str, str]] = [
    ("MFT ($MFT)",       "directory and file metadata for every file ever created on an NTFS volume"),
    ("USN journal ($UsnJrnl)", "ordered log of every change to files and directories on an NTFS volume"),
    ("LogFile ($LogFile)", "NTFS transactional log used for filesystem journalling and recovery"),
    ("ShellBags",        "Windows registry record of which directories the user browsed via Explorer"),
    ("ShimCache (AppCompatCache)", "list of executables that ran or were browsed, with last-modified timestamps"),
    ("Amcache (Amcache.hve)", "registry hive listing executed binaries, their SHA1, and metadata; richer than ShimCache"),
    ("Prefetch (.pf)",   "Windows file recording the first 10 seconds of an executable's execution to speed launch"),
    ("UserAssist",       "registry record of GUI-launched executables under the user's HKCU hive"),
    ("RecentDocs",       "registry record of recently opened documents per file extension under HKCU"),
    ("Jump Lists",       "per-application history of files the user opened recently"),
    ("Browser history",  "URLs the user visited, downloads, cookies, cache content"),
    ("Event Log Security", "Windows authentication, privilege-use, and policy-change events (4624, 4625, 4672, 4720)"),
    ("Event Log System", "Windows service, driver, and time-change events (7045, 7036)"),
    ("Sysmon Operational", "rich endpoint telemetry: process, network, file, registry, image-load, named-pipe events"),
    ("Powershell Operational", "deobfuscated PowerShell script-block content (EID 4104) and module logging (EID 4103)"),
    ("auth.log / secure", "Linux authentication events (sshd, sudo, su, login)"),
    ("auditd logs",      "Linux kernel audit records of syscalls, file access, and security-relevant events"),
    ("bash_history / .zsh_history", "interactive shell command history per user"),
    ("syslog / journalctl", "Linux system messages aggregated by rsyslog or systemd-journald"),
    ("Unified Audit Log (Microsoft 365)", "tenant-wide log of admin and user activity across Exchange, SharePoint, Teams"),
    ("CloudTrail (AWS)", "API-call audit log for every AWS API request, by principal, source IP, and parameters"),
    ("Azure Activity Log", "control-plane audit log of management operations against Azure resources"),
    ("VPC Flow Logs (AWS)", "5-tuple network connection metadata at the VPC ENI level"),
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

    # Pattern 1 -- scenario actions per phase
    scenarios = sorted({s for s, _, _ in SCENARIO_ACTIONS})
    phases = sorted({p for _, p, _ in SCENARIO_ACTIONS})
    actions_pool = [a for _, _, a in SCENARIO_ACTIONS]
    for scen, phase, action in SCENARIO_ACTIONS:
        for phr in [f"During the {phase} phase of a {scen} incident, which of the following actions is most appropriate?",
                    f"In NIST SP 800-61 terms, what {phase}-phase action best fits a {scen} incident?",
                    f"Which {phase}-phase step is standard for a {scen} response?"]:
            rows.append(make_mcq(
                rng, phr, action, pick_distractors(rng, actions_pool, action),
                f"In a {scen} incident, the {phase} phase includes the action: {action}.",
                shortname, instruction))
        for phr in [f"Which incident-response phase covers the action: '{action}'?",
                    f"Per NIST SP 800-61, the action '{action}' belongs to which IR phase?"]:
            rows.append(make_mcq(
                rng, phr, phase, pick_distractors(rng, phases, phase),
                f"In a {scen} incident, the {phase} phase includes the action: {action}.",
                shortname, instruction))
        for phr in [f"Which incident scenario does the action '{action}' most directly fit?"]:
            rows.append(make_mcq(
                rng, phr, scen, pick_distractors(rng, scenarios, scen),
                f"The action '{action}' is a standard {phase}-phase step in a {scen} incident.",
                shortname, instruction))

    # Pattern 2 -- volatility ordering
    sources = [s for _, s, _ in VOLATILITY_ORDER]
    defs = [d for _, _, d in VOLATILITY_ORDER]
    for rank, src, defn in VOLATILITY_ORDER:
        for phr in [f"At what position does '{src}' appear in the NIST SP 800-86 order of volatility (1 = most volatile)?"]:
            ranks = [str(r) for r, _, _ in VOLATILITY_ORDER]
            rows.append(make_mcq(
                rng, phr, str(rank), pick_distractors(rng, ranks, str(rank)),
                f"In NIST SP 800-86, '{src}' is at position {rank} in the order of volatility.",
                shortname, instruction))
        for phr in [f"Which evidence source is described by: '{defn}'?",
                    f"Which volatile data source is: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, src, pick_distractors(rng, sources, src),
                f"In NIST SP 800-86, '{src}' refers to {defn}.",
                shortname, instruction))
        for phr in [f"What does the evidence source '{src}' contain?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, defs, defn),
                f"'{src}' refers to {defn}.",
                shortname, instruction))

    # Pattern 3 -- forensic artefacts
    art_names = [n for n, _ in ARTEFACTS]
    art_defs = [d for _, d in ARTEFACTS]
    for name, defn in ARTEFACTS:
        for phr in [f"Which forensic artefact contains: {defn}?",
                    f"Which evidence source provides: {defn}?",
                    f"Which artefact does the following describe: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, art_names, name),
                f"The forensic artefact '{name}' contains {defn}.",
                shortname, instruction))
        for phr in [f"What does the {name} forensic artefact record?",
                    f"What information is captured in {name}?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, art_defs, defn),
                f"The forensic artefact '{name}' contains {defn}.",
                shortname, instruction))

    return rows

