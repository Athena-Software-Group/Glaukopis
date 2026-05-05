"""MITRE ATT&CK knowledge edges NOT traversed by AB.MCQ.* / JS.MCQ.*
templates: technique<->platform, sub-technique<->parent, software<->
platform, software<->category, kill-chain phase mappings, defense-evaded
relationships. Curated from MITRE ATT&CK Enterprise v15+; facts are
verifiable against attack.mitre.org.
"""

from __future__ import annotations

from collections.abc import Callable

from . import pick_distractors

# Tactic -> kill-chain phase mapping (Lockheed Martin Cyber Kill Chain
# alignment is a common MCQ pattern; not in templates).
TACTIC_KCP: list[tuple[str, str]] = [
    ("Reconnaissance", "Reconnaissance"),
    ("Resource Development", "Weaponization"),
    ("Initial Access", "Delivery / Exploitation"),
    ("Execution", "Exploitation / Installation"),
    ("Persistence", "Installation"),
    ("Privilege Escalation", "Installation / C2"),
    ("Defense Evasion", "Installation / Actions on Objectives"),
    ("Credential Access", "Actions on Objectives"),
    ("Discovery", "Actions on Objectives"),
    ("Lateral Movement", "Actions on Objectives"),
    ("Collection", "Actions on Objectives"),
    ("Command and Control", "Command and Control"),
    ("Exfiltration", "Actions on Objectives"),
    ("Impact", "Actions on Objectives"),
]

# Technique -> primary applicable platforms (MITRE ATT&CK Enterprise).
TECH_PLATFORMS: list[tuple[str, str, list[str]]] = [
    ("T1059.001", "PowerShell", ["Windows"]),
    ("T1059.003", "Windows Command Shell", ["Windows"]),
    ("T1059.004", "Unix Shell", ["Linux", "macOS"]),
    ("T1059.005", "Visual Basic", ["Windows"]),
    ("T1059.006", "Python", ["Linux", "Windows", "macOS"]),
    ("T1547.001", "Registry Run Keys / Startup Folder", ["Windows"]),
    ("T1543.003", "Windows Service", ["Windows"]),
    ("T1543.001", "Launch Agent", ["macOS"]),
    ("T1543.002", "Systemd Service", ["Linux"]),
    ("T1053.005", "Scheduled Task", ["Windows"]),
    ("T1053.003", "Cron", ["Linux", "macOS"]),
    ("T1218.011", "Rundll32", ["Windows"]),
    ("T1218.005", "Mshta", ["Windows"]),
    ("T1140",     "Deobfuscate/Decode Files or Information", ["Linux", "Windows", "macOS"]),
    ("T1027",     "Obfuscated Files or Information", ["Linux", "Windows", "macOS"]),
    ("T1110",     "Brute Force", ["Linux", "Windows", "macOS", "Office 365", "Azure AD", "SaaS"]),
    ("T1003.001", "LSASS Memory", ["Windows"]),
    ("T1003.008", "/etc/passwd and /etc/shadow", ["Linux"]),
    ("T1098",     "Account Manipulation", ["Linux", "Windows", "macOS", "Office 365", "Azure AD", "SaaS", "IaaS"]),
    ("T1136.003", "Cloud Account", ["Office 365", "Azure AD", "IaaS", "SaaS", "Google Workspace"]),
    ("T1078.004", "Cloud Accounts", ["Office 365", "Azure AD", "IaaS", "SaaS", "Google Workspace"]),
    ("T1190",     "Exploit Public-Facing Application", ["Linux", "Windows", "macOS", "Network", "Containers", "IaaS"]),
    ("T1133",     "External Remote Services", ["Windows", "Linux", "macOS", "Containers"]),
    ("T1566.001", "Spearphishing Attachment", ["Linux", "Windows", "macOS", "Office 365", "Google Workspace"]),
    ("T1566.002", "Spearphishing Link", ["Linux", "Windows", "macOS", "Office 365", "Google Workspace"]),
    ("T1486",     "Data Encrypted for Impact", ["Linux", "Windows", "macOS", "IaaS"]),
    ("T1490",     "Inhibit System Recovery", ["Linux", "Windows", "macOS"]),
    ("T1485",     "Data Destruction", ["Linux", "Windows", "macOS", "IaaS"]),
    ("T1610",     "Deploy Container", ["Containers"]),
    ("T1611",     "Escape to Host", ["Containers"]),
    ("T1613",     "Container and Resource Discovery", ["Containers"]),
]

# Sub-technique <-> parent technique mapping (anchor diversity gap in templates).
SUBTECH_PARENT: list[tuple[str, str, str, str]] = [
    ("T1059.001", "PowerShell", "T1059", "Command and Scripting Interpreter"),
    ("T1059.003", "Windows Command Shell", "T1059", "Command and Scripting Interpreter"),
    ("T1059.004", "Unix Shell", "T1059", "Command and Scripting Interpreter"),
    ("T1003.001", "LSASS Memory", "T1003", "OS Credential Dumping"),
    ("T1003.002", "Security Account Manager", "T1003", "OS Credential Dumping"),
    ("T1003.003", "NTDS", "T1003", "OS Credential Dumping"),
    ("T1547.001", "Registry Run Keys / Startup Folder", "T1547", "Boot or Logon Autostart Execution"),
    ("T1547.009", "Shortcut Modification", "T1547", "Boot or Logon Autostart Execution"),
    ("T1543.001", "Launch Agent", "T1543", "Create or Modify System Process"),
    ("T1543.002", "Systemd Service", "T1543", "Create or Modify System Process"),
    ("T1543.003", "Windows Service", "T1543", "Create or Modify System Process"),
    ("T1053.005", "Scheduled Task", "T1053", "Scheduled Task/Job"),
    ("T1053.003", "Cron", "T1053", "Scheduled Task/Job"),
    ("T1218.005", "Mshta", "T1218", "System Binary Proxy Execution"),
    ("T1218.011", "Rundll32", "T1218", "System Binary Proxy Execution"),
    ("T1566.001", "Spearphishing Attachment", "T1566", "Phishing"),
    ("T1566.002", "Spearphishing Link", "T1566", "Phishing"),
    ("T1566.003", "Spearphishing via Service", "T1566", "Phishing"),
    ("T1078.001", "Default Accounts", "T1078", "Valid Accounts"),
    ("T1078.002", "Domain Accounts", "T1078", "Valid Accounts"),
    ("T1078.003", "Local Accounts", "T1078", "Valid Accounts"),
    ("T1078.004", "Cloud Accounts", "T1078", "Valid Accounts"),
    ("T1110.001", "Password Guessing", "T1110", "Brute Force"),
    ("T1110.002", "Password Cracking", "T1110", "Brute Force"),
    ("T1110.003", "Password Spraying", "T1110", "Brute Force"),
    ("T1110.004", "Credential Stuffing", "T1110", "Brute Force"),
    ("T1071.001", "Web Protocols", "T1071", "Application Layer Protocol"),
    ("T1071.004", "DNS", "T1071", "Application Layer Protocol"),
    ("T1071.002", "File Transfer Protocols", "T1071", "Application Layer Protocol"),
    ("T1090.001", "Internal Proxy", "T1090", "Proxy"),
    ("T1090.003", "Multi-hop Proxy", "T1090", "Proxy"),
    ("T1574.001", "DLL Search Order Hijacking", "T1574", "Hijack Execution Flow"),
    ("T1574.002", "DLL Side-Loading", "T1574", "Hijack Execution Flow"),
]

# Software (S####) -> category and primary platform set (S0002 is Mimikatz, etc.).
SOFTWARE_CATEGORY: list[tuple[str, str, str, list[str]]] = [
    ("S0002", "Mimikatz",     "credential access tool", ["Windows"]),
    ("S0154", "Cobalt Strike", "post-exploitation framework / C2 platform", ["Linux", "Windows", "macOS"]),
    ("S0357", "Empire",       "post-exploitation framework", ["Linux", "Windows", "macOS"]),
    ("S0521", "BloodHound",   "Active Directory reconnaissance tool", ["Windows"]),
    ("S0029", "PsExec",       "remote-execution dual-use admin utility", ["Windows"]),
    ("S0008", "gsecdump",     "credential dumping utility", ["Windows"]),
    ("S0029", "PsExec",       "remote-execution dual-use admin utility", ["Windows"]),
    ("S0552", "AdFind",       "Active Directory enumeration utility", ["Windows"]),
    ("S0194", "PowerSploit",  "PowerShell post-exploitation framework", ["Windows"]),
    ("S0508", "ngrok",        "reverse-tunneling utility (dual-use)", ["Linux", "Windows", "macOS"]),
    ("S0363", "Empire",       "post-exploitation framework", ["Linux", "Windows", "macOS"]),
    ("S0367", "Emotet",       "loader / banking trojan", ["Windows"]),
    ("S0266", "TrickBot",     "modular banking trojan and loader", ["Windows"]),
    ("S0521", "BloodHound",   "Active Directory reconnaissance tool", ["Windows"]),
    ("S0650", "QakBot",       "modular banking trojan and loader", ["Windows"]),
    ("S0365", "Olympic Destroyer", "destructive wiper", ["Windows"]),
    ("S0606", "Bad Rabbit",   "destructive ransomware", ["Windows"]),
    ("S0496", "REvil",        "ransomware-as-a-service payload", ["Windows"]),
    ("S0446", "Ryuk",         "targeted ransomware payload", ["Windows"]),
    ("S1068", "BlackCat",     "ransomware-as-a-service payload (Rust)", ["Linux", "Windows"]),
    ("S0617", "HermeticWiper", "destructive wiper", ["Windows"]),
    ("S0640", "Avaddon",      "ransomware-as-a-service payload", ["Windows"]),
    ("S0260", "InvisiMole",   "modular spyware framework", ["Windows"]),
    ("S0568", "EvilBunny",    "scripting-based RAT", ["Windows"]),
    ("S0089", "BlackEnergy",  "modular malware framework", ["Windows"]),
    ("S0367", "Emotet",       "loader / banking trojan", ["Windows"]),
]

# Defense-evasion technique -> primary defensive control bypassed.
DEFENSE_BYPASSED: list[tuple[str, str]] = [
    ("T1027 -- Obfuscated Files or Information", "static signature-based antivirus detection"),
    ("T1140 -- Deobfuscate/Decode Files or Information", "static content inspection of payload-on-disk"),
    ("T1218 -- System Binary Proxy Execution", "application-allowlisting that trusts signed system binaries"),
    ("T1562.001 -- Disable or Modify Tools", "endpoint security agents / EDR sensors"),
    ("T1562.002 -- Disable Windows Event Logging", "Windows Event Log-based detection and forensics"),
    ("T1562.004 -- Disable or Modify System Firewall", "host-based firewall egress controls"),
    ("T1497 -- Virtualization/Sandbox Evasion", "automated sandbox malware analysis pipelines"),
    ("T1055 -- Process Injection", "process-image / PE-on-disk based detection"),
    ("T1036 -- Masquerading", "name- and path-based filename allowlists"),
    ("T1078 -- Valid Accounts", "anomaly-based behavioural detection of unauthenticated access"),
    ("T1574.002 -- DLL Side-Loading", "Authenticode signing checks on the loaded executable"),
    ("T1070.001 -- Clear Windows Event Logs", "Windows Event Log-based forensic timeline reconstruction"),
    ("T1070.004 -- File Deletion", "host-based file integrity monitoring on dropped artifacts"),
    ("T1553.002 -- Code Signing", "Authenticode trust chain enforcement on PE files"),
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

    # Pattern 1 -- tactic -> kill-chain phase
    tac_names = [t for t, _ in TACTIC_KCP]
    kcp_names = list({k for _, k in TACTIC_KCP})
    for tac, kcp in TACTIC_KCP:
        for phr in [f"In the Lockheed Martin Cyber Kill Chain mapping, the MITRE ATT&CK '{tac}' tactic most closely corresponds to which phase?",
                    f"Which kill-chain phase aligns with the MITRE ATT&CK tactic '{tac}'?",
                    f"Per the standard ATT&CK<->Kill Chain mapping, '{tac}' is part of which phase?"]:
            rows.append(make_mcq(
                rng, phr, kcp, pick_distractors(rng, kcp_names, kcp),
                f"The MITRE ATT&CK '{tac}' tactic maps to the '{kcp}' phase of the Lockheed Martin Cyber Kill Chain.",
                shortname, instruction))

    # Pattern 2 -- technique -> applicable platforms
    plat_pool = sorted({p for _, _, ps in TECH_PLATFORMS for p in ps})
    for mid, name, plats in TECH_PLATFORMS:
        canon = ", ".join(plats)
        if len(plats) == 1:
            wrong_pool = [p for p in plat_pool if p not in plats]
            for phr in [f"Which platform is technique {mid} ({name}) primarily applicable to per MITRE ATT&CK?",
                        f"On which OS/platform is {mid} ({name}) documented as applicable?",
                        f"Which platform does MITRE ATT&CK list as the primary scope of {mid} ({name})?"]:
                rows.append(make_mcq(
                    rng, phr, plats[0], pick_distractors(rng, wrong_pool, plats[0]),
                    f"MITRE ATT&CK lists {mid} ({name}) as applicable to {canon}.",
                    shortname, instruction))
        for phr in [f"Which statement about MITRE ATT&CK technique {mid} ({name}) is correct?",
                    f"What does MITRE ATT&CK document about {mid} ({name})?"]:
            correct = f"applicable to {canon}"
            wrong_set = []
            for p in plat_pool:
                if p not in plats:
                    wrong_set.append(f"applicable to {p}")
                if len(wrong_set) >= 8:
                    break
            rows.append(make_mcq(
                rng, phr, correct, pick_distractors(rng, wrong_set, correct),
                f"MITRE ATT&CK lists {mid} ({name}) as applicable to {canon}.",
                shortname, instruction))

    # Pattern 3 -- sub-technique -> parent
    par_names = sorted({f"{p_id} ({p_name})" for _, _, p_id, p_name in SUBTECH_PARENT})
    for sid, sname, pid, pname in SUBTECH_PARENT:
        correct = f"{pid} ({pname})"
        for phr in [f"Which parent technique does sub-technique {sid} ({sname}) belong to?",
                    f"Per MITRE ATT&CK, the sub-technique {sid} ({sname}) is a sub-technique of which parent?",
                    f"Which ATT&CK parent technique covers the sub-technique {sid} ({sname})?"]:
            rows.append(make_mcq(
                rng, phr, correct, pick_distractors(rng, par_names, correct),
                f"In MITRE ATT&CK, {sid} ({sname}) is a sub-technique of {pid} ({pname}).",
                shortname, instruction))

    # Pattern 4 -- software -> category
    cat_pool = sorted({cat for _, _, cat, _ in SOFTWARE_CATEGORY})
    seen_sw = set()
    for sid, sname, cat, plats in SOFTWARE_CATEGORY:
        key = (sid, sname)
        if key in seen_sw:
            continue
        seen_sw.add(key)
        for phr in [f"How is {sname} ({sid}) primarily categorised by MITRE ATT&CK?",
                    f"Which best describes the function of {sname} ({sid})?",
                    f"What kind of tool/software is {sname} ({sid}) per MITRE ATT&CK?"]:
            rows.append(make_mcq(
                rng, phr, cat, pick_distractors(rng, cat_pool, cat),
                f"MITRE ATT&CK categorises {sname} ({sid}) as a {cat}.",
                shortname, instruction))

    # Pattern 5 -- defense-evasion -> control bypassed
    ctrl_pool = [c for _, c in DEFENSE_BYPASSED]
    tech_pool = [t for t, _ in DEFENSE_BYPASSED]
    for tech, ctrl in DEFENSE_BYPASSED:
        for phr in [f"Which defensive control is most directly bypassed by ATT&CK technique {tech}?",
                    f"What detection/control category does {tech} aim to evade?",
                    f"Which defensive control does {tech} most directly undermine?"]:
            rows.append(make_mcq(
                rng, phr, ctrl, pick_distractors(rng, ctrl_pool, ctrl),
                f"The ATT&CK defense-evasion technique {tech} is used to bypass {ctrl}.",
                shortname, instruction))
        for phr in [f"Which ATT&CK technique is most commonly used to bypass {ctrl}?",
                    f"Which defense-evasion technique would an adversary use against {ctrl}?"]:
            rows.append(make_mcq(
                rng, phr, tech, pick_distractors(rng, tech_pool, tech),
                f"The ATT&CK defense-evasion technique {tech} is used to bypass {ctrl}.",
                shortname, instruction))

    return rows

