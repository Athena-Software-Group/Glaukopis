"""General application/network/OS security MCQ knowledge tables that
sit outside the MITRE ATT&CK ontology (covered by AB.MCQ.*) and outside
the CM crypto/access/compliance/governance scope (covered by CM.*).
Topics: OWASP Top-10 2021 + API Security Top-10, web vulnerability
classes, network protocol fundamentals, OS-security primitives.
"""

from __future__ import annotations

from collections.abc import Callable

from . import pick_distractors

# OWASP Top-10 2021 (id, name, definition)
OWASP_TOP10: list[tuple[str, str, str]] = [
    ("A01:2021", "Broken Access Control",
     "weaknesses in enforcing user permissions, allowing actions outside intended privileges"),
    ("A02:2021", "Cryptographic Failures",
     "failures related to cryptography that often lead to exposure of sensitive data"),
    ("A03:2021", "Injection",
     "user-supplied data is sent to an interpreter as part of a command or query"),
    ("A04:2021", "Insecure Design",
     "missing or ineffective control design that cannot be fixed by perfect implementation"),
    ("A05:2021", "Security Misconfiguration",
     "missing security hardening, insecure default configurations, or verbose error messages"),
    ("A06:2021", "Vulnerable and Outdated Components",
     "use of components with known vulnerabilities or unsupported versions"),
    ("A07:2021", "Identification and Authentication Failures",
     "weaknesses in confirming user identity, authentication, and session management"),
    ("A08:2021", "Software and Data Integrity Failures",
     "code and infrastructure that does not protect against integrity violations such as unsigned updates"),
    ("A09:2021", "Security Logging and Monitoring Failures",
     "insufficient logging, detection, monitoring, and active response to attacks"),
    ("A10:2021", "Server-Side Request Forgery (SSRF)",
     "the web application fetches a remote resource without validating the user-supplied URL"),
]

# OWASP API Security Top-10 2023 (id, name, definition)
OWASP_API: list[tuple[str, str, str]] = [
    ("API1:2023", "Broken Object Level Authorization",
     "API endpoints fail to verify that the authenticated user is allowed to access the requested object"),
    ("API2:2023", "Broken Authentication",
     "incorrectly implemented authentication mechanisms allow attackers to assume other users' identities"),
    ("API3:2023", "Broken Object Property Level Authorization",
     "API exposes object properties the user should not be able to read or modify"),
    ("API4:2023", "Unrestricted Resource Consumption",
     "lack of rate-limit, payload-size, or query-cost controls enables denial-of-service"),
    ("API5:2023", "Broken Function Level Authorization",
     "improper function-level access controls expose privileged functionality to lower-privileged users"),
    ("API6:2023", "Unrestricted Access to Sensitive Business Flows",
     "automation against sensitive flows (purchase, password reset) is not throttled or gated"),
    ("API7:2023", "Server-Side Request Forgery",
     "the API fetches a user-supplied URL without proper validation"),
    ("API8:2023", "Security Misconfiguration",
     "insecure defaults, verbose errors, missing security headers, or open CORS policies"),
    ("API9:2023", "Improper Inventory Management",
     "outdated, unused, or undocumented API versions and endpoints remain exposed"),
    ("API10:2023", "Unsafe Consumption of APIs",
     "developers trust data from third-party APIs without sufficient validation"),
]

# Web vulnerability classes (name, primary mitigation, attack mechanism)
WEB_VULNS: list[tuple[str, str, str]] = [
    ("SQL Injection (SQLi)",
     "use parameterised queries / prepared statements",
     "untrusted input is concatenated into a SQL statement, altering its structure"),
    ("Cross-Site Scripting (Reflected XSS)",
     "context-aware output encoding and a strict Content Security Policy",
     "untrusted input is reflected in the response and executes in the victim's browser"),
    ("Cross-Site Scripting (Stored XSS)",
     "context-aware output encoding on render plus input validation",
     "untrusted input is persisted and later rendered to other users"),
    ("Cross-Site Request Forgery (CSRF)",
     "synchroniser tokens or SameSite=Strict cookies",
     "an authenticated victim's browser is tricked into issuing an unintended state-changing request"),
    ("Server-Side Request Forgery (SSRF)",
     "outbound network egress allowlists and metadata service blocking (IMDSv2)",
     "the server fetches a user-supplied URL, often reaching internal-only resources"),
    ("XML External Entity (XXE)",
     "disable DTD processing in the XML parser",
     "the XML parser resolves external entities defined by the attacker, leaking files or causing SSRF"),
    ("Insecure Deserialization",
     "avoid deserialising untrusted data; use signed, schema-validated formats",
     "an attacker controls a serialised payload that triggers code execution on deserialisation"),
    ("Open Redirect",
     "validate the redirect target against an allowlist of internal URLs",
     "the application redirects to a user-supplied URL, enabling phishing or token leakage"),
    ("Path Traversal",
     "canonicalise paths and confine reads to a chroot/jailed directory",
     "user input is used to construct a filesystem path and escapes the intended directory"),
    ("Command Injection",
     "use exec-style APIs that take an argv list rather than a shell string",
     "user input is concatenated into a shell command, allowing arbitrary command execution"),
    ("LDAP Injection",
     "escape LDAP metacharacters with bind-style queries",
     "user input is concatenated into an LDAP filter, altering authentication or query results"),
    ("Mass Assignment",
     "explicit allowlist of bindable properties (DTO pattern)",
     "the framework auto-binds request fields to internal model attributes, including privileged ones"),
    ("Insecure Direct Object Reference (IDOR)",
     "per-request authorisation check that the user owns the referenced object",
     "the application uses a user-supplied identifier to fetch an object without an ownership check"),
    ("Clickjacking",
     "X-Frame-Options DENY or Content-Security-Policy frame-ancestors 'none'",
     "the application can be framed by an attacker page that overlays UI elements"),
]

# Network protocol facts (protocol, port_default, fact)
NETWORK_PROTOS: list[tuple[str, str, str]] = [
    ("HTTP",   "TCP/80",   "is a stateless application-layer protocol used by the World Wide Web"),
    ("HTTPS",  "TCP/443",  "is HTTP over a TLS-protected transport providing confidentiality and integrity"),
    ("SSH",    "TCP/22",   "provides authenticated encrypted remote shell access and tunneling"),
    ("Telnet", "TCP/23",   "is an unencrypted remote-login protocol now considered insecure"),
    ("DNS",    "UDP/53",   "resolves human-readable names to IP addresses using a hierarchical namespace"),
    ("DoH",    "TCP/443",  "tunnels DNS queries inside HTTPS to defeat passive on-path inspection"),
    ("DoT",    "TCP/853",  "encrypts DNS queries inside a dedicated TLS session"),
    ("SMTP",   "TCP/25",   "is the application-layer protocol for relaying outbound email between mail servers"),
    ("IMAP",   "TCP/143",  "lets a client read mail kept on the server, supporting folders and partial fetches"),
    ("POP3",   "TCP/110",  "lets a client download mail from the server, traditionally deleting it after retrieval"),
    ("FTP",    "TCP/21",   "is an unencrypted file-transfer protocol that uses separate control and data channels"),
    ("SFTP",   "TCP/22",   "provides file transfer over an SSH session"),
    ("RDP",    "TCP/3389", "is the Microsoft remote-desktop protocol"),
    ("SMB",    "TCP/445",  "is the Microsoft file- and printer-sharing protocol"),
    ("LDAP",   "TCP/389",  "is the directory access protocol used to query and modify directory services"),
    ("LDAPS",  "TCP/636",  "is LDAP over TLS"),
    ("Kerberos", "TCP/UDP 88", "is the network authentication protocol used by Active Directory"),
    ("NTP",    "UDP/123",  "is the protocol used to synchronise clocks across networked systems"),
    ("SNMP",   "UDP/161",  "is the protocol used to monitor and manage network devices"),
    ("Syslog", "UDP/514",  "is the de facto Unix protocol for forwarding log messages between systems"),
    ("BGP",    "TCP/179",  "is the inter-domain routing protocol of the public internet"),
    ("TLS 1.2", "varies",   "uses an RSA or (EC)DHE key exchange and a separate MAC; superseded by 1.3 in 2018"),
    ("TLS 1.3", "varies",   "removes RSA key exchange and uses AEAD ciphers exclusively, completing the handshake in 1-RTT"),
    ("WPA2",   "n/a",      "uses AES-CCMP for confidentiality and integrity in 802.11 wireless networks"),
    ("WPA3",   "n/a",      "uses Simultaneous Authentication of Equals (SAE) instead of WPA2's pre-shared-key handshake"),
]

# OS / endpoint security primitives (name, definition)
OS_PRIMITIVES: list[tuple[str, str]] = [
    ("ASLR (Address Space Layout Randomization)",
     "randomises the base addresses of code and data sections to make memory-corruption exploits less reliable"),
    ("DEP / NX (Data Execution Prevention)",
     "marks data pages as non-executable to defeat classic stack/heap shellcode execution"),
    ("Stack Canary",
     "places a known value before the saved return address to detect linear stack-buffer overflows on return"),
    ("Control Flow Integrity (CFI)",
     "constrains indirect branches to a static set of valid targets, blocking ROP/JOP-style hijacks"),
    ("SELinux",
     "is a Linux kernel security module implementing mandatory access control via type enforcement"),
    ("AppArmor",
     "is a Linux kernel security module implementing mandatory access control via per-binary path-based profiles"),
    ("seccomp-bpf",
     "is a Linux kernel facility that filters syscalls a process may make, reducing kernel attack surface"),
    ("Linux capabilities",
     "split the traditional all-or-nothing root privilege into discrete units that can be granted independently"),
    ("Windows Defender Application Control (WDAC)",
     "is a kernel-enforced application-control feature that restricts what code may execute on Windows"),
    ("Windows AppLocker",
     "is a user-mode application-control feature that restricts which executables, scripts, and installers may run"),
    ("Credential Guard",
     "uses Virtualization-Based Security to isolate LSASS secrets in a separate trustlet, blocking pass-the-hash"),
    ("Hypervisor-Protected Code Integrity (HVCI)",
     "uses Virtualization-Based Security to enforce kernel-mode code integrity, blocking unsigned drivers"),
    ("eBPF",
     "lets safe sandboxed programs run in the Linux kernel for observability, networking, and security enforcement"),
    ("Secure Boot",
     "verifies the digital signatures of bootloader and kernel images before execution to defeat boot-stage rootkits"),
    ("TPM (Trusted Platform Module)",
     "is a tamper-resistant cryptoprocessor that stores keys, performs measurements, and supports remote attestation"),
    ("UEFI",
     "is the firmware interface that succeeded BIOS, supporting Secure Boot, large disks, and a more capable runtime"),
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

    # Pattern 1 -- OWASP Top-10 (id <-> name <-> definition)
    owasp_names = [n for _, n, _ in OWASP_TOP10]
    owasp_defs = [d for _, _, d in OWASP_TOP10]
    owasp_ids = [i for i, _, _ in OWASP_TOP10]
    for oid, name, defn in OWASP_TOP10:
        for phr in [f"Which OWASP Top-10 2021 category is described as: {defn}?",
                    f"In the OWASP Top-10 2021, which entry covers: {defn}?",
                    f"Which OWASP 2021 risk corresponds to: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, owasp_names, name),
                f"OWASP {oid} {name} is the category defined as: {defn}.",
                shortname, instruction))
        for phr in [f"What is the OWASP {oid} category named?",
                    f"Which name does OWASP assign to the {oid} category?"]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, owasp_names, name),
                f"OWASP {oid} {name} is the category defined as: {defn}.",
                shortname, instruction))
        for phr in [f"Which OWASP Top-10 2021 identifier corresponds to '{name}'?"]:
            rows.append(make_mcq(
                rng, phr, oid, pick_distractors(rng, owasp_ids, oid),
                f"OWASP {oid} {name} is the category defined as: {defn}.",
                shortname, instruction))

    # Pattern 2 -- OWASP API Security Top-10 (similar shape)
    api_names = [n for _, n, _ in OWASP_API]
    api_defs = [d for _, _, d in OWASP_API]
    for aid, name, defn in OWASP_API:
        for phr in [f"Which OWASP API Security Top-10 2023 entry is described as: {defn}?",
                    f"In the OWASP API Security Top-10 2023, which item covers: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, api_names, name),
                f"OWASP API Security {aid} ({name}) is described as: {defn}.",
                shortname, instruction))
        for phr in [f"What is the OWASP API Security {aid} entry named?",
                    f"Which name does OWASP assign to the API {aid} risk?"]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, api_names, name),
                f"OWASP API Security {aid} ({name}) is described as: {defn}.",
                shortname, instruction))

    # Pattern 3 -- web vulnerability (name <-> mechanism <-> mitigation)
    vuln_names = [n for n, _, _ in WEB_VULNS]
    vuln_mechs = [m for _, _, m in WEB_VULNS]
    vuln_mits = [m for _, m, _ in WEB_VULNS]
    for name, mit, mech in WEB_VULNS:
        for phr in [f"Which web vulnerability class is described as: {mech}?",
                    f"Which class of web vulnerability does the following describe: {mech}?"]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, vuln_names, name),
                f"{name} is the vulnerability class where {mech}.",
                shortname, instruction))
        for phr in [f"What is the recommended primary mitigation for {name}?",
                    f"Which control is the standard primary mitigation for {name}?"]:
            rows.append(make_mcq(
                rng, phr, mit, pick_distractors(rng, vuln_mits, mit),
                f"The standard mitigation for {name} is to {mit}.",
                shortname, instruction))
        for phr in [f"By what mechanism does {name} occur?",
                    f"How does an attacker exploit {name}?"]:
            rows.append(make_mcq(
                rng, phr, mech, pick_distractors(rng, vuln_mechs, mech),
                f"{name} occurs when {mech}.",
                shortname, instruction))

    # Pattern 4 -- network protocol (name <-> default port <-> fact)
    proto_names = [n for n, _, _ in NETWORK_PROTOS]
    proto_ports = sorted({p for _, p, _ in NETWORK_PROTOS if p not in ("varies", "n/a")})
    proto_facts = [f for _, _, f in NETWORK_PROTOS]
    for name, port, fact in NETWORK_PROTOS:
        for phr in [f"What is true of the {name} protocol?",
                    f"Which statement about {name} is correct?"]:
            rows.append(make_mcq(
                rng, phr, fact, pick_distractors(rng, proto_facts, fact),
                f"{name} {fact}.", shortname, instruction))
        if port not in ("varies", "n/a"):
            for phr in [f"Which transport-protocol/port pair is the standard default for {name}?"]:
                rows.append(make_mcq(
                    rng, phr, port, pick_distractors(rng, proto_ports, port),
                    f"The standard default for {name} is {port}.", shortname, instruction))
        for phr in [f"Which protocol does the following describe: {fact}?"]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, proto_names, name),
                f"{name} {fact}.", shortname, instruction))

    # Pattern 5 -- OS / endpoint primitives
    os_names = [n for n, _ in OS_PRIMITIVES]
    os_defs = [d for _, d in OS_PRIMITIVES]
    for name, defn in OS_PRIMITIVES:
        for phr in [f"Which OS / endpoint security primitive is described as: {defn}?",
                    f"Which mechanism does the following describe: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, os_names, name),
                f"{name} {defn}.", shortname, instruction))
        for phr in [f"What does {name} do?",
                    f"How is {name} typically described?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, os_defs, defn),
                f"{name} {defn}.", shortname, instruction))

    return rows

