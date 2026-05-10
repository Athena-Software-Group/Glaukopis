"""CTI / cybersecurity terminology MCQ knowledge tables sourced from public
glossaries: NIST SP 800-150 (Guide to Cyber Threat Information Sharing),
NIST SP 800-61 Rev 2 (Computer Security Incident Handling Guide), MITRE
ATT&CK glossary (attack.mitre.org/resources/), CISA Stop-Ransomware
glossary, ENISA Threat Landscape glossary, and ISO/IEC 27000:2018
information-security vocabulary. Definitions are paraphrased to match
the source intent; the source attribution is preserved so the generated
MCQ rationale can cite the standard.

Coverage is complementary to AB.MCQ.* (MITRE ATT&CK Cypher templates),
AB.MCQ.EXT.MITRE.1 (MITRE relationships not traversed by templates),
and AB.MCQ.EXT.SEC.1 (OWASP / network / OS security): this family
covers terminology -- the canonical names and definitions a CTI analyst
must agree on with peers, regulators, and standards bodies.
"""

from __future__ import annotations

from collections.abc import Callable

from . import pick_distractors

# (term, source-tag, definition).  source-tag identifies the standard the
# definition is paraphrased from so the rationale can attribute it.
GLOSSARY: list[tuple[str, str, str]] = [
    # NIST SP 800-150 (cyber threat information sharing)
    ("Threat Intelligence", "NIST SP 800-150",
     "threat information that has been aggregated, transformed, analyzed, interpreted, or enriched to provide the necessary context for decision-making"),
    ("Threat Information", "NIST SP 800-150",
     "any information related to a threat that might help an organization protect itself against the threat or detect the activities of an actor"),
    ("Indicator of Compromise", "NIST SP 800-150",
     "a technical artifact or observable that suggests an attack is imminent, is currently underway, or that a compromise may have already occurred"),
    ("Tactics, Techniques, and Procedures", "NIST SP 800-150",
     "the behavior of an actor: tactics describe high-level objectives, techniques describe how those goals are achieved, and procedures describe specific implementations of techniques"),
    ("Information Sharing", "NIST SP 800-150",
     "the exchange of cyber threat information between two or more organizations to support shared situational awareness and coordinated defense"),
    ("Sharing Community", "NIST SP 800-150",
     "a group of organizations that have agreed to share cyber threat information among themselves under a common set of rules"),
    ("Trusted Third Party", "NIST SP 800-150",
     "an entity that facilitates information exchange between organizations and is trusted by all participants to handle the shared content appropriately"),
    ("Traffic Light Protocol", "NIST SP 800-150",
     "a four-color marking schema (RED, AMBER, GREEN, WHITE) that indicates the audience to which a piece of shared information may be redistributed"),

    # NIST SP 800-61 Rev 2 (incident handling)
    ("Event", "NIST SP 800-61 Rev 2",
     "any observable occurrence in a network or system"),
    ("Adverse Event", "NIST SP 800-61 Rev 2",
     "an event with a negative consequence, such as a system crash, network packet flood, or unauthorized use of system privileges"),
    ("Computer Security Incident", "NIST SP 800-61 Rev 2",
     "a violation or imminent threat of violation of computer security policies, acceptable use policies, or standard security practices"),
    ("Incident Handling", "NIST SP 800-61 Rev 2",
     "the mitigation of violations of security policies and recommended practices, including the four phases preparation, detection and analysis, containment/eradication/recovery, and post-incident activity"),
    ("Incident Response", "NIST SP 800-61 Rev 2",
     "the activities a CSIRT performs to detect, respond to, and recover from a security incident"),
    ("Precursor", "NIST SP 800-61 Rev 2",
     "a sign that an incident may occur in the future"),
    ("Indicator", "NIST SP 800-61 Rev 2",
     "a sign that an incident may have occurred or may be occurring now"),
    ("Containment", "NIST SP 800-61 Rev 2",
     "the activity of limiting the scope of an incident before it overwhelms resources or increases damage, prior to eradication and recovery"),
    ("Eradication", "NIST SP 800-61 Rev 2",
     "the activity of eliminating the components of an incident from the environment, such as deleting malware and disabling breached accounts"),
    ("Recovery", "NIST SP 800-61 Rev 2",
     "the activity of restoring affected systems to normal operation, confirming they are functioning correctly, and remediating exploited vulnerabilities"),
    ("Lessons Learned", "NIST SP 800-61 Rev 2",
     "post-incident activity in which the organization reviews the incident handling process to identify needed improvements"),

    # MITRE ATT&CK glossary
    ("MITRE ATT&CK", "MITRE ATT&CK glossary",
     "a globally accessible knowledge base of adversary tactics and techniques based on real-world observations, used as a foundation for the development of specific threat models and methodologies"),
    ("Tactic", "MITRE ATT&CK glossary",
     "the adversary's tactical goal: the reason for performing an action, such as initial access, execution, or exfiltration"),
    ("Technique", "MITRE ATT&CK glossary",
     "how an adversary achieves a tactical goal by performing an action"),
    ("Sub-technique", "MITRE ATT&CK glossary",
     "a more specific or lower-level description of adversarial behavior than a parent technique, sharing the same tactical goal"),
    ("Procedure", "MITRE ATT&CK glossary",
     "a specific in-the-wild implementation that an adversary uses for a technique or sub-technique"),
    ("Group", "MITRE ATT&CK glossary",
     "a set of related intrusion activity that is tracked by a common name in the security community"),
    ("Software", "MITRE ATT&CK glossary",
     "a custom or commercial code, operating system utility, open-source tool, or other tool used to conduct adversary behavior"),
    ("Mitigation", "MITRE ATT&CK glossary",
     "a security concept or class of technologies that can prevent a technique from being successfully executed"),
    ("Data Source", "MITRE ATT&CK glossary",
     "a subject or topic of information that can be collected by sensors or logs and that may be used to identify or detect adversary actions"),
    ("Data Component", "MITRE ATT&CK glossary",
     "a specific property or value of a data source that can be observed or collected to detect adversary behavior"),
    ("Detection", "MITRE ATT&CK glossary",
     "high-level analytic process, sensor, or data which can be used to identify a technique that has been used by an adversary"),

    # CISA stop-ransomware glossary + general CISA usage
    ("Ransomware", "CISA glossary",
     "a type of malicious software that encrypts files on a victim system or denies access to data and demands a payment in exchange for restoring access"),
    ("Double Extortion", "CISA glossary",
     "a ransomware tactic in which the actor both encrypts the victim's data and threatens to leak exfiltrated data unless an additional payment is made"),
    ("Initial Access Broker", "CISA glossary",
     "a threat actor that gains unauthorized access to organizations and sells that access to other criminal actors who then conduct follow-on attacks"),
    ("Living Off the Land", "CISA glossary",
     "the use of legitimate, pre-installed system tools by an adversary to perform malicious actions while evading detection"),
    ("Supply Chain Attack", "CISA glossary",
     "an attack that compromises a trusted third-party software, hardware, or service in order to reach the third party's downstream customers"),
    ("Phishing", "CISA glossary",
     "the fraudulent practice of sending communications that appear to come from reputable sources in order to induce victims to reveal sensitive information or run malicious code"),
    ("Spear Phishing", "CISA glossary",
     "a targeted phishing attack tailored to a specific individual or organization using research about the target to increase credibility"),
    ("Business Email Compromise", "CISA glossary",
     "a scam in which an attacker compromises or impersonates a legitimate business email account to defraud the organization or its partners"),

    # ENISA Threat Landscape glossary
    ("Threat Actor", "ENISA glossary",
     "an entity that is partially or wholly responsible for an incident or that has the intent to cause an incident"),
    ("Threat Vector", "ENISA glossary",
     "the path or means by which an attacker can gain access to a target asset to deliver a malicious payload or outcome"),
    ("Attack Surface", "ENISA glossary",
     "the sum of the different points where an unauthorized user can try to enter or extract data from an environment"),
    ("Zero-Day", "ENISA glossary",
     "a vulnerability that is unknown to or unaddressed by parties responsible for patching it, and against which no public mitigation yet exists"),
    ("Hacktivism", "ENISA glossary",
     "the use of cyber attacks to promote a political or social agenda"),
    ("Disinformation", "ENISA glossary",
     "verifiably false or misleading information that is created, presented, and disseminated for economic gain or to intentionally deceive the public"),
    ("Distributed Denial-of-Service", "ENISA glossary",
     "an attack in which multiple compromised systems are used to flood a target with traffic so that legitimate users cannot access it"),
    ("Botnet", "ENISA glossary",
     "a network of compromised devices controlled by an attacker that can be used collectively to carry out malicious actions such as DDoS or spam distribution"),
    ("Cryptojacking", "ENISA glossary",
     "the unauthorized use of a computing resource to mine cryptocurrency for the attacker's benefit"),

    # ISO/IEC 27000:2018 information-security vocabulary
    ("Asset", "ISO/IEC 27000:2018",
     "anything that has value to the organization, including information, software, hardware, services, people, and intangibles such as reputation"),
    ("Vulnerability", "ISO/IEC 27000:2018",
     "a weakness of an asset or control that can be exploited by one or more threats"),
    ("Threat", "ISO/IEC 27000:2018",
     "a potential cause of an unwanted incident, which can result in harm to a system or organization"),
    ("Risk", "ISO/IEC 27000:2018",
     "the effect of uncertainty on objectives, expressed in terms of a combination of the consequences of an event and the associated likelihood of occurrence"),
    ("Control", "ISO/IEC 27000:2018",
     "a measure that maintains and/or modifies risk, including any process, policy, device, practice, or other action"),
    ("Confidentiality", "ISO/IEC 27000:2018",
     "the property that information is not made available or disclosed to unauthorized individuals, entities, or processes"),
    ("Integrity", "ISO/IEC 27000:2018",
     "the property of accuracy and completeness of information"),
    ("Availability", "ISO/IEC 27000:2018",
     "the property of being accessible and usable upon demand by an authorized entity"),
    ("Non-repudiation", "ISO/IEC 27000:2018",
     "the ability to prove the occurrence of a claimed event or action and the originating entities, in order to resolve disputes about whether the event or action took place"),
    ("Information Security Management System", "ISO/IEC 27000:2018",
     "the part of the overall management system, based on a business-risk approach, used to establish, implement, operate, monitor, review, maintain, and improve information security"),

    # CTI tradecraft and detection-engineering canon (general accepted usage)
    ("STIX", "OASIS Cyber Threat Intelligence",
     "a structured language for representing cyber threat information so that it can be shared, stored, and analyzed in a consistent manner"),
    ("TAXII", "OASIS Cyber Threat Intelligence",
     "an application protocol for exchanging cyber threat information over HTTPS, designed to support STIX content"),
    ("YARA", "Detection-engineering canon",
     "a pattern-matching language and tool used by malware analysts to identify and classify malware samples based on textual or binary patterns"),
    ("Sigma", "Detection-engineering canon",
     "a generic, vendor-agnostic signature format that allows detection engineers to express SIEM detection logic in YAML which can then be converted to many specific query languages"),
    ("Snort", "Detection-engineering canon",
     "an open-source network intrusion detection and prevention system that uses a rule language to inspect packets and detect malicious traffic"),
    ("Suricata", "Detection-engineering canon",
     "an open-source network threat detection engine that supports IDS, IPS, network security monitoring, and offline PCAP processing"),
    ("Diamond Model", "CTI tradecraft canon",
     "an analytic framework that describes intrusion activity in terms of four core features: adversary, capability, infrastructure, and victim, connected by socio-political and technical meta-features"),
    ("Cyber Kill Chain", "Lockheed Martin",
     "an intrusion-modeling framework that decomposes an attack into seven phases: reconnaissance, weaponization, delivery, exploitation, installation, command and control, and actions on objectives"),
    ("Pyramid of Pain", "David Bianco / detection-engineering canon",
     "a model that ranks indicators by how much pain they cause an adversary when denied, from trivial (hash values) to maximal (TTPs)"),
    ("False Positive", "Detection-engineering canon",
     "a detection result that flags benign or expected activity as malicious"),
    ("True Positive", "Detection-engineering canon",
     "a detection result that correctly identifies actual malicious activity"),
    ("Dwell Time", "CTI tradecraft canon",
     "the length of time a threat actor remains undetected within a victim environment from the moment of initial compromise to detection"),
    ("Threat Hunting", "CTI tradecraft canon",
     "the proactive search through networks and datasets to detect and isolate advanced threats that evade existing automated detection"),
    ("Attribution", "CTI tradecraft canon",
     "the process of determining the threat actor responsible for an observed intrusion based on the totality of available technical, operational, and strategic evidence"),
    ("Red Team", "CTI tradecraft canon",
     "an authorized group that emulates the tactics, techniques, and procedures of real-world adversaries to test an organization's defenses"),
    ("Blue Team", "CTI tradecraft canon",
     "the defensive group within an organization responsible for maintaining the security posture and responding to attacks, often exercised against a red team"),
    ("Purple Team", "CTI tradecraft canon",
     "a structured collaboration between red and blue teams to exchange feedback during exercises so that defensive controls and detections can be improved iteratively"),

    # NIST CSF 2.0 functional pillars
    ("Govern", "NIST Cybersecurity Framework 2.0",
     "the cybersecurity governance function: establishing, communicating, and monitoring the organization's cybersecurity risk management strategy, expectations, and policy"),
    ("Identify", "NIST Cybersecurity Framework 2.0",
     "the function of developing the organizational understanding needed to manage cybersecurity risks to systems, assets, data, and capabilities"),
    ("Protect", "NIST Cybersecurity Framework 2.0",
     "the function of developing and implementing the safeguards needed to ensure delivery of critical services"),
    ("Detect", "NIST Cybersecurity Framework 2.0",
     "the function of developing and implementing the activities needed to identify the occurrence of a cybersecurity event in a timely manner"),
    ("Respond", "NIST Cybersecurity Framework 2.0",
     "the function of developing and implementing the activities needed to take action regarding a detected cybersecurity incident"),
    ("Recover", "NIST Cybersecurity Framework 2.0",
     "the function of developing and implementing the activities needed to maintain plans for resilience and to restore capabilities or services impaired by a cybersecurity incident"),
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

    terms = [t for t, _, _ in GLOSSARY]
    defns = [d for _, _, d in GLOSSARY]

    for term, src, defn in GLOSSARY:
        # Pattern 1 -- definition -> term (the most common eval shape)
        for phr in [f"Which cybersecurity term is defined as: {defn}?",
                    f"In {src}, which term is defined as: {defn}?"]:
            rows.append(make_mcq(
                rng, phr, term, pick_distractors(rng, terms, term),
                f"{src} defines {term} as {defn}.",
                shortname, instruction))

        # Pattern 2 -- term -> definition (the inverse)
        for phr in [f"How does {src} define {term}?",
                    f"Which definition matches the term '{term}' as used in {src}?"]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, defns, defn),
                f"{src} defines {term} as {defn}.",
                shortname, instruction))

        # Pattern 3 -- term -> source attribution (lighter, every other term)
        if len(rows) % 2 == 0:
            sources = sorted({s for _, s, _ in GLOSSARY})
            for phr in [f"Which standard or canon defines '{term}' for cybersecurity practitioners?"]:
                rows.append(make_mcq(
                    rng, phr, src, pick_distractors(rng, sources, src),
                    f"{term} is defined by {src} as {defn}.",
                    shortname, instruction))

    return rows
