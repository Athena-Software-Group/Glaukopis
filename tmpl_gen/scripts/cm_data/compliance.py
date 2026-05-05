"""CM.COMPLIANCE.1 knowledge tables and generator (~2,000 rows).

Covers the major compliance/standards corpora referenced in v12 plan:
  - NIST Cybersecurity Framework 2.0 (Govern/Identify/Protect/Detect/
    Respond/Recover) and representative subcategories
  - ISO/IEC 27001:2022 (4 themes / 93 controls in 27002:2022)
  - HIPAA Security Rule (Administrative/Physical/Technical safeguards)
    and Privacy/Breach Notification rules
  - PCI-DSS v4.0 (12 requirements grouped into 6 control objectives)
  - NIST SP 800-53 Rev 5 control families
  - GDPR (principles, rights, lawful bases, breach notification,
    territorial scope)
  - SOC 2 Trust Services Criteria
"""

from __future__ import annotations

from typing import Callable

from . import pick_distractors

# (function, definition) -- NIST CSF 2.0
CSF_FUNCTIONS: list[tuple[str, str]] = [
    ("Govern (GV)",
     "establish, communicate and monitor the organisation's cybersecurity risk-management strategy, expectations and policy"),
    ("Identify (ID)",
     "develop the organisational understanding to manage cybersecurity risk to systems, assets, data and capabilities"),
    ("Protect (PR)",
     "develop and implement appropriate safeguards to ensure delivery of critical services"),
    ("Detect (DE)",
     "develop and implement appropriate activities to identify the occurrence of a cybersecurity event"),
    ("Respond (RS)",
     "develop and implement appropriate activities to take action regarding a detected cybersecurity incident"),
    ("Recover (RC)",
     "develop and implement appropriate activities to maintain plans for resilience and to restore capabilities or services impaired by a cybersecurity incident"),
]

# (subcategory_id, parent_function, brief)
CSF_SUBCATS: list[tuple[str, str, str]] = [
    ("GV.OC", "Govern (GV)", "Organizational Context"),
    ("GV.RM", "Govern (GV)", "Risk Management Strategy"),
    ("GV.RR", "Govern (GV)", "Roles, Responsibilities and Authorities"),
    ("GV.PO", "Govern (GV)", "Policy"),
    ("GV.OV", "Govern (GV)", "Oversight"),
    ("GV.SC", "Govern (GV)", "Cybersecurity Supply Chain Risk Management"),
    ("ID.AM", "Identify (ID)", "Asset Management"),
    ("ID.RA", "Identify (ID)", "Risk Assessment"),
    ("ID.IM", "Identify (ID)", "Improvement"),
    ("PR.AA", "Protect (PR)", "Identity Management, Authentication and Access Control"),
    ("PR.AT", "Protect (PR)", "Awareness and Training"),
    ("PR.DS", "Protect (PR)", "Data Security"),
    ("PR.PS", "Protect (PR)", "Platform Security"),
    ("PR.IR", "Protect (PR)", "Technology Infrastructure Resilience"),
    ("DE.CM", "Detect (DE)", "Continuous Monitoring"),
    ("DE.AE", "Detect (DE)", "Adverse Event Analysis"),
    ("RS.MA", "Respond (RS)", "Incident Management"),
    ("RS.AN", "Respond (RS)", "Incident Analysis"),
    ("RS.CO", "Respond (RS)", "Incident Response Reporting and Communication"),
    ("RS.MI", "Respond (RS)", "Incident Mitigation"),
    ("RC.RP", "Recover (RC)", "Incident Recovery Plan Execution"),
    ("RC.CO", "Recover (RC)", "Incident Recovery Communication"),
]

# ISO/IEC 27002:2022 -- 4 themes
ISO_THEMES: list[tuple[str, str, int]] = [
    ("Organizational controls", "policies, roles, supplier relationships and asset management at the organisation level", 37),
    ("People controls", "screening, terms and conditions of employment, training and awareness, disciplinary process", 8),
    ("Physical controls", "secure perimeters, equipment siting, secure disposal, clear desk policy", 14),
    ("Technological controls", "access control, cryptography, malware protection, logging, monitoring, network security", 34),
]

# (control_topic, theme)
ISO_CONTROLS: list[tuple[str, str]] = [
    ("Information security policies", "Organizational controls"),
    ("Roles and responsibilities", "Organizational controls"),
    ("Segregation of duties", "Organizational controls"),
    ("Information security in supplier relationships", "Organizational controls"),
    ("Information security incident management", "Organizational controls"),
    ("Threat intelligence", "Organizational controls"),
    ("Information security for use of cloud services", "Organizational controls"),
    ("ICT readiness for business continuity", "Organizational controls"),
    ("Screening (background checks)", "People controls"),
    ("Terms and conditions of employment", "People controls"),
    ("Information security awareness, education and training", "People controls"),
    ("Disciplinary process", "People controls"),
    ("Remote working", "People controls"),
    ("Physical security perimeters", "Physical controls"),
    ("Physical entry controls", "Physical controls"),
    ("Securing offices, rooms and facilities", "Physical controls"),
    ("Clear desk and clear screen policy", "Physical controls"),
    ("Equipment siting and protection", "Physical controls"),
    ("Secure disposal or re-use of equipment", "Physical controls"),
    ("Cabling security", "Physical controls"),
    ("Access control", "Technological controls"),
    ("Identity management", "Technological controls"),
    ("Authentication information management", "Technological controls"),
    ("Cryptography", "Technological controls"),
    ("Protection against malware", "Technological controls"),
    ("Management of technical vulnerabilities", "Technological controls"),
    ("Logging", "Technological controls"),
    ("Monitoring activities", "Technological controls"),
    ("Networks security", "Technological controls"),
    ("Web filtering", "Technological controls"),
    ("Secure development life cycle", "Technological controls"),
    ("Test data", "Technological controls"),
    ("Configuration management", "Technological controls"),
    ("Information deletion", "Technological controls"),
    ("Data masking", "Technological controls"),
    ("Data leakage prevention", "Technological controls"),
]

# HIPAA Security Rule safeguards
HIPAA_SAFEGUARDS: list[tuple[str, str, str]] = [
    ("Security Management Process", "Administrative", "risk analysis, risk management, sanction policy, information system activity review"),
    ("Assigned Security Responsibility", "Administrative", "designation of a security official responsible for the security policies"),
    ("Workforce Security", "Administrative", "authorization, supervision, workforce clearance and termination procedures"),
    ("Information Access Management", "Administrative", "isolating clearinghouse functions, access authorisation, access establishment and modification"),
    ("Security Awareness and Training", "Administrative", "security reminders, malicious software protection, log-in monitoring, password management training"),
    ("Security Incident Procedures", "Administrative", "response and reporting of security incidents"),
    ("Contingency Plan", "Administrative", "data backup, disaster recovery, emergency mode operation, testing/revision, applications/data criticality analysis"),
    ("Evaluation", "Administrative", "periodic technical and non-technical evaluation of compliance with the Security Rule"),
    ("Business Associate Contracts", "Administrative", "written contracts requiring satisfactory assurances by business associates"),
    ("Facility Access Controls", "Physical", "contingency operations, facility security plan, access control and validation, maintenance records"),
    ("Workstation Use", "Physical", "policies specifying functions, manner and physical attributes of workstations accessing ePHI"),
    ("Workstation Security", "Physical", "physical safeguards for all workstations to restrict access to authorised users"),
    ("Device and Media Controls", "Physical", "disposal, media re-use, accountability, data backup and storage of ePHI"),
    ("Access Control", "Technical", "unique user identification, emergency access procedure, automatic logoff, encryption and decryption"),
    ("Audit Controls", "Technical", "hardware, software and procedural mechanisms that record and examine activity in information systems containing ePHI"),
    ("Integrity Controls", "Technical", "mechanisms to authenticate ePHI and to ensure ePHI has not been altered or destroyed in an unauthorised manner"),
    ("Person or Entity Authentication", "Technical", "verification that the person or entity seeking access is the one claimed"),
    ("Transmission Security", "Technical", "integrity controls and encryption for ePHI being transmitted over a network"),
]

HIPAA_FACTS: list[tuple[str, str]] = [
    ("HIPAA Breach Notification Rule",
     "requires covered entities to notify affected individuals of a breach of unsecured PHI without unreasonable delay and in no case later than 60 days after discovery"),
    ("HIPAA Privacy Rule",
     "establishes national standards for the protection of individually identifiable health information held by covered entities"),
    ("HIPAA minimum necessary standard",
     "requires uses and disclosures of PHI be limited to the minimum necessary to accomplish the intended purpose"),
    ("HIPAA Security Rule",
     "establishes administrative, physical and technical safeguards for electronic protected health information (ePHI)"),
    ("HIPAA business associate",
     "any person or entity that performs functions or activities involving the use or disclosure of PHI on behalf of a covered entity"),
]


# PCI-DSS v4.0 -- 12 requirements
PCI_REQS: list[tuple[str, str]] = [
    ("Requirement 1", "install and maintain network security controls"),
    ("Requirement 2", "apply secure configurations to all system components"),
    ("Requirement 3", "protect stored account data"),
    ("Requirement 4", "protect cardholder data with strong cryptography during transmission over open, public networks"),
    ("Requirement 5", "protect all systems and networks from malicious software"),
    ("Requirement 6", "develop and maintain secure systems and software"),
    ("Requirement 7", "restrict access to system components and cardholder data by business need to know"),
    ("Requirement 8", "identify users and authenticate access to system components"),
    ("Requirement 9", "restrict physical access to cardholder data"),
    ("Requirement 10", "log and monitor all access to system components and cardholder data"),
    ("Requirement 11", "test security of systems and networks regularly"),
    ("Requirement 12", "support information security with organisational policies and programs"),
]

# NIST SP 800-53 Rev 5 control families
SP80053_FAMILIES: list[tuple[str, str, str]] = [
    ("AC", "Access Control", "logical access enforcement, account management, separation of duties"),
    ("AT", "Awareness and Training", "security awareness training and role-based training"),
    ("AU", "Audit and Accountability", "audit event logging, content of audit records, review and reporting"),
    ("CA", "Assessment, Authorization, and Monitoring", "control assessments, system authorization, continuous monitoring"),
    ("CM", "Configuration Management", "baseline configurations, change control, configuration settings"),
    ("CP", "Contingency Planning", "contingency plan, alternate sites, system backup, recovery and reconstitution"),
    ("IA", "Identification and Authentication", "user identification, device authentication, multifactor authentication"),
    ("IR", "Incident Response", "incident handling, monitoring, reporting and response training"),
    ("MA", "Maintenance", "controlled maintenance, maintenance tools, non-local maintenance"),
    ("MP", "Media Protection", "media access, marking, storage, transport and sanitization"),
    ("PE", "Physical and Environmental Protection", "physical access, monitoring, fire/water/power protection"),
    ("PL", "Planning", "system security plan, rules of behaviour, baseline tailoring"),
    ("PM", "Program Management", "information security program plan, senior agency information security officer"),
    ("PS", "Personnel Security", "position categorisation, personnel screening, termination and transfer"),
    ("PT", "PII Processing and Transparency", "privacy notice, consent, system of records notice"),
    ("RA", "Risk Assessment", "security categorisation, risk assessment, vulnerability scanning, threat hunting"),
    ("SA", "System and Services Acquisition", "allocation of resources, SDLC, supply-chain risk management, external system services"),
    ("SC", "System and Communications Protection", "boundary protection, transmission confidentiality, cryptographic key establishment"),
    ("SI", "System and Information Integrity", "flaw remediation, malicious code protection, security alerts and advisories"),
    ("SR", "Supply Chain Risk Management", "supply chain risk management plan, supplier reviews, provenance, component authenticity"),
]

# GDPR principles
GDPR_PRINCIPLES: list[tuple[str, str]] = [
    ("Lawfulness, fairness and transparency",
     "personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject"),
    ("Purpose limitation",
     "personal data shall be collected for specified, explicit and legitimate purposes and not further processed in a manner incompatible with those purposes"),
    ("Data minimisation",
     "personal data shall be adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed"),
    ("Accuracy",
     "personal data shall be accurate and, where necessary, kept up to date"),
    ("Storage limitation",
     "personal data shall be kept in a form which permits identification of data subjects for no longer than is necessary"),
    ("Integrity and confidentiality",
     "personal data shall be processed in a manner that ensures appropriate security of the personal data"),
    ("Accountability",
     "the controller shall be responsible for, and be able to demonstrate compliance with, the GDPR principles"),
]

# (right_name, gdpr_article_brief)
GDPR_RIGHTS: list[tuple[str, str]] = [
    ("Right of access (Article 15)", "the right to obtain confirmation of processing and a copy of personal data"),
    ("Right to rectification (Article 16)", "the right to have inaccurate personal data corrected without undue delay"),
    ("Right to erasure / right to be forgotten (Article 17)", "the right to obtain erasure of personal data without undue delay under specified grounds"),
    ("Right to restriction of processing (Article 18)", "the right to obtain a temporary restriction on processing in defined circumstances"),
    ("Right to data portability (Article 20)", "the right to receive personal data in a structured, commonly used, machine-readable format"),
    ("Right to object (Article 21)", "the right to object to processing based on legitimate interests, public task, direct marketing or research/statistics"),
    ("Rights related to automated decision-making (Article 22)", "the right not to be subject to a decision based solely on automated processing producing legal or similarly significant effects"),
]

GDPR_FACTS: list[tuple[str, str]] = [
    ("GDPR breach notification to supervisory authority",
     "must occur without undue delay and where feasible no later than 72 hours after the controller becomes aware of a personal data breach"),
    ("GDPR maximum administrative fine -- higher tier",
     "up to EUR 20 million or 4% of total worldwide annual turnover of the preceding financial year, whichever is higher"),
    ("GDPR maximum administrative fine -- lower tier",
     "up to EUR 10 million or 2% of total worldwide annual turnover of the preceding financial year, whichever is higher"),
    ("GDPR territorial scope (Article 3)",
     "applies to controllers/processors established in the EU and to those targeting or monitoring data subjects in the EU"),
    ("GDPR Data Protection Impact Assessment (Article 35)",
     "required where processing is likely to result in a high risk to the rights and freedoms of natural persons"),
    ("GDPR Data Protection Officer (Article 37)",
     "must be designated where the core activities involve large-scale systematic monitoring or large-scale processing of special-category data"),
]

GDPR_BASES: list[tuple[str, str]] = [
    ("Consent", "the data subject has given consent to processing for one or more specific purposes"),
    ("Contract", "processing is necessary for the performance of a contract with the data subject"),
    ("Legal obligation", "processing is necessary for compliance with a legal obligation to which the controller is subject"),
    ("Vital interests", "processing is necessary to protect the vital interests of the data subject or another natural person"),
    ("Public task", "processing is necessary for the performance of a task carried out in the public interest or in the exercise of official authority"),
    ("Legitimate interests", "processing is necessary for the legitimate interests of the controller or a third party, except where overridden by the rights of the data subject"),
]

# SOC 2 Trust Services Criteria
SOC2_TSC: list[tuple[str, str]] = [
    ("Security (Common Criteria)", "the system is protected against unauthorised access, both physical and logical"),
    ("Availability", "the system is available for operation and use as committed or agreed"),
    ("Processing Integrity", "system processing is complete, valid, accurate, timely and authorised"),
    ("Confidentiality", "information designated as confidential is protected as committed or agreed"),
    ("Privacy", "personal information is collected, used, retained, disclosed and disposed of in conformity with the entity's privacy notice"),
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

    # Pattern 1 -- CSF function definitions
    func_names = [n for n, _ in CSF_FUNCTIONS]
    func_defs = [d for _, d in CSF_FUNCTIONS]
    for name, defn in CSF_FUNCTIONS:
        for phr in [
            f"Which NIST CSF 2.0 function is responsible for the following: {defn}?",
            f"In NIST CSF 2.0, which function does this describe: {defn}?",
            f"Which Core function in NIST CSF 2.0 covers: {defn}?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, func_names, name),
                f"In NIST CSF 2.0, the {name} function is the one defined as: {defn}.",
                shortname, instruction))
        for phr in [
            f"What is the purpose of the {name} function in NIST CSF 2.0?",
            f"How is the {name} function defined in NIST CSF 2.0?",
            f"Which statement best characterises the {name} function?",
        ]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, func_defs, defn),
                f"NIST CSF 2.0 defines {name} as: {defn}.",
                shortname, instruction))

    # Pattern 2 -- CSF subcategory parent function
    for sub_id, parent, brief in CSF_SUBCATS:
        for phr in [
            f"To which NIST CSF 2.0 function does the category {sub_id} ({brief}) belong?",
            f"Under which Core function is {sub_id} -- {brief} grouped?",
            f"The CSF 2.0 category {sub_id} ({brief}) sits under which function?",
        ]:
            rows.append(make_mcq(
                rng, phr, parent, pick_distractors(rng, func_names, parent),
                f"In NIST CSF 2.0, category {sub_id} ({brief}) belongs to the {parent} function.",
                shortname, instruction))

    # Pattern 3 -- ISO 27002:2022 control -> theme
    theme_names = [t for t, _, _ in ISO_THEMES]
    for control, theme in ISO_CONTROLS:
        for phr in [
            f"In ISO/IEC 27002:2022, the control '{control}' belongs to which theme?",
            f"Under which ISO/IEC 27002:2022 theme is '{control}' classified?",
            f"Which of the four ISO/IEC 27002:2022 themes contains '{control}'?",
        ]:
            rows.append(make_mcq(
                rng, phr, theme, pick_distractors(rng, theme_names, theme),
                f"ISO/IEC 27002:2022 places '{control}' under the {theme} theme.",
                shortname, instruction))

    # Pattern 4 -- HIPAA safeguard category
    safe_cats = ["Administrative", "Physical", "Technical"]
    safe_briefs = [b for _, _, b in HIPAA_SAFEGUARDS]
    for std, cat, brief in HIPAA_SAFEGUARDS:
        ans = f"{cat} safeguards"
        for phr in [
            f"Under the HIPAA Security Rule, the standard '{std}' is part of which safeguard category?",
            f"Which HIPAA Security Rule safeguard category contains the '{std}' standard?",
            f"The HIPAA Security Rule classifies '{std}' under which safeguard type?",
        ]:
            rows.append(make_mcq(
                rng, phr, ans, [f"{c} safeguards" for c in safe_cats if c != cat] +
                ["Breach Notification Rule"],
                f"The HIPAA Security Rule lists '{std}' as a {cat} safeguard.",
                shortname, instruction))
        for phr in [
            f"What does the HIPAA Security Rule '{std}' standard require?",
            f"Which statement best summarises the HIPAA '{std}' standard?",
        ]:
            rows.append(make_mcq(
                rng, phr, brief, pick_distractors(rng, safe_briefs, brief),
                f"The HIPAA Security Rule '{std}' standard covers {brief}.",
                shortname, instruction))

    # Pattern 5 -- HIPAA general facts
    hipaa_fact_names = [n for n, _ in HIPAA_FACTS]
    hipaa_fact_defs = [d for _, d in HIPAA_FACTS]
    for name, defn in HIPAA_FACTS:
        for phr in [
            f"What does the {name} require?",
            f"Which statement most accurately describes {name}?",
        ]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, hipaa_fact_defs, defn),
                f"The {name} {defn}.", shortname, instruction))
        for phr in [
            f"Which HIPAA provision is described by: {defn}?",
            f"Which rule corresponds to: {defn}?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, hipaa_fact_names, name),
                f"This is the {name}: {defn}.",
                shortname, instruction))

    # Pattern 6 -- PCI-DSS requirements
    pci_names = [n for n, _ in PCI_REQS]
    pci_briefs = [b for _, b in PCI_REQS]
    for req, brief in PCI_REQS:
        for phr in [
            f"In PCI-DSS v4.0, which requirement covers: {brief}?",
            f"Which PCI-DSS v4.0 requirement is defined as: {brief}?",
            f"Which numbered PCI-DSS v4.0 requirement is concerned with: {brief}?",
        ]:
            rows.append(make_mcq(
                rng, phr, req, pick_distractors(rng, pci_names, req),
                f"PCI-DSS v4.0 {req} is the requirement to {brief}.",
                shortname, instruction))
        for phr in [
            f"What is PCI-DSS v4.0 {req} about?",
            f"Which statement best describes PCI-DSS v4.0 {req}?",
        ]:
            rows.append(make_mcq(
                rng, phr, brief, pick_distractors(rng, pci_briefs, brief),
                f"PCI-DSS v4.0 {req} is the requirement to {brief}.",
                shortname, instruction))

    # Pattern 7 -- NIST SP 800-53 family classification
    sp_codes = [c for c, _, _ in SP80053_FAMILIES]
    sp_names = [n for _, n, _ in SP80053_FAMILIES]
    sp_briefs = [b for _, _, b in SP80053_FAMILIES]
    for code, fam_name, brief in SP80053_FAMILIES:
        for phr in [
            f"In NIST SP 800-53 Rev 5, what is the two-letter family code for '{fam_name}'?",
            f"Which NIST SP 800-53 Rev 5 family identifier abbreviates '{fam_name}'?",
        ]:
            rows.append(make_mcq(
                rng, phr, code, pick_distractors(rng, sp_codes, code),
                f"In NIST SP 800-53 Rev 5, '{fam_name}' is family code {code}.",
                shortname, instruction))
        for phr in [
            f"Which NIST SP 800-53 Rev 5 control family does the code '{code}' designate?",
            f"What is the full name of NIST SP 800-53 Rev 5 control family '{code}'?",
        ]:
            rows.append(make_mcq(
                rng, phr, fam_name, pick_distractors(rng, sp_names, fam_name),
                f"NIST SP 800-53 Rev 5 family code {code} is the {fam_name} family.",
                shortname, instruction))
        for phr in [
            f"Which control family in NIST SP 800-53 Rev 5 covers: {brief}?",
            f"Which 800-53 family is responsible for: {brief}?",
        ]:
            rows.append(make_mcq(
                rng, phr, fam_name, pick_distractors(rng, sp_names, fam_name),
                f"The NIST SP 800-53 Rev 5 {fam_name} family ({code}) covers {brief}.",
                shortname, instruction))

    # Pattern 8 -- GDPR principles
    gdpr_principle_names = [n for n, _ in GDPR_PRINCIPLES]
    gdpr_principle_defs = [d for _, d in GDPR_PRINCIPLES]
    for name, defn in GDPR_PRINCIPLES:
        for phr in [
            f"Which GDPR principle requires that {defn}?",
            f"The requirement '{defn}' corresponds to which GDPR principle?",
            f"Article 5 of the GDPR enumerates several principles; which one states: {defn}?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, gdpr_principle_names, name),
                f"The GDPR Article 5 principle of {name} requires that {defn}.",
                shortname, instruction))
        for phr in [
            f"What does the GDPR principle of {name} require?",
            f"How is the GDPR principle of {name} formulated?",
        ]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, gdpr_principle_defs, defn),
                f"The GDPR principle of {name} requires that {defn}.",
                shortname, instruction))

    # Pattern 9 -- GDPR rights
    gdpr_right_names = [n for n, _ in GDPR_RIGHTS]
    gdpr_right_defs = [d for _, d in GDPR_RIGHTS]
    for name, defn in GDPR_RIGHTS:
        for phr in [
            f"Which GDPR data subject right is described as: {defn}?",
            f"Which right under the GDPR provides: {defn}?",
            f"The right described as '{defn}' is which GDPR data subject right?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, gdpr_right_names, name),
                f"The {name} grants {defn}.",
                shortname, instruction))
        for phr in [
            f"What does the {name} entitle a data subject to?",
            f"How is the {name} defined under the GDPR?",
        ]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, gdpr_right_defs, defn),
                f"Under GDPR, the {name} grants {defn}.",
                shortname, instruction))

    # Pattern 10 -- GDPR lawful bases
    gdpr_basis_names = [n for n, _ in GDPR_BASES]
    gdpr_basis_defs = [d for _, d in GDPR_BASES]
    for name, defn in GDPR_BASES:
        for phr in [
            f"Which GDPR Article 6 lawful basis applies when: {defn}?",
            f"Which lawful basis under the GDPR fits the case where {defn}?",
            f"Article 6(1) lists six lawful bases; which one covers: {defn}?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, gdpr_basis_names, name),
                f"The GDPR Article 6 lawful basis of {name} applies when {defn}.",
                shortname, instruction))
        for phr in [
            f"What is the GDPR lawful basis of '{name}'?",
            f"When does the GDPR lawful basis '{name}' apply?",
        ]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, gdpr_basis_defs, defn),
                f"The GDPR lawful basis of {name} applies when {defn}.",
                shortname, instruction))

    # Pattern 11 -- GDPR general facts
    gdpr_fact_names = [n for n, _ in GDPR_FACTS]
    gdpr_fact_defs = [d for _, d in GDPR_FACTS]
    for name, defn in GDPR_FACTS:
        for phr in [
            f"What is the rule for {name}?",
            f"Which statement most accurately describes {name}?",
            f"How is {name} defined in the GDPR?",
        ]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, gdpr_fact_defs, defn),
                f"{name} {defn}.", shortname, instruction))
        for phr in [
            f"Which GDPR provision is described by: {defn}?",
            f"Which rule corresponds to: {defn}?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, gdpr_fact_names, name),
                f"This is {name}: {defn}.",
                shortname, instruction))

    # Pattern 12 -- SOC 2 Trust Services Criteria
    soc2_names = [n for n, _ in SOC2_TSC]
    soc2_defs = [d for _, d in SOC2_TSC]
    for name, defn in SOC2_TSC:
        for phr in [
            f"Which SOC 2 Trust Services Criterion requires that {defn}?",
            f"The criterion '{defn}' corresponds to which SOC 2 TSC?",
            f"Which of the five SOC 2 Trust Services Criteria covers: {defn}?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, soc2_names, name),
                f"The SOC 2 {name} criterion requires that {defn}.",
                shortname, instruction))
        for phr in [
            f"What does the SOC 2 {name} criterion require?",
            f"How is the SOC 2 {name} criterion defined?",
        ]:
            rows.append(make_mcq(
                rng, phr, defn, pick_distractors(rng, soc2_defs, defn),
                f"The SOC 2 {name} criterion requires that {defn}.",
                shortname, instruction))
    return rows
