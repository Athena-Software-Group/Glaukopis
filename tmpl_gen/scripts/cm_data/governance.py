"""CM.GOV.1 knowledge tables and generator (~1,000 rows).

Covers governance and risk management topics: quantitative risk
formulas (ALE = ARO * SLE; SLE = AV * EF), risk treatment options,
governance frameworks (COBIT, ITIL, COSO ERM, ISO 31000, NIST RMF,
FAIR), policy hierarchy, security roles, three lines of defence,
business continuity (BIA/RTO/RPO/MTD), incident response phases
(NIST SP 800-61), and change/vulnerability management.
"""

from __future__ import annotations

from typing import Callable

from . import pick_distractors

# (term, formula_or_definition)
RISK_FORMULAS: list[tuple[str, str]] = [
    ("Annualized Loss Expectancy (ALE)",
     "ALE = ARO * SLE; the expected monetary loss from a risk over one year"),
    ("Single Loss Expectancy (SLE)",
     "SLE = AV * EF; the expected monetary loss from a single occurrence of a risk"),
    ("Asset Value (AV)",
     "the monetary value assigned to an asset for use in quantitative risk analysis"),
    ("Exposure Factor (EF)",
     "the percentage of an asset's value lost if a specific threat materialises"),
    ("Annualized Rate of Occurrence (ARO)",
     "the expected number of times a specific threat will occur in one year"),
    ("Total Cost of Ownership (TCO)",
     "the cumulative cost of acquiring, deploying, operating and disposing of an asset over its life"),
    ("Return on Security Investment (ROSI)",
     "ROSI = (ALE_before - ALE_after - control_cost) / control_cost"),
    ("Residual risk",
     "the remaining risk after controls have been applied; never zero in practice"),
    ("Inherent risk",
     "the level of risk before any controls or mitigations have been applied"),
    ("Risk appetite",
     "the amount and type of risk an organisation is willing to take in pursuit of its objectives"),
    ("Risk tolerance",
     "the acceptable variation in outcomes related to specific objectives within the risk appetite"),
]

# (option, definition)
RISK_TREATMENT: list[tuple[str, str]] = [
    ("Avoid", "eliminate the risk by not engaging in the risk-bearing activity"),
    ("Mitigate (Reduce)", "implement controls to lower the likelihood or impact of the risk"),
    ("Transfer (Share)", "shift the risk to a third party, typically via insurance or outsourcing"),
    ("Accept (Retain)", "acknowledge the risk and accept the potential loss without further action"),
]

# (framework, sentence)
FRAMEWORKS: list[tuple[str, str]] = [
    ("COBIT 2019",
     "an ISACA enterprise governance and management of IT framework with 40 governance/management objectives"),
    ("ITIL 4",
     "the AXELOS service-management framework built around the Service Value System and 34 management practices"),
    ("COSO ERM (2017)",
     "a five-component, twenty-principle enterprise risk-management framework integrating strategy and performance"),
    ("ISO/IEC 31000:2018",
     "an ISO standard providing principles, framework and process for managing risk applicable to any organisation"),
    ("NIST Risk Management Framework (RMF)",
     "the NIST SP 800-37 seven-step framework: Prepare, Categorise, Select, Implement, Assess, Authorise, Monitor"),
    ("FAIR (Factor Analysis of Information Risk)",
     "a quantitative risk-analysis methodology that decomposes risk into loss-event frequency and loss magnitude"),
    ("OCTAVE Allegro",
     "an information security risk-assessment methodology focused on information assets and their containers"),
    ("Six Sigma",
     "a process-improvement methodology using DMAIC or DMADV to reduce defects and variation"),
    ("CMMI",
     "the Capability Maturity Model Integration; a process-improvement maturity model with five levels"),
    ("Zachman Framework",
     "an enterprise architecture ontology classifying artefacts across six perspectives and six aspects"),
]

# (rmf_step_name, rmf_step_description)
NIST_RMF_STEPS: list[tuple[str, str]] = [
    ("Prepare",
     "essential activities to prepare the organisation to manage security and privacy risks"),
    ("Categorise",
     "categorise the system and the information processed, stored and transmitted by the system"),
    ("Select",
     "select an initial set of controls for the system and tailor as needed to reduce risk"),
    ("Implement",
     "implement the controls and document how they are deployed"),
    ("Assess",
     "assess the controls to determine if they are implemented correctly and producing the desired outcome"),
    ("Authorise",
     "authorise system operation based on a determination that the risk is acceptable"),
    ("Monitor",
     "monitor the system and the associated controls on an ongoing basis"),
]

# Policy hierarchy
POLICY_HIERARCHY: list[tuple[str, str]] = [
    ("Policy",
     "a high-level statement of management intent that is mandatory and rarely changes"),
    ("Standard",
     "a mandatory specification of how policies will be implemented; specifies the 'what'"),
    ("Procedure",
     "a step-by-step set of mandatory instructions describing how to perform a task"),
    ("Guideline",
     "recommended, non-mandatory advice that supplements standards and procedures"),
    ("Baseline",
     "a minimum security configuration consistently applied to systems of a given type"),
]

# Security roles
ROLES: list[tuple[str, str]] = [
    ("Chief Information Security Officer (CISO)",
     "the executive responsible for the organisation's information and data security strategy and program"),
    ("Chief Information Officer (CIO)",
     "the executive responsible for the organisation's information technology strategy and operations"),
    ("Chief Privacy Officer (CPO)",
     "the executive responsible for managing risks and policies related to personal data privacy"),
    ("Data Protection Officer (DPO)",
     "the role required by GDPR Article 37 to monitor compliance with data-protection law"),
    ("Data Owner",
     "the senior business representative accountable for the classification and protection of a data set"),
    ("Data Custodian",
     "the role responsible for the technical handling, storage and operational protection of data"),
    ("Data Steward",
     "the role responsible for the day-to-day quality and usability of a data set"),
    ("System Owner",
     "the individual responsible for the overall operation, maintenance and security of an information system"),
    ("Information Security Manager",
     "the manager responsible for the day-to-day operation of the information-security program"),
    ("Internal Auditor",
     "the role providing independent assurance that risk-management, governance and control processes are effective"),
]

# (line_of_defence, what_it_does)
LINES_OF_DEFENCE: list[tuple[str, str]] = [
    ("First line of defence", "operational management owns and manages risk in the day-to-day business"),
    ("Second line of defence", "risk-management and compliance functions oversee and challenge the first line"),
    ("Third line of defence", "internal audit provides independent assurance to the board and senior management"),
]



# Business continuity / disaster recovery
BCP_TERMS: list[tuple[str, str]] = [
    ("Business Impact Analysis (BIA)",
     "the process that identifies critical business functions and the impact of their disruption over time"),
    ("Recovery Time Objective (RTO)",
     "the maximum tolerable time within which a business function must be restored after a disruption"),
    ("Recovery Point Objective (RPO)",
     "the maximum acceptable amount of data loss measured in time before a disruption"),
    ("Maximum Tolerable Downtime (MTD)",
     "the longest period a business function can be unavailable before threatening organisational viability"),
    ("Work Recovery Time (WRT)",
     "the time required to verify and restore business operations after technology recovery is complete"),
    ("Mean Time Between Failures (MTBF)",
     "the predicted elapsed time between inherent failures of a system during normal operation"),
    ("Mean Time To Repair (MTTR)",
     "the average time required to repair a failed component or system and return it to service"),
    ("Hot site",
     "a fully equipped alternate facility ready for immediate operation with current data replication"),
    ("Warm site",
     "a partially equipped alternate facility that can be made operational within hours to a day"),
    ("Cold site",
     "a basic alternate facility with infrastructure but no equipment or data, requiring days to bring online"),
]

# NIST SP 800-61 incident response phases
IR_PHASES: list[tuple[str, str]] = [
    ("Preparation",
     "establish capability, training, tools and policies to handle incidents before they occur"),
    ("Detection and Analysis",
     "identify precursors and indicators of incidents and analyse them to confirm the incident scope"),
    ("Containment, Eradication and Recovery",
     "limit damage, remove the cause of the incident and restore systems to normal operation"),
    ("Post-Incident Activity",
     "conduct a lessons-learned review, retain evidence and update procedures"),
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

    def emit(name_pool, def_pool, name, definition,
             name_phr, def_phr, fact_template):
        for phr in name_phr:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, name_pool, name),
                fact_template.format(name=name, definition=definition),
                shortname, instruction))
        for phr in def_phr:
            rows.append(make_mcq(
                rng, phr, definition, pick_distractors(rng, def_pool, definition),
                fact_template.format(name=name, definition=definition),
                shortname, instruction))

    # Pattern 1 -- risk formulas / definitions
    risk_names = [n for n, _ in RISK_FORMULAS]
    risk_defs = [d for _, d in RISK_FORMULAS]
    for name, defn in RISK_FORMULAS:
        emit(risk_names, risk_defs, name, defn,
             [f"Which risk-management term is defined as: {defn}?",
              f"Which concept does the following describe: {defn}?",
              f"In quantitative risk analysis, which term means: {defn}?"],
             [f"What does {name} mean in quantitative risk analysis?",
              f"How is {name} defined?",
              f"Which statement best describes {name}?"],
             "{name}: {definition}.")

    # Pattern 2 -- risk treatment options
    rt_names = [n for n, _ in RISK_TREATMENT]
    rt_defs = [d for _, d in RISK_TREATMENT]
    for name, defn in RISK_TREATMENT:
        emit(rt_names, rt_defs, name, defn,
             [f"Which risk-treatment option corresponds to: {defn}?",
              f"Which of the four risk-treatment strategies is described as: {defn}?",
              f"The action '{defn}' is which risk-treatment option?",
              f"How would you classify the risk-treatment action: {defn}?"],
             [f"What does the '{name}' risk-treatment strategy entail?",
              f"How is the '{name}' risk-treatment option defined?",
              f"Which statement best describes the '{name}' strategy?"],
             "The '{name}' risk-treatment option is to {definition}.")

    # Pattern 3 -- governance frameworks
    fw_names = [n for n, _ in FRAMEWORKS]
    fw_defs = [d for _, d in FRAMEWORKS]
    for name, defn in FRAMEWORKS:
        emit(fw_names, fw_defs, name, defn,
             [f"Which framework is described as: {defn}?",
              f"Which of the following frameworks fits this description: {defn}?",
              f"Which governance/risk framework matches: {defn}?"],
             [f"What is {name}?",
              f"How is {name} commonly described?",
              f"Which statement best characterises {name}?"],
             "{name} is {definition}.")

    # Pattern 4 -- NIST RMF steps
    rmf_names = [n for n, _ in NIST_RMF_STEPS]
    rmf_defs = [d for _, d in NIST_RMF_STEPS]
    for name, defn in NIST_RMF_STEPS:
        emit(rmf_names, rmf_defs, name, defn,
             [f"Which NIST RMF step performs the activity: {defn}?",
              f"Which step of the NIST Risk Management Framework is described as: {defn}?",
              f"In NIST SP 800-37, which RMF step covers: {defn}?"],
             [f"What is the purpose of the NIST RMF '{name}' step?",
              f"How is the NIST RMF '{name}' step defined?"],
             "The NIST RMF '{name}' step is to {definition}.")

    # Pattern 5 -- policy hierarchy
    ph_names = [n for n, _ in POLICY_HIERARCHY]
    ph_defs = [d for _, d in POLICY_HIERARCHY]
    for name, defn in POLICY_HIERARCHY:
        emit(ph_names, ph_defs, name, defn,
             [f"Which document type in the policy hierarchy is defined as: {defn}?",
              f"In the policy/standard/procedure/guideline hierarchy, which is: {defn}?",
              f"Which kind of governance document corresponds to: {defn}?"],
             [f"What is a {name} in the policy hierarchy?",
              f"How is a {name} defined in standard governance terminology?"],
             "A {name} is {definition}.")

    # Pattern 6 -- security roles
    role_names = [n for n, _ in ROLES]
    role_defs = [d for _, d in ROLES]
    for name, defn in ROLES:
        emit(role_names, role_defs, name, defn,
             [f"Which role is responsible for: {defn}?",
              f"Which security role is described as: {defn}?",
              f"Which of the following roles owns: {defn}?"],
             [f"What is the responsibility of the {name}?",
              f"How is the {name} role typically defined?"],
             "The {name} is {definition}.")

    # Pattern 7 -- three lines of defence
    lod_names = [n for n, _ in LINES_OF_DEFENCE]
    lod_defs = [d for _, d in LINES_OF_DEFENCE]
    for name, defn in LINES_OF_DEFENCE:
        emit(lod_names, lod_defs, name, defn,
             [f"In the Three Lines of Defence model, which line is the one where {defn}?",
              f"Which line of defence is the one where {defn}?",
              f"Which of the three lines of defence corresponds to: {defn}?",
              f"Under the Three Lines model, the activity '{defn}' is which line?"],
             [f"What does the {name} do in the Three Lines model?",
              f"How is the {name} defined?"],
             "In the Three Lines of Defence model, the {name} is the line where {definition}.")

    # Pattern 8 -- BCP/DR terms
    bcp_names = [n for n, _ in BCP_TERMS]
    bcp_defs = [d for _, d in BCP_TERMS]
    for name, defn in BCP_TERMS:
        emit(bcp_names, bcp_defs, name, defn,
             [f"Which business-continuity term is defined as: {defn}?",
              f"Which BCP/DR concept matches: {defn}?",
              f"In disaster-recovery planning, which term means: {defn}?"],
             [f"What is the {name}?",
              f"How is the {name} defined in business continuity planning?",
              f"Which statement best describes the {name}?"],
             "The {name} is {definition}.")

    # Pattern 9 -- NIST SP 800-61 IR phases
    ir_names = [n for n, _ in IR_PHASES]
    ir_defs = [d for _, d in IR_PHASES]
    for name, defn in IR_PHASES:
        emit(ir_names, ir_defs, name, defn,
             [f"In the NIST SP 800-61 incident-response life cycle, which phase: {defn}?",
              f"Which IR phase per NIST SP 800-61 is described as: {defn}?",
              f"Which incident-response phase covers: {defn}?",
              f"Per NIST SP 800-61, which phase is the one where you {defn}?"],
             [f"What is the purpose of the NIST SP 800-61 '{name}' phase?",
              f"How is the '{name}' phase of the IR life cycle defined?"],
             "The NIST SP 800-61 '{name}' phase is the one where you {definition}.")

    return rows
