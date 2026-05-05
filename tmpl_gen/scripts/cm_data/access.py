"""CM.ACCESS.1 knowledge tables and generator (~1,500 rows).

Covers access-control models (DAC/MAC/RBAC/ABAC/ReBAC), formal models
(Bell-LaPadula / Biba / Clark-Wilson / Brewer-Nash), authentication
mechanisms (passwords, OTP, biometrics, FIDO2, smart cards), federation
and SSO protocols (SAML, OAuth 2.0, OIDC, Kerberos, NTLM, LDAP), token
formats (JWT, SAML assertions, opaque bearer), and authorisation
principles (least privilege, separation of duties, need-to-know).
"""

from __future__ import annotations

from typing import Callable

from . import pick_distractors

# (model, one-line definition)
MODELS: list[tuple[str, str]] = [
    ("DAC (Discretionary Access Control)",
     "the resource owner decides who may access the resource"),
    ("MAC (Mandatory Access Control)",
     "the system enforces access decisions based on labels and clearances, not user choice"),
    ("RBAC (Role-Based Access Control)",
     "permissions are assigned to roles and users acquire permissions by being assigned to roles"),
    ("ABAC (Attribute-Based Access Control)",
     "policies evaluate attributes of subject, object, action and environment to make a decision"),
    ("ReBAC (Relationship-Based Access Control)",
     "access is granted based on relationships between subjects and objects in a graph"),
    ("RuBAC (Rule-Based Access Control)",
     "decisions are made by evaluating an explicit rule set independent of identity"),
    ("Bell-LaPadula model",
     "a confidentiality model that enforces 'no read up' and 'no write down'"),
    ("Biba model",
     "an integrity model that enforces 'no read down' and 'no write up'"),
    ("Clark-Wilson model",
     "an integrity model based on well-formed transactions and separation of duties"),
    ("Brewer-Nash (Chinese Wall) model",
     "a dynamic conflict-of-interest model used in financial/legal services"),
    ("Lattice-based access control",
     "access is decided by partial ordering of security labels arranged in a lattice"),
    ("Graham-Denning model",
     "defines a set of eight primitive protection rights operating on a protection matrix"),
    ("HRU (Harrison-Ruzzo-Ullman) model",
     "a protection matrix model that proved safety is undecidable in the general case"),
    ("Take-Grant model",
     "a graph-based protection model with take, grant, create and remove primitives"),
]

# (mechanism, factor_kind: knowledge|possession|inherence|location, brief)
AUTHN: list[tuple[str, str, str]] = [
    ("password", "knowledge", "a memorised secret string compared against a stored hash"),
    ("PIN", "knowledge", "a short numeric secret typically used with a possession factor"),
    ("security question", "knowledge", "a knowledge factor of low entropy and high guessability"),
    ("hardware token (TOTP)", "possession", "RFC 6238 time-based one-time password generator"),
    ("hardware token (HOTP)", "possession", "RFC 4226 counter-based one-time password generator"),
    ("smart card (PIV/CAC)", "possession",
     "X.509 certificate held on tamper-resistant hardware unlocked by PIN"),
    ("FIDO2/WebAuthn security key", "possession",
     "phishing-resistant public-key authenticator bound to the relying-party origin"),
    ("SMS one-time code", "possession",
     "OTP delivered via SMS; vulnerable to SIM-swap and SS7 interception"),
    ("push-notification approval", "possession",
     "out-of-band approval from a registered mobile authenticator app"),
    ("fingerprint", "inherence", "biometric matching of fingerprint minutiae"),
    ("face recognition", "inherence", "biometric matching of facial geometry"),
    ("iris scan", "inherence", "biometric matching of iris patterns"),
    ("voice recognition", "inherence", "biometric matching of voiceprint features"),
    ("geolocation policy", "location",
     "context factor that restricts access to specific geographic regions or networks"),
]

# (protocol, sentence)
PROTOCOLS: list[tuple[str, str]] = [
    ("SAML 2.0", "an XML-based federation protocol that issues signed assertions for browser SSO"),
    ("OAuth 2.0", "an authorisation-delegation framework that issues access tokens, not an authentication protocol"),
    ("OpenID Connect", "an identity layer on top of OAuth 2.0 that issues an ID token in JWT form"),
    ("Kerberos", "a symmetric-key authentication protocol that issues TGTs and service tickets via a KDC"),
    ("NTLM", "a Microsoft challenge-response protocol superseded by Kerberos in modern AD environments"),
    ("LDAP", "a directory access protocol commonly used as a backing store for authentication"),
    ("RADIUS", "an AAA protocol used for remote-access authentication, authorisation and accounting"),
    ("TACACS+", "a Cisco AAA protocol that separates authentication, authorisation and accounting and encrypts the body"),
    ("SCIM", "an HTTP-based protocol for cross-domain user and group provisioning"),
    ("CAS", "a single-sign-on protocol developed at Yale using service tickets and ticket-granting cookies"),
    ("X.509 client certificates", "TLS mutual authentication using PKI-issued certificates"),
    ("EAP-TLS", "a certificate-based EAP method offering mutual authentication for 802.1X"),
    ("EAP-PEAP", "an EAP method tunnelling MS-CHAPv2 inside a TLS server-authenticated tunnel"),
]

# (oauth_grant, when_to_use)
OAUTH_GRANTS: list[tuple[str, str]] = [
    ("authorization code with PKCE",
     "the recommended grant for confidential and public clients in browser/mobile flows"),
    ("client credentials",
     "machine-to-machine flows where the client itself is the resource owner"),
    ("device authorization grant",
     "input-constrained devices like TVs or CLI tools that cannot present a browser easily"),
    ("refresh token",
     "obtaining a fresh access token without re-prompting the user once authorised"),
    ("resource owner password credentials",
     "legacy flow now discouraged because the client handles the user's password"),
    ("implicit flow",
     "deprecated browser flow that returned tokens in the URL fragment"),
]

PRINCIPLES: list[tuple[str, str]] = [
    ("principle of least privilege",
     "subjects should be granted only the privileges strictly required for their tasks"),
    ("separation of duties",
     "no single person should be able to complete a sensitive transaction end-to-end alone"),
    ("need-to-know",
     "access to information should be limited to those whose duties require it"),
    ("defence in depth",
     "rely on multiple overlapping controls so a single failure does not lead to compromise"),
    ("complete mediation",
     "every access to every object must be checked for authority"),
    ("fail-safe defaults",
     "the default decision for an access check is to deny unless explicitly permitted"),
    ("psychological acceptability",
     "security mechanisms should not impose significant friction on legitimate users"),
    ("economy of mechanism",
     "the design of the security mechanism should be as simple and small as possible"),
    ("open design",
     "security should not depend on the secrecy of the design or implementation"),
    ("least common mechanism",
     "minimise the amount of mechanism shared by, and depended on by, multiple users"),
]

TOKENS: list[tuple[str, str]] = [
    ("JWT (JSON Web Token)",
     "a compact, URL-safe token consisting of a base64-encoded header, payload and signature"),
    ("SAML assertion",
     "an XML-signed statement issued by an identity provider conveying authentication and attributes"),
    ("OAuth 2.0 bearer token",
     "an opaque or structured access token presented in the Authorization header by anyone holding it"),
    ("Kerberos ticket",
     "a symmetric-encrypted credential issued by a KDC and presented to a service"),
    ("PASETO",
     "a token format designed as a safer alternative to JWT with versioned algorithm suites"),
]



def generate(rng, target: int, instruction: str, shortname: str,
             make_mcq: Callable) -> list[dict]:
    rows: list[dict] = []
    # Each pattern emits ~371 rows per single pass; loop until target met.
    while len(rows) < target:
        rows.extend(_one_pass(rng, instruction, shortname, make_mcq))
    return rows[:target]


def _one_pass(rng, instruction: str, shortname: str,
              make_mcq: Callable) -> list[dict]:
    rows: list[dict] = []
    model_names = [m for m, _ in MODELS]
    model_defs = [d for _, d in MODELS]

    # Pattern 1 -- model -> definition
    p1_phr = [
        "Which statement best describes {m}?",
        "How is {m} defined in standard access-control literature?",
        "Which of the following correctly characterises {m}?",
        "Pick the option that matches the definition of {m}.",
    ]
    for m, d in MODELS:
        for phr in p1_phr:
            rows.append(make_mcq(
                rng, phr.format(m=m), d,
                pick_distractors(rng, model_defs, d),
                f"{m} is the model under which {d}.",
                shortname, instruction))

    # Pattern 2 -- definition -> model
    p2_phr = [
        "Which access-control model is described by the following property: \"{d}\"?",
        "An access policy where {d} corresponds to which model?",
        "Which model best matches this characterisation: \"{d}\"?",
    ]
    for m, d in MODELS:
        for phr in p2_phr:
            rows.append(make_mcq(
                rng, phr.format(d=d), m,
                pick_distractors(rng, model_names, m),
                f"The property '{d}' is the defining trait of {m}.",
                shortname, instruction))

    # Pattern 3 -- authentication factor classification
    factor_label = {
        "knowledge": "something you know (knowledge factor)",
        "possession": "something you have (possession factor)",
        "inherence": "something you are (inherence factor)",
        "location": "somewhere you are (location/context factor)",
    }
    auth_brief_pool = [b for _, _, b in AUTHN]
    for mech, fac, brief in AUTHN:
        ans = factor_label[fac]
        opts = list(factor_label.values())
        for phr in [
            f"Which authentication factor category does {mech} belong to?",
            f"How is {mech} classified among the four authentication factors?",
            f"In the standard authentication-factor taxonomy, {mech} is an example of which factor?",
            f"Which factor type does {mech} provide?",
        ]:
            rows.append(make_mcq(
                rng, phr, ans, [o for o in opts if o != ans],
                f"{mech} is {brief}; this is {ans}.",
                shortname, instruction))
        for phr in [
            f"Which statement most accurately describes {mech}?",
            f"What does {mech} actually do?",
        ]:
            rows.append(make_mcq(
                rng, phr, brief, pick_distractors(rng, auth_brief_pool, brief),
                f"{mech} is {brief}.", shortname, instruction))

    # Pattern 4 -- protocols
    proto_briefs = [b for _, b in PROTOCOLS]
    proto_names = [n for n, _ in PROTOCOLS]
    for name, brief in PROTOCOLS:
        for phr in [
            f"Which statement best characterises {name}?",
            f"How would you describe {name} to a junior analyst?",
            f"What is {name} primarily?",
            f"Which option most accurately defines {name}?",
        ]:
            rows.append(make_mcq(
                rng, phr, brief, pick_distractors(rng, proto_briefs, brief),
                f"{name} is {brief}.", shortname, instruction))
        for phr in [
            f"Which protocol matches this description: \"{brief}\"?",
            f"Which option corresponds to: \"{brief}\"?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, proto_names, name),
                f"{name} is {brief}.", shortname, instruction))

    # Pattern 5 -- OAuth grant selection
    grant_names = [g for g, _ in OAUTH_GRANTS]
    grant_uses = [u for _, u in OAUTH_GRANTS]
    for grant, use in OAUTH_GRANTS:
        for phr in [
            f"Which OAuth 2.0 grant type is appropriate when: {use}?",
            f"Which grant should be used in the following scenario: {use}?",
            f"For the use case '{use}', which OAuth 2.0 grant is recommended?",
            f"Which OAuth 2.0 grant fits this scenario best: {use}?",
        ]:
            rows.append(make_mcq(
                rng, phr, grant, pick_distractors(rng, grant_names, grant),
                f"The {grant} grant is the appropriate choice when {use}.",
                shortname, instruction))
        for phr in [
            f"What is the {grant} grant used for?",
            f"In what scenario is the {grant} grant intended to be used?",
        ]:
            rows.append(make_mcq(
                rng, phr, use, pick_distractors(rng, grant_uses, use),
                f"The {grant} grant is used when {use}.",
                shortname, instruction))

    # Pattern 6 -- security principles
    principle_names = [n for n, _ in PRINCIPLES]
    principle_defs = [d for _, d in PRINCIPLES]
    for name, definition in PRINCIPLES:
        for phr in [
            f"Which security principle states that {definition}?",
            f"The rule that '{definition}' is which classical security principle?",
            f"Which principle does the following describe: {definition}?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, principle_names, name),
                f"The principle that '{definition}' is {name}.",
                shortname, instruction))
        for phr in [
            f"Which statement is the canonical formulation of the {name}?",
            f"How is the {name} formally stated?",
        ]:
            rows.append(make_mcq(
                rng, phr, definition, pick_distractors(rng, principle_defs, definition),
                f"The {name} states that {definition}.",
                shortname, instruction))

    # Pattern 7 -- token formats
    token_names = [n for n, _ in TOKENS]
    token_defs = [d for _, d in TOKENS]
    for name, definition in TOKENS:
        for phr in [
            f"Which statement best describes {name}?",
            f"How is {name} structured?",
            f"What is {name}?",
        ]:
            rows.append(make_mcq(
                rng, phr, definition, pick_distractors(rng, token_defs, definition),
                f"{name} is {definition}.", shortname, instruction))
        for phr in [
            f"Which token format matches this description: {definition}?",
            f"Which option corresponds to: {definition}?",
        ]:
            rows.append(make_mcq(
                rng, phr, name, pick_distractors(rng, token_names, name),
                f"{name} is {definition}.", shortname, instruction))

    return rows
