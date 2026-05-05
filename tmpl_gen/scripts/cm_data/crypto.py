"""CM.CRYPTO.1 knowledge tables and generator (~1,500 rows)."""

from __future__ import annotations

from typing import Callable

# (name, category) — categories drawn from the CATEGORIES list below so that
# distractors are mutually-exclusive plausible alternatives.
ALGORITHMS: list[tuple[str, str]] = [
    ("AES", "symmetric block cipher"), ("DES", "symmetric block cipher"),
    ("3DES", "symmetric block cipher"), ("Blowfish", "symmetric block cipher"),
    ("Twofish", "symmetric block cipher"), ("Camellia", "symmetric block cipher"),
    ("ARIA", "symmetric block cipher"), ("Serpent", "symmetric block cipher"),
    ("CAST-128", "symmetric block cipher"), ("IDEA", "symmetric block cipher"),
    ("RC2", "symmetric block cipher"), ("RC5", "symmetric block cipher"),
    ("RC6", "symmetric block cipher"), ("SM4", "symmetric block cipher"),
    ("ChaCha20", "symmetric stream cipher"), ("Salsa20", "symmetric stream cipher"),
    ("RC4", "symmetric stream cipher"), ("A5/1", "symmetric stream cipher"),
    ("A5/2", "symmetric stream cipher"), ("Trivium", "symmetric stream cipher"),
    ("RSA", "asymmetric public-key encryption"),
    ("ElGamal", "asymmetric public-key encryption"),
    ("Paillier", "asymmetric public-key encryption"),
    ("DSA", "digital signature algorithm"),
    ("ECDSA", "digital signature algorithm"),
    ("EdDSA", "digital signature algorithm"),
    ("Ed25519", "digital signature algorithm"),
    ("Ed448", "digital signature algorithm"),
    ("RSA-PSS", "digital signature algorithm"),
    ("DH", "key agreement protocol"), ("ECDH", "key agreement protocol"),
    ("X25519", "key agreement protocol"), ("X448", "key agreement protocol"),
    ("Kyber", "post-quantum key encapsulation mechanism"),
    ("ML-KEM", "post-quantum key encapsulation mechanism"),
    ("Dilithium", "post-quantum digital signature"),
    ("ML-DSA", "post-quantum digital signature"),
    ("SPHINCS+", "post-quantum digital signature"),
    ("Falcon", "post-quantum digital signature"),
    ("MD5", "cryptographic hash function"), ("SHA-1", "cryptographic hash function"),
    ("SHA-224", "cryptographic hash function"), ("SHA-256", "cryptographic hash function"),
    ("SHA-384", "cryptographic hash function"), ("SHA-512", "cryptographic hash function"),
    ("SHA3-256", "cryptographic hash function"), ("SHA3-512", "cryptographic hash function"),
    ("SHAKE128", "extendable-output hash function"),
    ("SHAKE256", "extendable-output hash function"),
    ("BLAKE2b", "cryptographic hash function"), ("BLAKE3", "cryptographic hash function"),
    ("RIPEMD-160", "cryptographic hash function"),
    ("Whirlpool", "cryptographic hash function"),
    ("HMAC", "message authentication code"), ("CMAC", "message authentication code"),
    ("GMAC", "message authentication code"), ("Poly1305", "message authentication code"),
    ("SipHash", "message authentication code"), ("KMAC", "message authentication code"),
    ("PBKDF2", "key derivation function"), ("HKDF", "key derivation function"),
    ("scrypt", "password hashing function"), ("bcrypt", "password hashing function"),
    ("Argon2", "password hashing function"),
]

CATEGORIES: list[str] = [
    "symmetric block cipher", "symmetric stream cipher",
    "asymmetric public-key encryption", "digital signature algorithm",
    "key agreement protocol", "cryptographic hash function",
    "extendable-output hash function", "message authentication code",
    "key derivation function", "password hashing function",
    "post-quantum key encapsulation mechanism",
    "post-quantum digital signature",
]

# (algorithm, key_size_bits, block_size_bits_or_None, standard, year)
KEY_BLOCK: list[tuple[str, int, int | None, str, int]] = [
    ("AES-128", 128, 128, "FIPS 197", 2001),
    ("AES-192", 192, 128, "FIPS 197", 2001),
    ("AES-256", 256, 128, "FIPS 197", 2001),
    ("DES", 56, 64, "FIPS 46-3", 1977),
    ("3DES", 168, 64, "NIST SP 800-67", 1998),
    ("Blowfish", 128, 64, "Schneier 1993", 1993),
    ("Twofish", 256, 128, "AES finalist 1998", 1998),
    ("Camellia", 256, 128, "RFC 3713", 2000),
    ("ChaCha20", 256, None, "RFC 8439", 2008),
    ("RC4", 128, None, "RSA Security 1987", 1987),
    ("RSA-2048", 2048, None, "FIPS 186-5", 2023),
    ("RSA-3072", 3072, None, "FIPS 186-5", 2023),
    ("RSA-4096", 4096, None, "FIPS 186-5", 2023),
    ("ECDSA P-256", 256, None, "FIPS 186-5", 2023),
    ("ECDSA P-384", 384, None, "FIPS 186-5", 2023),
    ("ECDSA P-521", 521, None, "FIPS 186-5", 2023),
    ("Ed25519", 256, None, "RFC 8032", 2017),
    ("Ed448", 448, None, "RFC 8032", 2017),
]

# (hash, output_bits, standard)
HASH_SIZES: list[tuple[str, int, str]] = [
    ("MD5", 128, "RFC 1321"), ("SHA-1", 160, "FIPS 180-4"),
    ("SHA-224", 224, "FIPS 180-4"), ("SHA-256", 256, "FIPS 180-4"),
    ("SHA-384", 384, "FIPS 180-4"), ("SHA-512", 512, "FIPS 180-4"),
    ("SHA3-224", 224, "FIPS 202"), ("SHA3-256", 256, "FIPS 202"),
    ("SHA3-384", 384, "FIPS 202"), ("SHA3-512", 512, "FIPS 202"),
    ("BLAKE2s", 256, "RFC 7693"), ("BLAKE2b", 512, "RFC 7693"),
    ("RIPEMD-160", 160, "ISO/IEC 10118-3"), ("Whirlpool", 512, "ISO/IEC 10118-3"),
]

# (mode, type, parallel_enc, requires_unique_iv, provides_integrity, supports_aead)
MODES: list[tuple[str, str, bool, bool, bool, bool]] = [
    ("ECB", "confidentiality-only block mode", True,  False, False, False),
    ("CBC", "confidentiality-only block mode", False, True,  False, False),
    ("CFB", "confidentiality-only block mode", False, True,  False, False),
    ("OFB", "confidentiality-only block mode", False, True,  False, False),
    ("CTR", "confidentiality-only block mode", True,  True,  False, False),
    ("XTS", "tweakable storage encryption mode", True, True, False, False),
    ("GCM", "authenticated encryption with associated data", True, True, True, True),
    ("CCM", "authenticated encryption with associated data", False, True, True, True),
    ("EAX", "authenticated encryption with associated data", True, True, True, True),
    ("OCB", "authenticated encryption with associated data", True, True, True, True),
    ("GCM-SIV", "nonce-misuse-resistant authenticated encryption", True, False, True, True),
    ("ChaCha20-Poly1305", "authenticated encryption with associated data",
     True, True, True, True),
]



# (attack, target, mitigation_one_line)
ATTACKS: list[tuple[str, str, str]] = [
    ("padding oracle attack", "CBC mode without authenticated encryption",
     "use AEAD modes such as AES-GCM or ChaCha20-Poly1305"),
    ("length-extension attack", "Merkle-Damgaard hash functions like SHA-256 used as a MAC",
     "use HMAC instead of bare hash construction"),
    ("birthday attack", "n-bit hash functions reduced to n/2-bit collision resistance",
     "use a hash with at least 256-bit output for 128-bit collision resistance"),
    ("Bleichenbacher attack", "RSA PKCS#1 v1.5 padding oracle",
     "use RSA-OAEP for encryption and RSA-PSS for signing"),
    ("BEAST", "TLS 1.0 CBC ciphersuites with predictable IV",
     "upgrade to TLS 1.2 or 1.3 with AEAD ciphers"),
    ("CRIME", "TLS compression leaking session cookies",
     "disable TLS-level compression"),
    ("BREACH", "HTTP-level compression leaking secrets in TLS",
     "disable HTTP compression for sensitive responses"),
    ("POODLE", "SSL 3.0 CBC padding oracle",
     "disable SSL 3.0 entirely"),
    ("Heartbleed", "OpenSSL TLS heartbeat extension memory disclosure (CVE-2014-0160)",
     "patch OpenSSL to >=1.0.1g and rotate keys/certs"),
    ("Logjam", "TLS DH key exchange downgrade to 512-bit primes",
     "use >=2048-bit DH groups or switch to ECDHE"),
    ("FREAK", "TLS RSA export-grade ciphersuite downgrade",
     "disable EXPORT ciphersuites"),
    ("ROBOT", "RSA PKCS#1 v1.5 oracle in TLS implementations",
     "disable RSA key transport in TLS, use ECDHE"),
    ("DROWN", "SSLv2 cross-protocol attack on shared RSA keys",
     "disable SSLv2 and never share keys across SSL/TLS versions"),
    ("Lucky 13", "TLS CBC timing side-channel on MAC verification",
     "use AEAD ciphers; constant-time MAC verification"),
    ("Sweet32", "64-bit block ciphers (3DES, Blowfish) in long-lived TLS sessions",
     "switch to 128-bit block ciphers like AES"),
    ("RC4 biases", "RC4 keystream statistical biases in TLS",
     "remove RC4 from TLS cipher suites (RFC 7465)"),
    ("MD5 collision attack", "MD5 hash function pre-image and collision resistance",
     "deprecate MD5; use SHA-256 or SHA-3 family"),
    ("SHA-1 SHAttered collision", "SHA-1 collision resistance (Google 2017)",
     "deprecate SHA-1; use SHA-256 or SHA-3"),
    ("nonce reuse in GCM", "AES-GCM authentication when the same nonce is reused",
     "use a 96-bit random nonce or AES-GCM-SIV for misuse resistance"),
    ("Dual_EC_DRBG backdoor", "NIST SP 800-90A Dual_EC_DRBG default constants",
     "use HMAC_DRBG, CTR_DRBG, or Hash_DRBG instead"),
    ("side-channel timing attack", "non-constant-time implementations of crypto primitives",
     "use constant-time implementations (e.g. libsodium, BoringSSL)"),
    ("cache-timing attack", "table-lookup AES implementations sharing CPU caches",
     "use AES-NI hardware instructions or bitsliced implementations"),
    ("rainbow table attack", "unsalted password hashes",
     "salt every password hash with a unique per-user random value"),
    ("dictionary attack", "weak password hashing functions like MD5/SHA-1",
     "use Argon2id, scrypt, or bcrypt with appropriate work factor"),
    ("downgrade attack", "TLS clients accepting weaker protocol versions",
     "enforce TLS_FALLBACK_SCSV or pin minimum TLS version"),
    ("session fixation", "applications accepting attacker-supplied session IDs",
     "regenerate session IDs after authentication"),
]

# (protocol, key_fact_sentence)
PROTOCOLS: list[tuple[str, str]] = [
    ("TLS 1.3", "removed RSA key exchange and mandates ephemeral key agreement (forward secrecy)"),
    ("TLS 1.3", "completes handshake in 1-RTT (or 0-RTT for resumption)"),
    ("TLS 1.3", "encrypts all handshake messages after ServerHello"),
    ("TLS 1.3", "removed support for CBC ciphersuites and only allows AEAD"),
    ("TLS 1.2", "introduced authenticated encryption (GCM, CCM) ciphersuites"),
    ("TLS 1.0", "deprecated by RFC 8996 due to BEAST and other CBC weaknesses"),
    ("DTLS", "TLS adapted for unreliable transports (UDP)"),
    ("QUIC", "transport-layer protocol with built-in TLS 1.3 cryptography"),
    ("IPsec ESP", "provides confidentiality and authentication at the IP layer"),
    ("IPsec AH", "provides only authentication and integrity at the IP layer"),
    ("IKEv2", "the modern key exchange protocol used to negotiate IPsec SAs"),
    ("SSH", "uses Diffie-Hellman or ECDH for key exchange and AEAD ciphers for transport"),
    ("Kerberos", "uses symmetric tickets and a trusted KDC for authentication"),
    ("SAML", "an XML-based federation protocol used for browser SSO"),
    ("OAuth 2.0", "an authorisation delegation framework, not an authentication protocol"),
    ("OpenID Connect", "an identity layer built on top of OAuth 2.0 using ID tokens"),
    ("WPA3", "uses Simultaneous Authentication of Equals (SAE) for password-based auth"),
    ("WPA2", "uses a 4-way handshake with PSK or 802.1X EAP authentication"),
    ("S/MIME", "uses X.509 certificates and CMS for email signing and encryption"),
    ("PGP/OpenPGP", "uses a web-of-trust model with RSA, ElGamal, ECDH and ECDSA"),
    ("DNSSEC", "signs DNS responses with public-key cryptography (RRSIG records)"),
    ("DKIM", "signs email headers/body with a domain key published in DNS"),
    ("SPF", "publishes authorised sending mail-servers in a DNS TXT record"),
    ("DMARC", "policy layer that aligns SPF and DKIM results to the From: domain"),
    ("HSTS", "instructs browsers to always use HTTPS for a given domain"),
    ("Certificate Transparency", "RFC 9162 logs that publish all issued TLS certificates"),
    ("OCSP", "online protocol for checking the revocation status of X.509 certificates"),
    ("OCSP stapling", "lets the TLS server present a fresh OCSP response in the handshake"),
    ("FIDO2/WebAuthn", "phishing-resistant public-key authentication bound to the origin"),
    ("U2F", "the predecessor to WebAuthn; second-factor public-key challenge protocol"),
    ("TOTP", "RFC 6238 time-based one-time password derived from a shared HMAC secret"),
    ("HOTP", "RFC 4226 HMAC-based counter-driven one-time password"),
]



def _pick_distractors(rng, pool: list, correct, k: int = 4) -> list:
    cands = [c for c in pool if c != correct]
    rng.shuffle(cands)
    return cands[:k]


def generate(rng, target: int, instruction: str, shortname: str,
             make_mcq: Callable) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < target:
        rows.extend(_one_pass(rng, instruction, shortname, make_mcq))
    return rows[:target]


def _one_pass(rng, instruction: str, shortname: str,
              make_mcq: Callable) -> list[dict]:
    rows: list[dict] = []

    # Pattern 1 -- algorithm -> category (4 paraphrases per algo)
    p1_phr = [
        "Which category best describes the {a} algorithm?",
        "{a} is classified as which type of cryptographic primitive?",
        "In standard cryptographic taxonomy, {a} belongs to which family?",
        "If asked to categorise {a}, which of the following best applies?",
        "What kind of cryptographic primitive is {a}?",
        "Which family does {a} fall into?",
    ]
    for a, cat in ALGORITHMS:
        for phr in p1_phr:
            rows.append(make_mcq(
                rng, phr.format(a=a), cat,
                _pick_distractors(rng, CATEGORIES, cat),
                f"{a} is a {cat}.", shortname, instruction))

    # Pattern 2 -- category -> algorithm (4 paraphrases per algo)
    by_cat: dict[str, list[str]] = {}
    for a, c in ALGORITHMS:
        by_cat.setdefault(c, []).append(a)
    p2_phr = [
        "Which of the following is a {c}?",
        "Which algorithm in the list is an example of a {c}?",
        "Pick the {c} from the choices below.",
        "Which option is a representative {c}?",
    ]
    for a, cat in ALGORITHMS:
        peers = [n for n, c in ALGORITHMS if c != cat]
        for phr in p2_phr:
            rows.append(make_mcq(
                rng, phr.format(c=cat), a,
                _pick_distractors(rng, peers, a),
                f"{a} is a {cat}.", shortname, instruction))

    # Pattern 3 -- key sizes / block sizes / standards from KEY_BLOCK
    sizes_pool = [56, 64, 80, 112, 128, 160, 192, 224, 256, 384, 448, 521,
                  1024, 2048, 3072, 4096, 7680]
    std_pool = ["FIPS 197", "FIPS 186-5", "FIPS 180-4", "FIPS 202",
                "NIST SP 800-67", "RFC 8032", "RFC 8439", "RFC 3713",
                "Schneier 1993", "AES finalist 1998", "RSA Security 1987"]
    for algo, ks, bs, std, _yr in KEY_BLOCK:
        # key size
        for phr in [f"What is the standard key size in bits for {algo}?",
                    f"How many key bits does {algo} use in its standard configuration?",
                    f"Which value (in bits) is the canonical key length for {algo}?"]:
            rows.append(make_mcq(
                rng, phr, str(ks),
                [str(d) for d in _pick_distractors(rng, sizes_pool, ks)],
                f"{algo} uses a {ks}-bit key per its specification.",
                shortname, instruction))
        # block size (skip stream ciphers / asymmetric where bs is None)
        if bs is not None:
            for phr in [f"What is the block size in bits used by {algo}?",
                        f"How wide is the {algo} cipher block, in bits?"]:
                rows.append(make_mcq(
                    rng, phr, str(bs),
                    [str(d) for d in _pick_distractors(rng, sizes_pool, bs)],
                    f"{algo} operates on a {bs}-bit block.",
                    shortname, instruction))
        # standard
        for phr in [f"Which standard or specification defines {algo}?",
                    f"Under which document is {algo} normatively specified?"]:
            rows.append(make_mcq(
                rng, phr, std,
                _pick_distractors(rng, std_pool, std),
                f"{algo} is specified by {std}.",
                shortname, instruction))

    # Pattern 4 -- hash output sizes / standards
    hash_sizes_pool = [128, 160, 224, 256, 384, 512]
    hash_std_pool = ["FIPS 180-4", "FIPS 202", "RFC 1321", "RFC 7693",
                     "ISO/IEC 10118-3"]
    for h, sz, std in HASH_SIZES:
        for phr in [f"What is the output digest size in bits of {h}?",
                    f"How many bits does the {h} hash function produce?",
                    f"Which value (in bits) is the digest length emitted by {h}?",
                    f"What is the native output width, in bits, of {h}?"]:
            rows.append(make_mcq(
                rng, phr, str(sz),
                [str(d) for d in _pick_distractors(rng, hash_sizes_pool, sz)],
                f"{h} produces a {sz}-bit digest.",
                shortname, instruction))
        for phr in [f"Which standard specifies the {h} hash function?",
                    f"Under which standard is {h} defined?"]:
            rows.append(make_mcq(
                rng, phr, std,
                _pick_distractors(rng, hash_std_pool, std),
                f"{h} is specified in {std}.",
                shortname, instruction))

    # Pattern 5 -- modes of operation properties
    aead_modes = [m for (m, t, _, _, _, aead) in MODES if aead]
    cb_modes = [m for (m, _, _, _, _, _) in MODES]
    for m, t, par, iv, integ, aead in MODES:
        for phr in [f"Which classification best fits the {m} mode of operation?",
                    f"How is the {m} mode of operation classified?",
                    f"Which category does the {m} mode belong to?"]:
            rows.append(make_mcq(
                rng, phr, t,
                _pick_distractors(rng,
                    list(set(c for (_, c, *_rest) in MODES) - {t}), t),
                f"{m} is a {t}.", shortname, instruction))
        ans_aead = "yes -- it is an AEAD mode" if aead else "no -- it provides confidentiality only"
        for phr in [f"Does the {m} mode provide authenticated encryption (AEAD)?",
                    f"Is the {m} mode of operation an authenticated encryption scheme?"]:
            rows.append(make_mcq(
                rng, phr, ans_aead,
                ["yes -- it is an AEAD mode", "no -- it provides confidentiality only",
                 "yes, but only when combined with HMAC",
                 "no, but it provides integrity via CRC"],
                f"{m} is{'' if aead else ' not'} an AEAD mode.",
                shortname, instruction))
    # Pattern 5b -- "which AEAD mode?"
    for m in aead_modes:
        rows.append(make_mcq(
            rng, "Which of the following is an authenticated encryption (AEAD) mode of operation?",
            m, [d for d in cb_modes if d not in aead_modes][:4],
            f"{m} is an AEAD mode of operation.",
            shortname, instruction))

    # Pattern 6 -- attacks (target + mitigation per attack, multiple paraphrases)
    all_targets = [t for (_, t, _) in ATTACKS]
    all_mits = [m for (_, _, m) in ATTACKS]
    for atk, tgt, mit in ATTACKS:
        for phr in [f"What does the {atk} primarily exploit?",
                    f"Which weakness is the {atk} aimed at?",
                    f"What is the underlying vulnerability behind the {atk}?"]:
            rows.append(make_mcq(
                rng, phr, tgt, _pick_distractors(rng, all_targets, tgt),
                f"The {atk} targets {tgt}.", shortname, instruction))
        for phr in [f"What is the recommended mitigation for the {atk}?",
                    f"Which control mitigates the {atk}?",
                    f"How would you remediate exposure to the {atk}?"]:
            rows.append(make_mcq(
                rng, phr, mit, _pick_distractors(rng, all_mits, mit),
                f"To mitigate the {atk}, {mit}.", shortname, instruction))


    # Pattern 7 -- protocols / standards (description <-> name)
    proto_names = [n for n, _ in PROTOCOLS]
    proto_facts = [f for _, f in PROTOCOLS]
    for name, fact in PROTOCOLS:
        for phr in [f"Which statement most accurately describes {name}?",
                    f"How is {name} characterised in standard references?",
                    f"Which fact about {name} is correct?",
                    f"What is true of {name}?"]:
            rows.append(make_mcq(
                rng, phr, fact, _pick_distractors(rng, proto_facts, fact),
                f"{name} {fact}.", shortname, instruction))
        for phr in [f"Which protocol or standard matches this description: \"{fact}\"?",
                    f"Which option corresponds to: \"{fact}\"?",
                    f"Which protocol/standard does the following describe: {fact}?"]:
            rows.append(make_mcq(
                rng, phr, name, _pick_distractors(rng, proto_names, name),
                f"{name} {fact}.", shortname, instruction))

    return rows
