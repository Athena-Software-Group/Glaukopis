# Where to submit Glaukopis next — venue analysis (2026-10-24)

**Constraint set by the author:** the paper must satisfy an FAU EECS doctoral committee's single-publication requirement, so the venue must be unambiguously reputable to computer scientists (Web of Science SCIE strongly preferred, ESCI at most; Scopus; a recognized publisher; not MDPI/Hindawi), and within that set the choice should maximize probability of acceptance and speed of a decision. Computers & Security is excluded (rejected). Figures below were collected on 2026-10-24 from publisher pages and independent mirrors; Elsevier's own "Journal Insights" timing pages were not reachable from the research environment, so Elsevier decision times are marked unverified and should be read off the ScienceDirect "Insights" tab before submitting.

## Recommendation

| Rank | Venue | Why | Watch out for |
|---|---|---|---|
| **1** | **International Journal of Information Security** (Springer) — SCIE, 2025 JIF 5.0, JCR Q1 (Software Eng.; Theory & Methods), Q2 (Information Systems) | Closest reputable sibling to C&S; publishes CTI quality/actionability and LLM-for-security papers monthly in 2025–26; Springer's reported **median 9 days to first decision** (includes desk decisions, so a desk reject costs little); DBLP/ACM-DL indexed; no committee will question it. | **Now fully gold OA: APC US$3,690** (€2,990). Check whether FAU Libraries has a Springer Nature read-and-publish agreement; request a discretionary waiver *at submission* (cannot be requested later). Reviewer pools overlap with C&S, so the two unresolved reviewer points (matched control, blind test set) must be handled as v1.6 does — explicitly and up front. Uses `sn-jnl.cls`; port the front matter. |
| **2** | **Journal of Information Security and Applications** (Elsevier) — SCIE, JIF ≈ 4.4 (one mirror says 3.7), Scopus Q1 / JCR Q2 | Same publisher and tier as C&S, so the committee has already implicitly accepted it; aims statement is practice-oriented ("original research and practice-driven applications"), which fits a systems paper with honest negatives; **hybrid, so the subscription route is free**; elsarticle template — v1.6 compiles as-is. Recent papers: DroidTTP (CTI TTP mapping), fine-tuned-LLM anomaly detection, LLMs for software security. | Decision speed not verified (check the Insights tab). Scope text excludes "routine applications of existing AI/ML models" — the cover letter must lead with Ariadne/Sophia (traceable graph-materialized data) and the diagnostic protocol, not the fine-tune. CCF-C. |
| **3** | **IEEE Access** (IEEE) — SCIE, 2025 JIF 4.2, JCR Q2 | Fastest reliable path: IEEE reports **~4 weeks to a binary decision** and 4–6 weeks submission-to-publication; its scope statement explicitly welcomes applied engineering **including negative results**, which is exactly this paper; IEEE brand and SCIE indexing meet the formal criteria. | Some picky CS committees discount IEEE Access as a mega-journal (~14k papers/yr). **Ask Dr. Cardei / the committee chair before relying on it.** Binary decision: "needs a matched control" is a rejection, not a revision, so submit only the strengthened version. APC US$2,160 + tax (IEEE member discount available). Template `ieeeaccess.cls`. |

**Alternate #3 if the committee accepts ESCI indexing:** *Cybersecurity* (Springer Nature, sponsored by IIE-CAS) — the best thematic fit of any venue (2025–26 papers: KGAgent4CTI, Bridg-ICS knowledge graphs for threat analytics, knowledge-injected NER for CTI), CCF-B, 2025 JIF 4.1, **median 15 days to first decision, APC US$0** (sponsor-paid). Weaknesses: ESCI not SCIE, and Chinese-Academy-of-Sciences sponsorship optics may matter to some committee members. *ACM Digital Threats: Research and Practice* (ESCI, JIF 3.1, APC US$1,800 non-member; FAU is not in ACM Open) is the other ESCI alternate with a strong practice-bridging fit but unpublished review times.

**Suggested sequence:** submit to IJIS first (fast desk decision tells you quickly whether you are in review); if desk-rejected or if the APC cannot be covered, go to JISA; keep IEEE Access as the time-boxed fallback once the committee confirms it is acceptable. Do not submit to two journals at once.

## Full comparison

| Journal (publisher) | WoS / Scopus | 2025 JIF / CiteScore | Quartile | Time to first decision | Time to accept | Acceptance | OA / APC | Fit (1–5) |
|---|---|---|---|---|---|---|---|---|
| Int. J. Information Security (Springer) | SCIE + Scopus | 5.0 / 8.8 | JCR Q1/Q2 | 9 d median (publisher) | not published (~12 mo total, user-reported) | not published | Gold OA, $3,690 | 5 |
| J. Information Security and Applications (Elsevier) | SCIE + Scopus | ~4.4 / 8.7 | Scopus Q1, JCR Q2 | not verified | not verified | not published | Hybrid; free subscription route | 4 |
| IEEE Access | SCIE + Scopus | 4.2 / 9.3 | JCR Q2 | ~4 wk (publisher) | 4–6 wk to publication | ~20% (publisher) | Gold OA, $2,160 | 4 |
| Cybersecurity (Springer Nature / IIE-CAS) | ESCI + Scopus | 4.1 / — | SJR Q1; CCF-B | 15 d median (publisher) | not published | not published | Gold OA, $0 | 5 |
| ACM DTRAP | ESCI + Scopus | 3.1 / ~6.9 | Scopus Q1 | not published | not published | not published | Gold OA (ACM, 2026), $1,300 member / $1,800 | 5 |
| Computers & Electrical Engineering (Elsevier) | SCIE + Scopus | 5.5 / 12.2 | JCR Q1/Q2 | conflicting reports (9 wk vs ~12 mo) | — | — | Hybrid | 3 |
| Array (Elsevier) | ESCI + Scopus | 5.3 / 10.1 | JCR Q1 (ESCI) | ~15 wk review (LetPub) | — | — | Gold OA, ~$1,420 | 3 |
| J. Cybersecurity and Privacy (MDPI) | ESCI + Scopus | 3.8 / ~8.8 | JCR Q2 | 23.7 d median | +5.6 d to publish | — | Gold OA, CHF 1,200 | 5 scope; publisher concern |
| Expert Systems with Applications (Elsevier) | SCIE | 9.4 | Q1 | 62 d after review | 147 d | 11% | Hybrid | 2–3 |
| Knowledge-Based Systems (Elsevier) | SCIE | 8.0 | Q1 | ~2.8 mo first round | ~8.4 mo | ~20–25% | Hybrid, $3,240 | 2–3 |
| IEEE TDSC | SCIE | 6.8 | Q1 | multi-month | >12 mo typical | low | Hybrid | 3 — not advised for a time-bound requirement |
| J. Computer Security (IOS) | ESCI (per mirror) | 1.3 | low | slow | — | — | Hybrid | 2 (formal/theory) |
| Security and Privacy (Wiley) | ESCI | 2.9 | — | not verified | — | — | Hybrid | 4; no advantage over IJIS/JISA |
| Cyber Security and Applications (KeAi) | Scopus only (no WoS evidence) | CiteScore 17.8 | — | — | — | — | OA | fails the WoS constraint |

## Venues to avoid for this paper

- **MDPI (JCP, Electronics), Hindawi** — JCP is a genuine scope fit with fast decisions, but a single required publication is not the place to test the committee's tolerance for MDPI.
- **KeAi Cyber Security and Applications** — Scopus-only.
- **TDSC, Information Fusion, Information Sciences, ESWA, KBS, JNCA, FGCS** — reputable but slow (months to a year), selective (11–25%), and reviewed by ML/systems audiences who will demand ML novelty and will not credit the CTI engineering or the honest MMLU-Pro loss.
- **IOS J. Computer Security** — low IF, formal-methods leaning.

## Unverified items to check before submitting

- JISA / CEE / Array time-to-first-decision (ScienceDirect "Insights" tab).
- Exact 2025 JIFs (Clarivate JCR; the figures above are from publisher pages and the wos-journal.info mirror; JISA shows 4.4 vs 3.7 across mirrors).
- Whether FAU has a Springer Nature or Elsevier OA agreement (FAU is confirmed absent from ACM Open).
- IJIS fully-OA transition date (headline confirmed on the journal page).

## Sources

- IJIS: https://link.springer.com/journal/10207 · https://link.springer.com/journal/10207/how-to-publish-with-us · https://link.springer.com/journal/10207/aims-and-scope · https://www.letpub.com/journal-selector/journal/3782 · https://wos-journal.info/journalid/17960
- JISA: https://shop.elsevier.com/journals/journal-of-information-security-and-applications/2214-2126 · https://www.letpub.com/index.php?journalid=10762&page=journalapp&view=detail · https://wos-journal.info/journalid/13316 · https://www.myhuiban.com/journal/393?lang=en-us
- IEEE Access: https://ieeeaccess.ieee.org/about/ · https://ieeeaccess.ieee.org/about/rapid-peer-review/ · https://ieeeaccess.ieee.org/about/article-processing-charge/ · https://wos-journal.info/journalid/15430
- Cybersecurity (Springer): https://link.springer.com/journal/42400 · https://cybersecurity.springeropen.com/submission-guidelines/fees-and-funding · https://wos-journal.info/journalid/12364 · https://www.myhuiban.com/journal/1220?lang=en-us
- DTRAP: https://dtrap.acm.org/ · https://wos-journal.info/journalid/24439 · https://libraries.acm.org/acmopen/open-participants · https://sigdoc.acm.org/blog/2025/10/31/acm-transition-to-open-access/
- CEE: https://www.letpub.com/journal-selector/journal/2004 · https://wos-journal.info/journalid/14185 — Array: https://www.letpub.com/journal-selector/journal/31705 · https://wos-journal.info/journalid/23928
- JCP (MDPI): https://www.mdpi.com/journal/jcp · https://www.mdpi.com/journal/jcp/apc · https://wos-journal.info/journalid/25243
- ESWA: https://manusights.com/blog/expert-systems-with-applications-review-time — KBS: https://manusights.com/blog/knowledge-based-systems-submission-guide — TDSC: https://wos-journal.info/journalid/2761 — JCS: https://wos-journal.info/journalid/8877 — Security and Privacy: https://wos-journal.info/journalid/10616 — KeAi CSA: https://www.keaipublishing.com/en/journals/cyber-security-and-applications/
- Scope-fit papers (Crossref): IJIS 10.1007/s10207-025-01004-4, 10.1007/s10207-025-01033-z, 10.1007/s10207-025-01018-y; Cybersecurity 10.1186/s42400-025-00458-2, 10.1186/s42400-026-00597-0; JISA 10.1016/j.jisa.2025.104162, 10.1016/j.jisa.2025.104037; DTRAP 10.1145/3748264, 10.1145/3822596; IEEE Access 10.1109/access.2026.3658308
