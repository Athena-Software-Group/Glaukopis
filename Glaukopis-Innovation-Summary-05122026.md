---
title: "Glaukopis: Knowledge-Grounded SFT for Cyber Threat Intelligence"
subtitle: "Sovereign, verifiable CTI intelligence at a fraction of frontier cost"
author: "Athena Labs — Division of Athena Security Group"
date: "May 12, 2026"
classification: "Proprietary & Confidential. All rights reserved. Copyright 2026."
---

# Slide 1 — Title

**GLAUKOPIS**
Knowledge-Grounded Supervised Fine-Tuning for Cyber Threat Intelligence

*"Bridging logos (measurement, math, method) and mission (operational outcomes for defenders)."*

DIVISION OF ATHENA SECURITY GROUP | www.athenasecuritygroup.ai
Athena Labs Research — Glaukopis Program Update | 05.12.2026

---

# Slide 2 — Why this matters now

## The frontier-API tax is a systemic risk for defenders

| Pressure                                    | Reality                                                                                                                                                                          |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Token costs are climbing, not falling       | GPT-5.5-Pro lists at **$30 in / $180 out per 1M tokens** (6× and 18× GPT-5 respectively). Gemini-3.1-Pro at $2/$12. Each new generation widens the gap.                          |
| CTI sweeps are token-heavy by construction  | A single AthenaBench pass = ~6.6K rows × ~3K avg-output-tokens ≈ ~20M output tokens. On GPT-5.5-Pro that is **$3.6K per sweep**; production triage usage is orders larger.       |
| Data sovereignty is non-negotiable for SOCs | Threat reports, telemetry, IR notes, host artifacts — none of this can leave the customer's perimeter. Hosted frontier APIs are structurally incompatible with air-gap mandates. |
| Frontier models are CTI generalists         | They were not trained on CTI standards. They guess at MITRE T-codes, hallucinate CVE→CWE mappings, and emit free-text where STIX/JSON is required.                               |

> **Athena Labs thesis:** the right unit of compute for production CTI is a **domain-trained Small Language Model (SLM, 8B–32B)** delivered behind the customer's perimeter, measured continuously against a CTI benchmark, and constrained by deterministic verifiers.

---

# Slide 3 — The Athena Labs CTI stack

A three-pillar architecture engineered to deliver **accurate, structured, verifiable** CTI outputs into production SOCs.

| Pillar         | Component       | Function                                                                                                                                            |
| -------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Measure**    | **AthenaBench** | Six-task dynamic CTI benchmark (CKT, RCM, ATE, TAA, RMS, VSP) with live API connectors to MITRE ATT&CK, NVD, CWE/CAPEC, EPSS. Ground truth is *current*, not stale. |
| **Train**      | **Glaukopis (Sophia)** | Knowledge-graph-driven Instruction Fine-Tuning (IFT). Converts curated CTI/OSINT corpora and the Ariadne knowledge graph into structured supervised examples. |
| **Verify**    | **Minerva (RLVR)** | Reinforcement Learning with Verifiable Rewards. Programmatic verifiers replace opaque preference models, enforcing schema adherence and factual grounding. |

Together these feed **Pallas AI** (on-prem SOC analyst, Qwen2.5-14B at the core) and **Promachos** (transformer-based PIDS) — the production endpoints where the research lands.

---

# Slide 4 — What is Glaukopis (technically)?

A reproducible, instrumented SFT pipeline that turns a structured CTI knowledge graph into a domain-aligned LLM.

**Inputs**
- **Ariadne knowledge graph** — Threat actors, TTPs, tools, CVEs, CWEs, CAPECs, mitigations, with explicit *uses / targets / exploits / mitigated_by* edges.
- **Curated CTI corpora** — MITRE ATT&CK, CWE, CAPEC, CISA KEV/ICS/CSA, NVD descriptions, OSINT feeds, vendor advisories.
- **Sophia template engine** — A grammar that walks the graph and emits `{instruction, input, output}` tuples with strict schema constraints.

**Outputs**
- A fine-tuned base model (Qwen2.5-14B-Instruct today; Llama-3.1-8B is the historical anchor) that:
  - Emits valid JSON / STIX 2.1 by construction.
  - Cites canonical identifiers (T-codes, CVE/CWE/CAPEC IDs) instead of paraphrasing them.
  - Performs explicit chain-of-thought with verifiable intermediate steps.
- Published versioned checkpoints on Hugging Face under `asg-ai/athena-cti-sft-*`.

**Why a knowledge graph is the right substrate**
Free-text fine-tuning lets the model learn surface forms (`"PowerShell"` ≈ `"powershell.exe"`); graph-derived templates teach *relations* (`APT29 — uses → T1059.001 — mitigated_by → M1038`). Relations transfer across unseen incidents; surface forms do not.


---

# Slide 5 — The v18.1 SFT recipe (current, in flight)

We've moved through 18 SFT vintages on the Llama-3.1-8B (v0–v8) and Qwen2.5-14B (v9–v18.1) lineages. v18.1 is the corrective "Core-only redo" that addresses the v18 cumulative-training regression by tightening the curriculum and isolating recovery axes.

**Two-phase shape (Qwen2.5-14B-Instruct, full SFT, ZeRO-3, Liger kernel)**

| Phase | Cutoff | Packing | LR    | Effective batch | Composition                                                                 | Goal                                                |
| ----- | ------ | ------- | ----- | --------------- | --------------------------------------------------------------------------- | --------------------------------------------------- |
| **A** | 8,192  | on      | 1e-5  | 16              | Knowledge-grounded breadth: KB / MCQ / TAA / SOC / CM / MS / YN + Tulu / Alpaca anchor | Re-anchor general instruction-following; lift CKT.  |
| **B** | 16,384 | off     | 5e-6  | 16              | AthenaBench-axis drill: RMS / ATE / VSP / RCM at full context              | Recover historical task-axis peaks without forgetting Phase A. |

**Recovery floors (Glaukopis v18.1 acceptance criteria)**
- MCQ ≥ 75.0 (target: recover 8B-era 77.6 peak on the larger base)
- RMS ≥ 64.0 strict-F1 (target: recover Llama v7's 65.8)
- VSP ≥ 84.0 (target: recover historical 86.7)
- ATE / RCM / SOC / CM ≥ v18-core − 2pp (no-regression guards)

**Hardware footprint** — 4× H100 80GB, ~13h wallclock with `A_BATCH=2` (the launcher default committed in `c1e65bc`). On 8× H100 the same recipe runs in ~6.5h.

---

# Slide 6 — How the Sophia template engine works

Glaukopis training data is **synthesized**, not scraped. The Sophia template engine deterministically walks the Ariadne knowledge graph and emits training examples that are guaranteed to be schema-valid and benchmark-decontaminated.

**Pipeline**
1. **Walk** — Pick a node (e.g., a CWE) and traverse outward N hops along whitelisted relations.
2. **Render** — Apply a template family (`MCQ.*`, `RMS.*`, `ATE.*`, `TAA.*`, `KB.*`, `MS.*`, `YN.*`) to the subgraph, producing a tuple with a Sophia ID.
3. **Decontaminate** — `dedup_against_evals.py` indexes every AthenaBench prompt at n=13 and rejects any training row that overlaps. *Catalog overlap is allowed; prompt overlap is not.*
4. **Balance** — Stratify by template family and target schema before shuffling.
5. **Emit** — Write `ift_data_<date>_<vintage>_<phase>.json` shards keyed by phase.

**Why this matters for evaluators**
- Reproducibility: every training row carries a Sophia ID traceable to a graph walk and a template version.
- Auditability: regressions get root-caused to specific template families (e.g., the v15 W1 audit fingered TAA.CANON as the wrong TAA flavor for AthenaBench TAA Classic, which we then retired).
- Schema integrity: outputs are valid by construction, not by post-hoc filtering.


---

# Slide 7 — AthenaBench: the measurement substrate

If you cannot measure, you cannot improve. AthenaBench is the dynamic CTI benchmark we use to gate every Glaukopis vintage and to compare against frontier models on the same axes.

| Task   | What it tests                                                       | Why it's hard for generalists                                    |
| ------ | ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **CKT** (MCQ) | Baseline CTI domain literacy (concepts, frameworks, terminology) | Requires recall of canonical definitions, not paraphrase.        |
| **RCM** | Root-cause mapping — symptom → CVE / CWE                            | CVE descriptions are short and ambiguous; the mapping is exact. |
| **ATE** | ATT&CK technique extraction — prose → T-codes                       | Demands precise mapping to MITRE's canonical T-code set.        |
| **TAA** | Threat-actor attribution from TTPs and IOCs                         | Requires actor-level pattern recognition, not paraphrase.       |
| **RMS** | Risk-mitigation strategies (NIST/CIS-aligned) at the right cardinality | Free-text generalist outputs miss schema and N=1..8 cardinality. |
| **VSP** | CVSS severity prediction from vulnerability descriptions            | Numeric reasoning over a domain-specific vector.                |

Connectors: live MITRE ATT&CK, NVD/CVE, CWE/CAPEC, EPSS — so the benchmark stays *current* (a static benchmark goes stale within a quarter in CTI).

Combined Score = average across the six tasks; per-task floors govern Glaukopis sign-off.

---

# Slide 8 — Where the field stands today (AthenaBench, Q1–Q2 2026)

| Rank | Model                              | Class       | Combined | RMS  | VSP  | Notes                                                        |
| ---- | ---------------------------------- | ----------- | -------: | ---: | ---: | ------------------------------------------------------------ |
| 1    | **Gemini-3-Pro**                   | Frontier    |  **69.7** | 43.1 | 90.7 | Current SOTA on combined score.                              |
| 2    | **GPT-5.2 (high reasoning)**       | Frontier    |    67.1   | 35.6 | 86.1 | High output-token cost; reasoning tax.                       |
| —    | DeepSeek-V4-Pro (HF Inference)     | Frontier OS |  ~88.8 *MCQ partial sweep* | n/a | n/a | Strongest open-weights frontier on MCQ probe; full sweep deferred. |
| 6    | GPT-4o                             | Frontier    |    58.0   | 20.2 | 84.7 | Reference for "yesterday's" frontier.                        |
| 7    | **Minerva-Llama-8B (Athena Labs)** | **SLM + RLVR** |  **56.3** | **41.2** | **87.6** | 8B params; **2nd best RMS, 3rd best VSP** in the field.      |
| —    | Foundation-sec-8B-Reasoning        | OS SLM      |    53.75  | n/a  | n/a  | Best non-Athena open-source baseline.                         |
| 8    | Gemini-2.5-Flash                   | Frontier    |    54.0   | 13.4 | 78.5 |                                                              |
| 10   | GPT-4                              | Frontier    |    51.4   | 15.1 | 84.7 |                                                              |

**Headline:** an Athena-trained 8B model already places 7th overall while outperforming GPT-4 and Gemini-2.5-Flash on the two highest-stakes operational axes (RMS, VSP). The 14B and 32B Glaukopis models target the gap between #7 and the frontier top-2.

---

# Slide 9 — The frontier-API tax, quantified

Cost per **single AthenaBench sweep** (~6.6K rows, ~3K avg output tokens ≈ ~20M output tokens, ~10M input tokens — real-world ranges from prior sweeps), using `SFT/test/pipelines/api_usage.py` rate cards:

| Model                    | Input $/1M | Output $/1M | Est. cost / AthenaBench sweep | Multiple vs. self-hosted Qwen2.5-14B |
| ------------------------ | ---------: | ----------: | ----------------------------: | -----------------------------------: |
| GPT-5.5-Pro              |      30.00 |      180.00 |                **~$3,900**    | **~1,950×** |
| GPT-5.5                  |       5.00 |       30.00 |                ~$650          | ~325×       |
| Gemini-3.1-Pro           |       2.00 |       12.00 |                ~$260          | ~130×       |
| GPT-5.2 (high reasoning) |       1.75 |       14.00 |                ~$300          | ~150×       |
| Gemini-3-Flash           |       0.50 |        3.00 |                ~$65           | ~32×        |
| DeepSeek-V4-Pro (HF)     |       1.74 |        3.48 |                ~$87           | ~43×        |
| DeepSeek-V3.2-Exp (HF)   |       0.27 |        0.40 |                ~$11           | ~5.5×       |
| Qwen2.5-14B (HF Router)  |       0.20 |        0.20 |                ~$6            | ~3×         |
| **Self-hosted Glaukopis-Qwen2.5-14B** | — | — | **~$2** (amortized GPU-hour, on-prem) | **1× (baseline)** |

Two points the table makes loudly:
1. **Frontier sweep costs are not a one-time research expense.** Multiply by *every* analyst query in production and the curve becomes prohibitive.
2. **The cheapest route is also the most sovereign.** The on-prem model has no per-token bill, no egress, no third-party dependency.

---

# Slide 10 — Sovereign by design

The **operational** advantages of a self-hosted SLM are at least as decisive as the cost advantages.

| Axis                | Hosted frontier API                                  | Self-hosted Glaukopis SLM (8B / 14B / 32B)               |
| ------------------- | ---------------------------------------------------- | -------------------------------------------------------- |
| **Data sovereignty** | All prompts and completions traverse third-party infrastructure. | Zero egress. Threat data, IR notes, host artifacts never leave the customer perimeter. |
| **Air-gap compatible** | No.                                                  | **Yes** — runs in fully air-gapped enclaves with Ollama / vLLM. |
| **Compliance posture** | Requires DPA, SOC2 attestation chain, data-residency carve-outs per region. | Tenant retains full custody. FedRAMP / IL5-style envelopes are achievable. |
| **Latency floor**   | Network RTT + provider queue.                        | Local PCIe — millisecond-class first-token latency.      |
| **Cost model**      | Per-token, asymptotically unbounded.                 | Fixed CapEx (or amortized OpEx for hosted GPU); marginal cost ≈ electricity. |
| **Domain alignment** | Generalist; CTI is one of thousands of long-tail domains. | Trained on CTI standards; emits canonical IDs and valid schemas by construction. |
| **Auditability**    | Black-box; no insight into reasoning trace.          | Open weights, open templates, open verifiers. Every output traceable to a graph walk + RL trajectory. |

This is why Pallas (Athena's SOC analyst product) embeds Qwen2.5 on-premise rather than calling a hosted API: **defenders cannot afford to ship their telemetry to vendors who can be subpoenaed, breached, or repriced.**


---

# Slide 11 — Glaukopis SFT trajectory: what we've learned across 18 vintages

The lineage is dense; the lessons are stable.

| Vintage          | Base                  | Best result                                     | Lesson learned                                                              |
| ---------------- | --------------------- | ----------------------------------------------- | --------------------------------------------------------------------------- |
| **v0–v3 (Llama)** | Llama-3.1-8B-Instruct | Baseline; AthenaBench RMS ≈ 6%                  | Out-of-the-box generalists can't hold CTI schema or cardinality.            |
| **v6 (Llama)**    | Llama-3.1-8B-Instruct | RMS regression to 0%                            | Naive multi-task SFT *destroys* recovered axes if RMS isn't anchored explicitly. |
| **v7 (Llama)** ⭐  | Llama-3.1-8B-Instruct | **RMS 62.6 strict-F1, MCQ 57.6, VSP 75.0**      | The RMS.3a..3h cardinality-stratified template family is the unlock.        |
| **v12 (Qwen2.5-14B)** | Qwen2.5-14B-Instruct | MCQ 70.4, ATE 55.1                          | Qwen2.5 base lifts knowledge axes by ~13pp without lifting catalog axes.    |
| **v17.1**         | Qwen2.5-14B-Instruct | MCQ 70.0, ATE 56.6                              | The CSE additions held; CKT/ATE refused to break the ceiling.               |
| **v18 cumulative** | Qwen2.5-14B-Instruct | Regressed across MCQ/RMS/VSP                   | Adding TAA + CSE on top of Core in one cumulative pass diluted the recovery axes. |
| **v18.1 (in flight)** | Qwen2.5-14B-Instruct | *target*: MCQ ≥ 75 / RMS ≥ 64 / VSP ≥ 84 | Two-phase corrective: re-anchor breadth in Phase A, drill catalog axes in Phase B. |

Two findings that generalize:
- **Curriculum order matters more than corpus size.** Phase A (knowledge) before Phase B (catalog) consistently beats a single-shot mixed pass.
- **Schema-stratified templates outperform free-text augmentation** by a wide margin on the operational axes (RMS, ATE, VSP).

---

# Slide 12 — Closing the gap to the frontier with 32B SLMs

Glaukopis-Qwen2.5-14B is the current production target. But the *next* base candidates — Qwen2.5-32B-Instruct and Gemma-32B (or its successor) — sit in a sweet spot that puts the frontier-2 within reach.

| Model class              | Params | Single-H100 inference | AthenaBench combined (projected, post-Glaukopis SFT) | Distance to Gemini-3-Pro (69.7) |
| ------------------------ | -----: | --------------------- | ---------------------------------------------------: | ------------------------------: |
| Llama-3.1-8B-Instruct (raw) | 8B   | yes (PCIe)            | ~30 (v0 baseline)                                   | -39.7                           |
| Glaukopis-Llama-8B v7    |    8B  | yes                   | ~50 (v7-class)                                      | -19.7                           |
| Minerva-Llama-8B (RLVR layered on v7-class SFT) | 8B | yes        | **56.3 (measured)**                                 | **-13.4**                       |
| Glaukopis-Qwen2.5-14B v18.1 (in flight) | 14B | yes (single H100) | **~62–65 (projected)**                              | **~-5 to -8**                   |
| Glaukopis-Qwen2.5-14B + Minerva (planned) | 14B | yes               | **~67–70 (projected)**                              | **~-2 to +1**                   |
| **Glaukopis-Qwen2.5-32B + Minerva (roadmap)** | 32B | 2× H100 | **~70–73 (projected)**                              | **+1 to +4 (parity / past)**    |

**Why we believe this trajectory:**
- The Minerva paper measures **+15.8 pp mean uplift** over base models and **+4.3 pp** over GRPO across four backbones and 12 CTI benchmarks.
- The 8B → 14B base step has historically delivered ~8–12 pp on knowledge axes in our chain.
- The 14B → 32B step is empirically smaller (~3–5 pp) but predictable, and Minerva's RLVR phase typically extracts the most uplift on RMS / ATE — the hardest axes for the larger bases.

The plausible end-state is a **32B Glaukopis + Minerva model that meets or beats Gemini-3-Pro on AthenaBench combined** while running fully on-premise on a single 2-H100 node.

---

# Slide 13 — Minerva: RLVR primer

After Glaukopis SFT lands schema and breadth, **Minerva** (Reinforcement Learning with Verifiable Rewards) hardens factuality and structure. From the Minerva v3 paper (arXiv:2602.00513, Alam, Piplai, Cardei, Rastogi, Worth Jr.):

> *"Averaged across four backbones and 12 CTI benchmarks, MinervaRL improves the mean score by 15.8 percentage points over the corresponding base models and by 4.3 points over GRPO."*

**Why RLVR (not RLHF) for CTI**
- CTI standards (MITRE ATT&CK, CVE, CWE, CAPEC, EPSS, STIX) define **deterministic** identifiers and schemas.
- That structure means every model output can be **programmatically verified** (does the T-code exist? is the JSON valid? does the CWE actually map to the CVE?) — no learned reward model is needed.
- Programmatic verifiers are auditable, air-gap compatible, and immune to reward hacking.

**The Minerva loop**
1. **Generate** — Model produces candidate completions with reasoning traces.
2. **Verify** — Deterministic verifiers score each output: schema validity, identifier grounding, citation existence, logical consistency. Reward ∈ [0, 1] (binary or graded).
3. **Update** — GRPO (critic-free, PPO-style) updates the policy *only* when verification passes.

**MinervaRL extensions** (v3 paper)
- **Answer-Conditioned Reasoning (ACR)** — uses gold labels during training to elicit concise, grounded reasoning traces for hard examples where standard rollouts fail (addresses reward sparsity).
- **Deferred generation & filtering** — an EMA teacher generates candidates that undergo heuristic + ML-based filtering before they enter the RL buffer.
- **Self-training distillation** — accepted high-quality traces are distilled back to the original answer-free prompts via lightweight SFT, closing the loop.


---

# Slide 14 — Glaukopis + Minerva: composed uplift

The two systems are designed to **compose**, not compete.

```
Base model (Qwen2.5-14B-Instruct)
        │
        ▼  +Glaukopis SFT (knowledge + schema)
Glaukopis-CTI-LLM   ──── teaches the model what CTI is
        │
        ▼  +Minerva RLVR (verifiable rewards)
Athena-CTI-LLM      ──── teaches the model when it's right
        │
        ▼ Deployment (Pallas / Promachos / Athena Core)
SOC-grade analyst   ──── verifiable, sovereign, on-prem
```

**Why composition wins**
- **Glaukopis** alone improves CTI literacy and schema adherence but cannot self-correct factual drift.
- **Minerva** alone is reward-sparse without a domain-aligned base — RLVR rollouts mostly fail verification on a generalist model, which collapses signal.
- **Together** the SFT phase produces a base whose rollouts pass verification often enough to make RLVR sample-efficient. The published Minerva-Llama-8B (#7 on AthenaBench, 2nd-best RMS in the field) is the empirical proof point.

**Operational reading**
- Glaukopis is the **breadth** vector — every CTI task gets baseline coverage.
- Minerva is the **depth** vector — the highest-stakes tasks (RMS, VSP, ATE) get verified outputs that a SOC can act on without human re-validation.

---

# Slide 15 — Where this lands in product

| Athena product       | Glaukopis / Minerva role                                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Pallas AI**        | The Glaukopis-tuned Qwen2.5 model is the on-prem reasoning core. 73 investigation strategies route to Glaukopis whenever the regex / keyword tiers fail. |
| **Promachos (PIDS)** | Consumes Glaukopis attack-narrative outputs to seed transformer-based provenance anomaly detection.                     |
| **Athena Core / XDR**| Receives verified mitigations (RMS) and severity scores (VSP) from Minerva-verified outputs to drive automated playbooks. |
| **AthenaBench**      | Continuously gates every Glaukopis vintage and every Minerva run; results published openly.                             |

Closed-loop pipeline (per the Athena Labs unabridged update, p.18):

```
AthenaBench ──► Glaukopis & Minerva ──► Pallas / Promachos ──► Athena Core / XDR
   ▲                                                                    │
   └────────── telemetry, edge cases, regression tests ◄────────────────┘
```

This is what "research that ships" looks like in CTI: every measurement feeds a training loop, every checkpoint feeds a product, every deployment feeds new measurement data.

---

# Slide 16 — Innovation & impact (recap)

**What's novel**
- **First domain-trained CTI LLM with RLVR.** Programmatic verifiers replace generic RLHF, ensuring factual correctness and schema adherence rather than aesthetic preference.
- **Knowledge-graph-derived SFT corpus.** Sophia templates produce reproducible, schema-valid training data with built-in benchmark decontamination.
- **Two-phase corrective curriculum.** v18.1's Phase A → Phase B structure addresses the multi-task-SFT regression problem we observed in v18 cumulative.
- **Dynamic benchmark with live API connectors.** AthenaBench refreshes against MITRE ATT&CK / NVD / EPSS continuously — no benchmark gaming, no staleness.
- **Open lineage.** Every checkpoint is published on Hugging Face under `asg-ai/athena-cti-sft-*`. Every recipe is in-repo.

**Operational value (per Athena Labs unabridged update, p.19)**
- **Sovereignty: 100%** — air-gapped, on-prem, zero data egress.
- **Noise reduction: ↓ 60%** fewer false positives vs. traditional SIEM (Pallas measurement).
- **Response time: ↓ MTTR** via automated triage and playbook execution.
- **Throughput: ↑ 3×** analyst capacity via AI-assisted investigation.

**Cost framing**
- A frontier-API-driven SOC pays the **token tax** on every analyst query, every day, in perpetuity.
- A Glaukopis-driven SOC pays a **fixed CapEx** for the GPU node and amortizes it across unlimited queries with zero egress.
- The breakeven for a 100-analyst SOC vs. GPT-5.5-Pro routing happens in **weeks**, not quarters.

---

# Slide 17 — Roadmap (next 6 months)

| Quarter      | Milestone                                                                                  | Owner                |
| ------------ | ------------------------------------------------------------------------------------------ | -------------------- |
| **Q2 2026 (now)** | Land **v18.1** Core (in flight, ~13h on 4× H100). Sign off against MCQ/RMS/VSP floors. | Glaukopis core       |
| Q2 2026      | Append **v18.1 + TAA** and **v18.1 + CSE** stages on top of validated v18.1 Core.           | Glaukopis chain      |
| Q2 2026      | Re-baseline the frontier comparator panel (Gemini-3.1-Pro, GPT-5.5-Pro, DeepSeek-V4-Pro) with per-token cost reporting now in place. | Bench               |
| Q3 2026      | **Minerva-Qwen-14B** — apply RLVR over the v18.1-final Glaukopis checkpoint.                | Minerva team         |
| Q3 2026      | Begin **Glaukopis-Qwen2.5-32B** Core training; same two-phase recipe, scaled.               | Glaukopis core       |
| Q4 2026      | **Minerva-Qwen-32B** — full Athena-CTI-LLM-32B target; AthenaBench parity push.             | Minerva team         |
| Q4 2026      | **Athena-CTI-LLM** manuscript submission; Promachos / PIDS manuscript submission.           | Research leadership  |

---

# Slide 18 — Closing

**Athena Labs is building the production SLM stack the CTI community actually needs.**

- **Measure** with AthenaBench.
- **Train** with Glaukopis (knowledge-grounded SFT).
- **Verify** with Minerva (RLVR).
- **Ship** in Pallas, Promachos, Athena Core.

The frontier-API tax, data-egress risk, and air-gap requirements that block hosted models from production SOCs are exactly the constraints a domain-trained SLM is built to satisfy. The trajectory from Llama-3.1-8B (v0) to Glaukopis-Qwen2.5-32B + Minerva (roadmap) is **reproducible, instrumented, and within reach** on the hardware our customers already own.

*"Research that ships. Measurable. Verifiable. Operational."*

---

**Athena Labs** | DIVISION OF ATHENA SECURITY GROUP
www.athenasecuritygroup.ai | https://athenasecuritygroup.ai/athena-labs/
labs@athenasecuritygrp.com

**Academic partnerships:** Rochester Institute of Technology (RIT) | Florida Atlantic University (FAU)

**Research leadership:** Peter J. Worth, Jr. (President & CEO) | Dr. Ionut Cardei (Chief Scientist)

**Key references**
- Alam, M.T., Piplai, A., Cardei, I., Rastogi, N., Worth Jr., P. — *Minerva: Reinforcement Learning with Verifiable Rewards for Cyber Threat Intelligence LLMs.* arXiv:2602.00513 (v3, May 2026).
- Alam et al. — *AthenaBench: Dynamic CTI Benchmark for LLMs (with New Risk Mitigation Task).* arXiv:2511.01144 (2025).
- Worth Jr., P. & Cardei, I. — *Layer Cake: Language Representation & Compute.* International Journal of Intelligence Science (2025).
- Worth, P.J. — *Word Embeddings & Semantic Spaces in NLP.* International Journal of Intelligence Science (2023).

*Proprietary & Confidential. All rights reserved. Copyright 2026.*
