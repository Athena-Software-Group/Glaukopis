# Sophia CTI Templates — v18.1 (May 11, 2026 vintage)

v18.1 is the **Core-only redo** of the v18 chain Stage 1 after v18 regressed
three historical AthenaBench peaks. It mirrors the v18-Core two-phase
training recipe byte-for-byte (Phase A broad re-anchor + Phase B catalog
drill, 8xH100, ~13 h wallclock) and changes only the Core corpus. v18 Stage
2 (TAA Classic refresher) and Stage 3 (CSE drill) are NOT re-trained — the
existing v18 chained checkpoints are renamed on HF to reflect their
domain-specific (not cumulative) nature and re-chained off the v18.1-Core
base.

The branch produces one new HF checkpoint and (eventually, on §5 sign-off)
one re-chained final:

| checkpoint | answers the question |
|---|---|
| `asg-ai/athena-cti-sft-qwen25-14b-v18-1-core` | When the v18 Core MCQ shard is purged of the AB.MCQ.EXT.{MITRE,SEC,GLOSS}.1 KB-flashcard families and rebuilt to the v8small scenario-only recipe (with three substrate-bound MCQ templates added to clear the 6,000-row floor), the AB.RMS.{4,5} ten-paraphrase fragmentation is consolidated back to the v9_rms single-template-per-direction shape, and the AB.VSP.{1..4}+V.CPE.{1..4} families are explicitly capped at v10's 12K shape — does CKT lift to ≥75 (target 77.6 = v8small), RMS to ≥64 (target 65.8 = v9_rms), and VSP to ≥84 (target 86.7 = v10) without regressing ATE/RCM/SOC/CM/MS/TAA against the v18-Core baseline? |
| `asg-ai/athena-cti-sft-qwen25-14b-v18-1` *(only on §5 sign-off)* | Does re-chaining the existing v18-taa + v18-cse stage refreshers off the new v18.1-Core base preserve CSE-TI / CSE-Malware / TAA-attribution within 2 pp of the v18 chained model? |

The vintage directory is self-contained per project convention; only the
v18.1 Core manifest lives here. Stages 2 and 3 reuse the v16 (`05092026/`)
and v17.1 (`05102026/`) manifests verbatim — same as the original v18 chain.

```
05112026/
  Sophia-CTI-Templates-v18.1.txt   v18.1 Core manifest. Body identical to
                                   Sophia-CTI-Templates-v18.txt except:
                                     - AB.MCQ.EXT.{MITRE,SEC,GLOSS}.1 dropped
                                     - AB.RMS.{4a..4j,5a..5j} consolidated
                                       back to AB.RMS.4 / AB.RMS.5 (Count 600)
                                     - AB.VSP.{1..4} + V.CPE.{1..4} explicit
                                       Count: 1500 (12K cap, v10 shape)
                                     - AB.MCQ.{7,8,9} appended (3 new
                                       scenario-shape templates against
                                       higher-cardinality MITRE substrates)
  v18_1_plan.txt                   master plan (RCA in §1.2; deltas in §2;
                                   row-count plan in §3; recipe in §4;
                                   falsification in §5)
  v18_1_row_count_gate.json        per-axis REJECT_IF_BELOW thresholds for
                                   the v18.1 Core shard. MCQ floor 8100→6000
                                   (matches the v8 scenario-only ~6K shape);
                                   RMS 11700→12000; VSP 8100→5400 (anchored
                                   on v10's ~12K total). Other axes carry
                                   v18 floors verbatim.
  README-18-1.md                   this document
  README.md                        v18 chain documentation (the predecessor)
  Sophia-CTI-Templates-v18.txt     v18 Core manifest (frozen; predecessor)
  v18_plan.txt                     v18 chain plan (frozen; predecessor)
  v18_row_count_gate.json          v18 Core build-time floors (frozen)
```

Local build artefact dir is `_v18p1_build/` (forked from `_v18_build/`).

## 1. Why v18.1 exists — the v18 Core regression

The v18 chain landed CSE-TI / CSE-Malware and TAA-attribution on target via
Stages 2 and 3. The chained pattern itself is validated. But Stage 1
(Core, `asg-ai/athena-cti-sft-qwen25-14b-v18-core`) regressed against three
historical AthenaBench peaks:

| axis | v18-core | historical peak | delta |
|---|---|---|---|
| CKT (MCQ) | 62.6 | v8small  77.6 (LLaMA 8B) | **−15.0 pp** |
| RMS       | 55.6 | v9_rms   65.8 (Qwen 14B) | **−10.2 pp** |
| VSP       | 76.8 | v10      86.7 (Qwen 14B) | **−9.9 pp**  |

Diagnosis (from inspection of `ift_data_2026_05_13_v18_core_*.json` and the
v18 manifest, see request_id `ff1b9f2f-7d19-4f8d-8300-1b28bd6f43cd`):

- **MCQ.** The v18 Core MCQ shard was 61% AB.MCQ.EXT.{MITRE,SEC,GLOSS}.1 KB
  flashcards (5,478 of 8,949 rows). The eval shape — five-option scenario
  MCQ — got 39% of the gradient. v8small ran with ~6K scenario MCQ rows
  and no EXT family and posted CKT 77.6.
- **RMS.** v18 fragmented AB.RMS.{4,5} into ten paraphrases each at
  Count: 50 per paraphrase. The catalog-recall capability that v9_rms got
  from a single 600-row template per direction was diluted across 10× of
  template-string variation that never appears in the eval set.
- **VSP.** `count_max=3500` with no explicit AB.VSP / V.CPE Counts produced
  ~27K VSP rows — 2.25× v10's ~12K. Template exhaustion on a fixed 8
  phrasings drove phrasing-overfit (v18 plateaus on the train shape and
  regresses on the eval shape).

v18.1 is therefore CORPUS-DRIVEN, not a recipe change. The Stage 1
training recipe is the v18-Core two-phase recipe verbatim. The only
deltas are in the manifest (`Sophia-CTI-Templates-v18.1.txt`) and the
row-count gate (`v18_1_row_count_gate.json`).

## 2. v18 → v18.1 deltas (manifest only)

1. **MCQ recipe (v8small).** Drop AB.MCQ.EXT.{MITRE,SEC,GLOSS}.1.
   AB.MCQ.{1..6} and JS.MCQ.{1,2,5} retain v18 Counts (the v8 ~6K scenario
   budget). Watcher Phase 3c (mcq_generator.py + GLOSS merge) is bypassed
   in `_v18p1_build/watcher.sh`.

2. **MCQ substrate patch.** The first v18.1 build yielded only 4,762 MCQ
   rows (below the 6,000 floor) because AB.MCQ.{2,4,6} are bound by the
   CAPEC.map_ap edge cardinality. To clear the floor without re-introducing
   flashcard noise, three new scenario-shape MCQ templates were added
   against higher-cardinality MITRE ATT&CK substrates:

   | template | edge | substrate cardinality | Count |
   |---|---|---|---|
   | `AB.MCQ.7` | malware -[:uses]-> attack-pattern        | 9,836 edges | 800 |
   | `AB.MCQ.8` | intrusion-set -[:uses]-> malware         |   647 edges | 500 |
   | `AB.MCQ.9` | attack-pattern -[:subtechnique-of]-> ap  |   477 edges | 800 |

   All three use four `{negack*}/{negmw*}` distractors with the v18
   force-distinct constraint pattern. AB.MCQ.9 is substrate-bound at ~445
   rows by its 477-edge ceiling; AB.MCQ.8 lands at ~395 rows; AB.MCQ.7
   reaches ~795 rows. Combined +1,639 rows → MCQ axis post-dedup = 6,401
   (margin +401 over the 6,000 floor).

3. **RMS recipe (v9_rms).** Roll AB.RMS.{4a..4j} into a single AB.RMS.4
   `Count: 600` (ID → name + description). Roll AB.RMS.{5a..5j} into a
   single AB.RMS.5 `Count: 600` (name → ID + description). Same total
   catalog-drill volume as v18, same template strings as v9_rms.

4. **VSP recipe (v10).** Explicit `Count: 1500` added to AB.VSP.{1..4}
   and V.CPE.{1..4} so the build caps at v10's `4×1500 + 4×1500 = 12K`
   shape regardless of `count_max`. The substrate (CVE descriptions +
   cpe_matches) is the same as v10's; only the sampling cap changes.

5. **Row-count gate floors** (`v18_1_row_count_gate.json`):
   - MCQ 8100 → 6000 (EXT family removed; floor matches v8 scenario shape)
   - RMS 11700 → 12000 (anchor on v9_rms shape with paraphrases consolidated)
   - VSP 8100 → 5400 (anchor on v10's ~12K total less eval-overlap dedup tail)
   - all other axes carry v18 floors verbatim.

6. **Stage 2 (TAA) and Stage 3 (CSE) are NOT re-trained.** The existing v18
   chained checkpoints are renamed on HF to reflect domain-specific (not
   cumulative) names:
   - `asg-ai/athena-cti-sft-qwen25-14b-v18-core-plus-taa`     → `…-v18-taa`
   - `asg-ai/athena-cti-sft-qwen25-14b-v18-core-plus-taa-cse` → `…-v18-cse`

   The vLLM aliases in `SFT/test/pipelines/models.py` track the new names.
   On §5 sign-off the v18-taa / v18-cse stages are re-chained off the new
   v18.1-Core base; the resulting chained model is published as
   `asg-ai/athena-cti-sft-qwen25-14b-v18-1`.

## 3. Build outcome (recorded 2026-05-11)

`_v18p1_build/` Phase-0 substrate gate, generation, dedup, row-count gate,
and stratified shuffle complete; both axis shards plus the val slice are
written to `SFT/data/`:

| shard | rows |
|---|---|
| `ift_data_2026_05_11_v18p1_core_train.json` | 329,850 |
| `ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn.json` *(Phase A)* | 258,403 |
| `ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm.json` *(Phase B)* |  71,447 |
| `ift_data_2026_05_11_v18p1_core_val.json` |     550 |

Row-count gate result (all ten axes pass; see
`tmpl_gen/templates/05112026/v18_1_row_count_gate.json` and
`_v18p1_build/row_count_gate_report.json`):

| axis | actual | floor | margin |
|---|---:|---:|---:|
| **MCQ** | 6,401 | 6,000 | **+401** |
| RMS | 14,158 | 12,000 | +2,158 |
| ATE | 16,950 | 12,500 | +4,450 |
| VSP | 11,738 |  5,400 | +6,338 |
| RCM | 28,799 |  9,000 | +19,799 |
| MS  |  4,301 |  3,600 | +701 |
| TAA-attribution |  3,500 |  3,150 | (cap) |
| TAA-IE-NEG |  5,983 |  5,400 | +583 |
| SOC |  9,923 |  9,000 | +923 |
| CM  |  6,000 |  5,400 | +600 |

Cross-validation (against the four shards listed above):

- MCQ letter balance — AB.MCQ.{1..9}: every correct-answer letter A–E
  lands within 16–24% (no per-template bias); JS.MCQ.{1,2,5} emit JSON
  letter-set payloads, not "Therefore, X." prose, and are exempt.
- AB.MCQ.{7,8,9} per-axis row counts (substrate-bound):
  AB.MCQ.7 = 797, AB.MCQ.8 = 397, AB.MCQ.9 = 445.
- Train/val disjointness — fingerprint hash over (instruction, input,
  output): 0 overlap (|train|=329,270 unique; |val|=550 unique).
- Val coverage — 85 distinct shortnames in the val slice (≈6.5 rows / axis).

## 4. Run-book

### 4.1 Generation (already complete; for reproduction only)

```bash
cd /Users/pietro/code/Glaukopis    # or the cluster equivalent

python _v18p1_build/_neo4j_check.py    # smoke-test (also runs in watcher Phase 0)

mkdir -p _v18p1_build/triples
nohup bash tmpl_gen/data_generation/make_dataset.sh \
     tmpl_gen/templates/05112026/Sophia-CTI-Templates-v18.1.txt \
     _v18p1_build/triples \
     SFT/data/ift_data_2026_05_11_v18p1_core.raw.json \
     2500 1500 > _v18p1_build/build.log 2>&1 &
echo "PID=$!" > _v18p1_build/build.pid
nohup bash _v18p1_build/watcher.sh > _v18p1_build/watcher.log 2>&1 &
```

The watcher polls the build PID, then runs Phase 0..6: substrate gate,
seed-provenance gate, generator merges, actor-balance, dedup, row-count
gate, licence gate, stratified shuffle, val/train split, two-shard phase
split (broad + axis). Final status in `_v18p1_build/watcher_status.json`.

### 4.2 Dataset registration

The launcher resolves dataset names through `SFT/data/dataset_info.json`.
The v18.1 entries are already registered:
- `ift_data_2026_05_11_v18p1_core_a_kb_mcq_taa_soc_cm_ms_yn`
- `ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm`
- `ift_data_2026_05_11_v18p1_core_val`

### 4.3 Training (Stage 1; Core only)

Single launcher; auto-detects GPU count and toggles DeepSpeed CPU offload
(off for ≥4 GPUs, on otherwise) so it runs unchanged on either an 8×H100
or a 4×H100 system:

```bash
# ~13 h on 8xH100 80GB; ~24 h on 4xH100 80GB
bash SFT/autotrain/run_sft_qwen25_14b_v18p1_core.sh
# defaults to ${HF_USERNAME}/athena-cti-sft-qwen25-14b-v18-1-core
# base of stage 1 : Qwen/Qwen2.5-14B-Instruct
```

Phase A runs broad re-anchor (cutoff 8192, packing on, lr 1e-5,
effective batch 16, --max-samples 240000); Phase B runs the
RMS+ATE+VSP+RCM catalog drill (cutoff 16384, packing off, lr 5e-6,
effective batch 8, --max-samples 70000, eval/save every 400 steps).
Only Phase B's final merged model is pushed to HF.

Single-phase reruns (e.g. for debugging Phase A regression):
```bash
bash SFT/autotrain/run_sft_qwen25_14b_v18p1_core.sh --phase a
bash SFT/autotrain/run_sft_qwen25_14b_v18p1_core.sh --phase b \
     --phase-a-dir SFT/saves/Qwen_Qwen2.5-14B-Instruct/full/v18p1_core_phase_a_<TS>
```

### 4.4 Bench

Standard 14B AthenaBench + CyberMetric + CyberSOCEval sweep against
`asg-ai/athena-cti-sft-qwen25-14b-v18-1-core`. Sign-off criteria
(see `v18_1_plan.txt §5`):

- CKT (MCQ) ≥ 75.0  (target 77.6 = v8small)
- RMS       ≥ 64.0  (target 65.8 = v9_rms)
- VSP       ≥ 84.0  (target 86.7 = v10)
- ATE / RCM / SOC / CM ≥ v18-core − 2 pp (no recipe change to those axes)

If §5 passes the v18-taa and v18-cse stages are re-chained off the new
v18.1-Core base (existing launchers `run_sft_qwen25_14b_v18_plus_taa.sh`
and `run_sft_qwen25_14b_v18_final.sh` re-pointed at the v18.1-Core repo)
and the chained final is pushed as
`asg-ai/athena-cti-sft-qwen25-14b-v18-1`.

If MCQ/RMS/VSP individually miss their floors, the failing axis is the
next iteration's target (v18.2). v18.1 itself does not add new families
beyond AB.MCQ.{7,8,9}; it only re-aligns the failing axes to their
historical-peak recipes.

