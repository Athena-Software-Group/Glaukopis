# AAIML 2027 (L3544): additional runs on Jarvis Labs

These runs close the review items the existing data cannot: comment 1 (matched single-pass baseline, control without the attribution layer) and comment 5 (matched Qwen3 serving, training seeds). The camera-ready is complete without them. Each run replaces a hedged sentence with a measured result. See `RESPONSE_TO_REVIEW.md` for the comment-by-comment status.

**Deadline:** final paper and registration form to aaiml_conf@163.com by **30 Nov 2026**. Start the training runs by about 18 Nov to leave time for evaluation, analysis and edits.

**Same data, same harness (decided 7 Nov 2026).** Every run trains on the seven v21 shard files already in `SFT/data` (dated 2026-05-18; row counts match the `_v21_*_build` reports) and is scored with the unchanged benchmark code. Do not regenerate the templates or refresh the Ariadne graph for these runs. A trial rebuild of Core from the same graph and the same templates reproduced only 35% of the rows: 0% of MCQ, 2% of TAA, 1.7% of RCM and 4.2% of VSP rows. The generator samples with `ORDER BY rand()`, so a controlled comparison is only possible on the original files. The library versions are pinned to May's below.

## Handoff: working from the GPU host (8 Nov 2026)

For a session running on the GPU box. The planning conversations and their notes are not on this machine, so this section holds what the rest of the plan does not.

**What is where.** The camera-ready is finished; these runs replace hedged sentences in it. `doc/` is no longer in the repo. The paper, `RESPONSE_TO_REVIEW.md`, `doc/AAIML/analysis/` and the published per-item files (`SFT/test/responses/`) exist only on the author's Mac, so analysis and paper edits happen there. This box only trains, scores, and hands results back.

**Check the box first.** It may be a new instance with the earlier `/home` attached, so do not assume its state: run `git pull --ff-only`, confirm `SFT/.env` exists (never print it), and compare any existing conda envs against the pinned versions under Setup before reusing them. **Always pass `--no-disk-cleanup` to `setup.sh` on a box with earlier work:** by default it deletes every `SFT/saves/<model>/<type>/<run>` folder older than 14 days, which would remove the May 8B saves still on `/home`.

**Verify the training data before any run.** These files are gitignored and must be the originals. If any is missing or differs, stop: they can only come from the Mac, so ask the user to rsync them. Do not rebuild them.

| `SFT/data/ift_data_2026_05_18_v21_*.json` | Rows | sha256 starts |
|---|---:|---|
| `core_a_kb_mcq_taa_soc_cm_ms_yn` | 168,859 | `527208fe9a0de10a` |
| `core_b_rms_ate_vsp_rcm` | 49,581 | `f8b79bb56d157c7a` |
| `taa` | 20,359 | `f29806b374c9ae37` |
| `cse` | 17,320 | `3d3e32550c31bde4` |
| `core_val` | 550 | `8e16ad2078f87666` |
| `taa_val` | 150 | `55399171ae52369c` |
| `cse_val` | 350 | `6ccb275a32b2315d` |

**Hardware: 4×H200 with `--no-offload` is the working setup (8 Nov).** The table below assumes 8×H100, which is what the paper states, so these runs need the disclosure clause noted further down. On 4×H200 (141 GB each, 1,200 GiB RAM limit), pass `--no-offload` to every launcher, which keeps the optimizer state on the GPUs; the effective batch is unchanged.
- Measured on G2a: 9.6 s per optimizer step, 22.6 min of training, GPUs about 100% busy, peak 95 GB per GPU and 70 GB of system memory. A resumable checkpoint was 242 GB.
- `--resume` is proven: from G2a's final checkpoint it reloaded the optimizer state in about 50 s and trained on for five steps, with loss in line with where it stopped.
- 2×H200 (600 GiB RAM limit) was tried first and abandoned, with nothing pushed. Full CPU offload was killed at the RAM limit before the first step. With `--offload-opt` it ran at about 190 s per step, because one CPU thread per rank ran the 8-bit optimizer while the GPUs idled, and the ranks held 553 GB, leaving no room for the save.

**Status (8 Nov).** G2a is on Hugging Face and its upload was verified by hash. G2b, G1 and the G4 seeds run as a queue on the training box: `~/train_queue.sh` in `tmux`, progress in `~/logs/queue.log`, one log per job in `~/logs/`. A finished job leaves `~/logs/<name>.done` and is skipped on a rerun.

**What survives on `/home`** if the new instance mounts it: the repo, the seven verified shards, `SFT/.env`, the Hugging Face cache (including `…-v21-core`) and the May 8B saves. The conda environments are in `/root/miniconda3` on the container disk, so expect to reinstall them with the Setup commands.

**Running the queue.**
- On a new box, run G2a's equivalent first: it is short and proves the setup. Watch `nvidia-smi` and `free -g` through its first save, when the model is gathered into system RAM.
- **Resume (decided 8 Nov):** by default the launchers pass `--save_only_model True`, so checkpoints cannot resume. Run G1 and G2a with `--resumable`, which writes full ZeRO checkpoints (about 260 GB each at 32B) and keeps one. If G1 crashes, rerun it with `--output-dir <its folder> --resume`. This was proven on G2a on 8 Nov (see above).
- **Disk:** `SFT/saves` and the Hugging Face cache are on `/home` (2 TB, about 1.1 TB free on 8 Nov). Conda and `/tmp` are on the 1 TB overlay. G1 with `--resumable` peaks near 650 GB. Remove a run's local saves only after verifying its Hugging Face push, and look before deleting anything else: `SFT/saves` still holds the May 8B runs.
- **Scoring** runs on a separate instance named `alkidemos`, after all training has finished. The training instance only trains, so do not start vLLM there.
- Any run not on eight 80-GB H100 GPUs needs a disclosure clause in the paper, and if the G4 seeds differ in hardware from the released run, report their spread across the new seeds only.

**Rules.**
- Same data, same harness: no template regeneration, no Neo4j, no library changes beyond the pins under Setup.
- The repo is public. Never commit `.env`, tokens, shards, per-item responses, checkpoints or logs; stage files by explicit name; never print token values.
- Starting paid training, deleting checkpoints, and pushing to Hugging Face or git beyond what a launcher does are the user's call.

## Cross-model single-pass runs (9 Nov 2026)

G1 showed a single pass matching the layered chain on Qwen2.5-32B. These runs test the same question on the four other v21 models. Their layered checkpoints and per-item scores already exist, so each model needs one training run.

| Model | `--model-key` | Layered comparator | Hardware | Training estimate |
|---|---|---|---|---:|
| Foundation-Sec-8B-Instruct | `foundation_8b` | `v21-recalibrate` | 2×H200 | about 8 h |
| Llama-3.1-8B-Instruct | `llama31_8b` | `v21-recalibrate` | 2×H200 | about 8 h |
| Qwen2.5-14B-Instruct | `qwen25_14b` | `v21-recalibrate` | 4×H200 | about 7 h |
| Qwen3-30B-A3B-Thinking-2507 | `qwen3_30b_thinking` | `v21-cse` | 4×H200 | about 16 h |

**Recipe.** `SFT/autotrain/run_sft_v21_single_stage_port.sh`: the same six datasets as G1 at each model's own Stage-1A settings (lr 1e-5, cutoff 8192, packing, effective batch 16, one epoch). Pass `--no-offload --resumable`. Same v21 shards and harness; the shard hashes above must match.

**Scoring.** Run `SFT/eval/utils/serve_and_bench_v21_single_stage_port.sh` for each single-pass checkpoint and again for its layered comparator, so both sides of each comparison use the same stack. Use the May-dated package pins from the 9 Nov scoring round.

**Pre-set decision rule (written before any of these runs was trained).**
- A model counts as equivalent if the 90% paired interval of single pass minus layered lies entirely inside ±1.5 points on $B_{bal}$ and ±2 points on MMLU-Pro. This is two one-sided tests at the 5% level.
- ±1.5 is just below the smallest fine-tuning gain the paper reports (+1.6). ±2 is about one seed SD on MMLU-Pro (1.74).
- All five models (these four and Qwen2.5-32B) are reported whichever way they land, and the claim covers only those that pass. G1 meets the rule, but the margins were set after seeing G1, so only these four are a clean test.
- The ten-test mean and per-task differences are reported alongside, not used for the decision. "Equivalent" applies to the overall scores only: on 32B the two approaches differ on individual tasks.

**Analysis.** `python3 analysis/aaiml_2027/equivalence_ports.py` (needs numpy; writes `doc/AAIML/data/equivalence_ports.json`) prints one block per model and applies the rule above to whichever models have all twelve tests scored on both sides. It also prints each layered re-score against its May per-item results, failed-request counts, and one sensitivity line, `B_failed_as_wrong`. That line matters because the harness marks an empty answer list (`{"correct_answers": []}`) as a failed request and CyberSOCEval drops failed requests from its denominator, so a model that declines hard items is not charged for them. On Qwen2.5-32B the layered checkpoint did this on 36 malware items and the single pass on none. The sensitivity line counts those items as wrong; it is reported alongside and does not enter the decision.

**Order on the hardware.** The two 8B models train and score on 2×H200 (Foundation measured 9.4 s/step, 2,584 steps, about 6.8 h, 76–79 GB per GPU), and the 14B and Qwen3 layered comparators are re-scored there too. Qwen2.5-14B and Qwen3 training need 4×H200: the 14B default-optimizer state is estimated at about 128 GB per GPU on two GPUs, too tight on 141 GB, and an earlier 30B-class run failed on two. The queue scripts are in `analysis/aaiml_2027/run_scripts/` (`port_pipeline_2x.sh`, `train_queue_ports.sh`, `score_queue_ports.sh`, `verify_hf_ports.py`).

**Hardware control (10 Nov).** The May layered chains ran on 8×H100 and the single passes on 2×H200 or 4×H200, so method and hardware change together. To separate them, the full layered chain is retrained on 2×H200 for Foundation-Sec-8B and Llama-3.1-8B, using the same stage launchers and shards as May, with each stage's effective batch matched to May's (16, 8, 16, 16, 8). Layer 4 needs `--eff-batch 8`, because below 8 GPUs its launcher otherwise targets 4. The checkpoints go to new repos, `-v21-2xh200-{core,taa,cse,recalibrate}`, with local parents, so the published checkpoints are neither read nor overwritten; the chain wrapper is not used because it cannot redirect them. Only the two final checkpoints are scored. `control_chain_2x.sh` and `control_pipeline_2x.sh` drive this. `equivalence_ports.py` prints the retrained chain against the single pass on the same hardware, and against the published layered checkpoint, which measures any hardware effect directly. The pre-set rule does not apply to these comparisons. If the retrained chains land where the published layered checkpoints did, hardware is ruled out and the Qwen models get no control. If they land elsewhere, the same control runs for Qwen2.5-14B and Qwen3 on the 4×H200 box.

**Results so far (10 Nov, 08:50 UTC).** Single pass minus layered, 90% paired interval, 2,000 bootstrap draws:

| Model | $B_{bal}$ layered / single | Difference | MMLU-Pro difference | Ten-test difference | Rule |
|---|---|---|---|---|---|
| Qwen2.5-32B (reference) | 64.92 / 65.16 | +0.24 [−0.49, +0.96] | −0.61 [−1.30, +0.11] | +0.60 | inside both margins |
| Foundation-Sec-8B | 53.60 / 60.12 | +6.53 [+5.70, +7.36] | +10.97 [+10.17, +11.76] | +6.13 | outside: single pass better |
| Llama-3.1-8B | 49.79 / 57.55 | +7.76 [+7.04, +8.47] | +5.01 [+4.25, +5.79] | +6.65 | outside: single pass better |

On both 8B models the single pass is clearly ahead on MCQ, CyberSOCEval TI (Llama: 6.3 layered vs 41.4) and both CyberMetric sets, and behind on RCM by 2–3 points; VSP goes −8.9 on Foundation and +7.2 on Llama. Every layered re-score matches its May per-item results: $B_{bal}$ re-score minus May is −0.10 (32B), +0.12 (14B), +0.15 (Foundation) and −0.01 (Llama), and no MMLU-Pro difference exceeds 0.4. Qwen3 CSE moves +0.80 on $B_{bal}$ because the automatic context length removes the May context-overflow failures on CyberSOCEval TI (159 to 10); its MMLU-Pro is 52.95 against 53.00. The Foundation control started at 08:29 UTC (Phase A: 2,330 steps at effective batch 16, as in May).

**Foundation hardware control (10 Nov, 21:07 UTC).** The layered chain retrained on 2×H200 took the same step counts as May (2,330, 6,198, 74, 107, 755) and 12 h 20 min. It does not land where the published layered checkpoint did: $B_{bal}$ 51.13 vs 53.60, −2.46 [−3.15, −1.73]; MMLU-Pro 24.37 vs 25.63, −1.26 [−1.95, −0.61]; ten-test −1.75; `B_failed_as_wrong` −1.90. $A_{bal}$ is unchanged (+0.09), but single tasks move both ways: MCQ +5.5, RMS +9.6, canonical TAA −21.8, CyberSOCEval TI −12.1, malware −4.4, CyberMetric −1.6 and −2.2. The retrained model also writes prose instead of a CVSS vector on 203 VSP items (206 failed requests vs 4), which VSP leaves out of its denominator. On the same hardware the single pass leads by +8.99 [+8.21, +9.73] on $B_{bal}$ and +12.23 [+11.47, +13.02] on MMLU-Pro, so for Foundation the single-pass result does not depend on the hardware. One retrain cannot separate the hardware from ordinary retraining variation; either way, a layered checkpoint trained on other hardware is not a like-for-like comparator. By the condition above this triggers the Qwen controls; Llama's control (due 11 Nov) is the second data point.

**Llama hardware control (11 Nov, 09:42 UTC).** Same step counts as May again, 12 h 23 min. The retrained chain lands above the published layered checkpoint: $B_{bal}$ 53.49 vs 49.79, +3.70 [+3.00, +4.46]; MMLU-Pro +1.28 [+0.58, +1.89]; ten-test +3.87. Tasks move both ways again: CyberSOCEval TI +21.4, VSP +8.3, MCQ +5.7, canonical TAA −9.8. On the same hardware the single pass still leads, by +4.06 [+3.25, +4.81] on $B_{bal}$ and +3.73 [+3.03, +4.50] on MMLU-Pro (ten-test +2.78). $A_{bal}$ is a tie (+0.92 [−0.10, +1.99]), and the single pass is behind on MCQ, ATE, RCM and RMS; its lead comes from canonical TAA (+42.9), TI (+13.7), CyberMetric and MMLU-Pro.

**What the two controls show.** The retrained layered chains move in opposite directions (Foundation −2.46, Llama +3.70 on $B_{bal}$), so there is no consistent hardware effect; a full layered retrain moves the overall score by 2.5 to 3.7 points and single tasks by up to 22. That is larger than the ±1.5 margin of the pre-set rule, so the rule can only be applied cleanly when both arms are trained on the same hardware. On matched hardware the single pass beats the layered chain on both 8B models ($B_{bal}$ +8.99 and +4.06; MMLU-Pro +12.23 and +3.73).

**Decision, 11 Nov: the single pass is the baseline.** The single consolidated instruction fine-tuning run matched or beat the layered chain on every model compared on matched hardware, so the study takes it as the baseline result for all five reported models and does not retrain the layered chain again. Layered retrains of Qwen2.5-14B and -32B on 4×H200 were scripted the same morning and cancelled before any stage ran.

**4×H200 round (from 11 Nov).** The box was resized to 4×H200 to finish the single passes. `pipeline_4x.sh` trains and scores the Qwen2.5-14B single pass (2,692 steps at effective batch 16, the same count as G1; about 7 h); `pipeline_qwen3_4x.sh` then trains and scores the Qwen3-30B-A3B-Thinking single pass (about 16 h), after checking for 560 GB of free disk. To make room, the cached copies of five checkpoints that are scored and on Hugging Face were deleted from the box (G2b, the three G4 seeds and Qwen3 CSE). With these two runs every reported model has a single-pass result: Qwen2.5-32B (G1, 4×H200), Foundation-Sec-8B and Llama-3.1-8B (2×H200), Qwen2.5-14B and Qwen3 (4×H200).

## The runs

| ID | Run | Answers | Hardware | Wall time | GPU-h | Cost at $2.69/GPU-h |
|---|---|---|---|---:|---:|---:|
| G1 | Matched single pass: Qwen2.5-32B-Instruct, one epoch over Core-A + Tülu-3 (240K) + Alpaca + Core-B + TAA + CSE at the Stage-1A recipe | 1 | 8×H100 80GB SXM | 18–23 h | 144–184 | $387–495 |
| G2a | Skip-attribution: Core → CSE (TAA layer omitted) | 1 | 8×H100 | 2–4 h | 16–32 | $43–86 |
| G2b | Skip-attribution: → Recal-32B on top of G2a | 1 | 8×H100 | 3–4 h | 24–32 | $65–86 |
| G3 | Qwen3-30B-A3B CSE checkpoint, CTI suite with thinking on (matches its base) | 5a | 2×H100 | 1–2 h | 2–4 | $5–11 |
| G4 | Stage-4 Recal-32B seeds 1, 2, 3 (released run = seed 42) | 5b | 8×H100 | 9–12 h | 72–96 | $194–258 |
| Eval | Full suite for the 6 new 32B checkpoints, about 1–1.5 h each | — | 2×H100 | 6–9 h | 12–18 | $32–48 |
| **Total** | | | | | **270–366** | **$726–985** |

Budget about $850–1,150 to cover setup, model downloads and idle time. Pricing is the Jarvis Labs on-demand H100 SXM rate (billed per minute); Jarvis changed its rates on 5 Oct 2026, so confirm the current figure before launching. Optional extra: the G1 learning-rate sensitivity run (`--lr 5e-06`, same cost as G1). The launcher header suggests running both and choosing between them by validation loss; one run at the Stage-1A recipe already answers the reviewer.

If the budget is tight, prioritize:
1. **G3**, about $10: it removes the last serving caveat.
2. **G2a/G2b plus their evaluations**, about $120–190: the attribution-omitted control.
3. **G1 plus its evaluation**, about $400–520: the reviewer's main request.
4. **G4**, about $200–270.

### Sources for the estimates

- **G1:** the launcher's estimate, Stage 1A (16–18 h) plus 10–25%. The run consumes 349.4M tokens, 99.4% of the chain's 351.6M (replayed with `analysis/stage_tokens.py`). The 2.2M difference is the Layer-4 replay, whose rows are already in the union. Only 8 Core-B rows exceed the 8,192-token cutoff.
- **G2a:** the CSE launcher's header estimate (9–13 h) predates the v21 shard. v21 CSE is 7.3M tokens, about 2% of Stage 1A's, so the run is under an hour of compute plus model load, the ZeRO-3 save and the HF push.
- **G2b and G4:** the Recal-32B launcher's estimate of 3–4 h on 8×H100 with offload. Do not run it on fewer than 8 GPUs; the header records a host out-of-memory failure at save time on 4×H100.
- **Evaluation:** the v21 32B checkpoints took 0.5–1.2 h each for the full suite on 2×H100 (`SFT/eval/responses/cost_summary.csv`), plus about 15 min to download each 65 GB checkpoint.
- **G3:** the checkpoint was trained to emit an empty reasoning trace, so thinking-on serving should cost about as much as its 1.2 h no-think run. If it emits real traces, the 8,192-token floor makes the run much longer; for comparison, the thinking base took 38 h.

## Instances

- **Training box:** 8×H100 80GB SXM. Request at least 1 TB of disk; each 32B run writes about 65 GB per saved model and keeps up to two checkpoints. Delete local saves once the HF push is verified.
- **Evaluation box:** 2×H100 80GB.

One 8-GPU box can also do everything in sequence. That takes 2.5–3 days, versus about 1.5 days with two training boxes plus the evaluation box running in parallel.

## Setup (each box)

Prerequisite: `main` must include the `--seed` option in `run_sft_qwen25_32b_v21_recal_32b.sh` and the new aliases in `SFT/eval/pipelines/models.py` (the commit that added this file).

`SFT/.env` on your machine must hold the following:
- `GITHUB_TOKEN`, `GIT_USER_NAME` and `GIT_USER_EMAIL`;
- `HF_TOKEN`, with write access to `asg-ai`, and `HF_USERNAME=asg-ai`;
- optionally `WANDB_API_KEY`, `WANDB_ENTITY` and `WANDB_PROJECT`. Without them, pass `--report-to none` to the training launchers.

Never commit `.env`.

```bash
# From your machine (Jarvis shows the SSH host and port for each instance; add -p PORT / -e "ssh -p PORT")
scp SFT/utils/bootstrap_remote.sh SFT/.env root@HOST:/root/
ssh root@HOST bash /root/bootstrap_remote.sh     # clones to ~/Glaukopis, moves .env into SFT/

# The v21 shards are gitignored: copy the seven files the runs read (~344 MB)
rsync -avP SFT/data/ift_data_2026_05_18_v21_{core_a_kb_mcq_taa_soc_cm_ms_yn,core_b_rms_ate_vsp_rcm,taa,cse,core_val,taa_val,cse_val}.json \
      root@HOST:~/Glaukopis/SFT/data/

# Environments
# Pinned to the May 2026 versions (see "Same data, same harness")
ssh root@HOST 'cd ~/Glaukopis/SFT/utils && LIGER_SPEC="liger-kernel==0.8.0" BNB_SPEC="bitsandbytes==0.49.2" \
    ./setup.sh --cuda cu128 --mode train --no-disk-cleanup'    # training box: llm-sft
ssh root@HOST 'cd ~/Glaukopis/SFT/utils && VLLM_SPEC="vllm==0.21.0" \
    ./setup.sh --cuda cu128 --mode vllm --no-disk-cleanup'     # eval box: vllm + ctibench + CyberSOCEval fetch
```

On the training box, the in-repo LlamaFactory is unchanged since 18 May. Its capped dependencies resolve to the same releases as in May: transformers 5.2.0, deepspeed 0.18.4, accelerate 1.11.0, peft 0.18.1, trl 0.24.0 and torch 2.8.0. Only liger-kernel and bitsandbytes (which provides `adamw_8bit`) are uncapped, hence the two pins. If the pinned vLLM fails to install, drop `VLLM_SPEC`; the anchor check below then decides whether the results are comparable.

Run long jobs inside `tmux` so they survive a dropped SSH session. Tülu-3, the base model and the chain checkpoints download from the Hugging Face Hub on first use.

## Training (training box: `conda activate llm-sft; cd ~/Glaukopis/SFT/autotrain`)

Dry-run each command first (`--dry-run`): it prints the resolved base model, datasets, hyperparameters and HF repo. Then run it.

```bash
# G1 - matched single pass  ->  asg-ai/athena-cti-sft-qwen25-32b-v21-single-stage
./run_sft_qwen25_32b_v21_single_stage.sh --resumable

# G2a - Core -> CSE, attribution layer omitted  ->  asg-ai/athena-cti-sft-qwen25-32b-v21-skiptaa-cse
./run_sft_qwen25_32b_v21_final.sh --resumable \
    --base-model asg-ai/athena-cti-sft-qwen25-32b-v21-core \
    --repo-id    asg-ai/athena-cti-sft-qwen25-32b-v21-skiptaa-cse

# G2b - Recal-32B on top of G2a (after G2a's push completes)  ->  ...-v21-skiptaa-recal-32b
./run_sft_qwen25_32b_v21_recal_32b.sh \
    --base-model asg-ai/athena-cti-sft-qwen25-32b-v21-skiptaa-cse \
    --repo-id    asg-ai/athena-cti-sft-qwen25-32b-v21-skiptaa-recal-32b

# G4 - Stage-4 seed replicates from the released CSE parent  ->  ...-v21-recal-32b-seed{1,2,3}
for s in 1 2 3; do ./run_sft_qwen25_32b_v21_recal_32b.sh --seed "$s" || break; done
```

The seed changes both the training order and the interleaved replay sample, so G4 measures Stage-4 training variability. Seeding the whole chain would cost about 50 h of 8×H100 time per seed, which is out of scope for this deadline.

## Evaluation (eval box: `conda activate vllm; cd ~/Glaukopis/SFT/eval/utils`)

**Check the serving stack first.** The benchmark data and scoring code are unchanged since the published scoring runs of 20–21 May 2026 (the only edits since are cache keying and the `SFT/test` → `SFT/eval` rename). The upstream sets fetched at setup are also unchanged: CyberSOCEval_data last changed 2025-09-22, PurpleLlama `crwd_meta` 2026-05-04, MMLU-Pro 2026-05-02. No vLLM version was recorded for those runs. The setup above pins 0.21.0, the latest release from 15 to 29 May and so the most likely one; an unpinned install now gets 0.33.0. Confirm with an anchor before scoring any new checkpoint:

```bash
pip freeze > ~/Glaukopis/SFT/eval/responses/eval_stack_$(date +%Y%m%d).txt
# Anchor: re-score the published Recal-32B on this box (~1-1.5 h)
MODEL_ALIAS=athena-cti-sft-qwen25-32b-v21-recal-32b-vllm BENCH_CONDA_ENV=ctibench \
  bash serve_and_bench_qwen25_32b_v21_single_stage.sh
```

Copy it back with the other results and run `bootstrap_new_runs.py`, which pairs `Recal-32B rerun` with the published run. If every aggregate's interval includes 0, score the new checkpoints on this stack. If not, re-score all five published checkpoints on this stack (about 5–7 h, $27–38) and compare the new runs with those, so every comparison stays within one stack.

```bash
# G3 - needs no training; start it first
BENCH_CONDA_ENV=ctibench bash serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_cse_thinking_on.sh

# Each new 32B checkpoint, once its HF push has completed (same serving and suite as the chain: TP 2, 32K context, MMLU-Pro on)
for a in single-stage skiptaa-cse skiptaa-recal-32b recal-32b-seed1 recal-32b-seed2 recal-32b-seed3; do
  MODEL_ALIAS=athena-cti-sft-qwen25-32b-v21-$a-vllm BENCH_CONDA_ENV=ctibench \
    bash serve_and_bench_qwen25_32b_v21_single_stage.sh
done
```

Results land in `~/Glaukopis/SFT/eval/responses/<alias>/`. That directory is gitignored. Do not commit per-item files; they are among the large artifacts the paper offers on request.

## Analysis (your machine)

```bash
rsync -avP 'root@HOST:~/Glaukopis/SFT/eval/responses/athena-cti-sft-qwen25-32b-v21-*-vllm' SFT/eval/responses/
rsync -avP root@HOST:~/Glaukopis/SFT/eval/responses/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-vllm SFT/eval/responses/
python3 -I doc/AAIML/analysis/bootstrap_new_runs.py      # -> doc/AAIML/data/bootstrap_new_runs.json
```

`bootstrap_new_runs.py` picks up each new checkpoint automatically once its scored files exist. It uses the same scorer, aggregates and resampling seed as the published intervals, and with only the five published checkpoints it reproduces them exactly (Recal-32B − Core: $B_{bal}$ +3.18 [2.20, 4.16]). It prints:
- **Comment 1:** Recal-32B − 1-pass, CSE − SkipTAA-CSE and Recal-32B − SkipTAA-Recal, with paired 95% intervals. Positive values mean the layered chain scores higher.
- **Comment 5b:** the mean and SD of each aggregate across seeds 42, 1, 2 and 3.

G3 is a Qwen3 checkpoint, so it is not in the paired script. Compute its $B_{bal}$ and ten-test mean with the paper's uniform aggregation and compare them with the thinking-on base (61.8 and 54.9). Use the `Final  (v21) 14B 32B` sheet of `SFT/eval/Glaukopis Results.xlsx`, which applies the TAA blend consistently. Do not use the "AB Avg II" column of `Final wMMLUPro-v21`.

## What happened in the 9 Nov 2026 scoring round

G1, G2a, G2b, G3 and G4 were all scored on one 2×H200 box, because the separate scoring instance would not come up.

- **Stack.** Both scoring environments were pinned to what PyPI offered on 20 May 2026 with `pip --uploaded-prior-to` (`pin_may.sh`): vLLM 0.21.0, torch 2.11.0, transformers 5.8.1 in `vllm`; pandas 3.0.3, numpy 2.4.6, openai 2.37.0, scikit-learn 1.8.0 in `ctibench`. The two `pip freeze` files are `SFT/eval/responses/eval_stack_20261109_{vllm,ctibench}.txt`. A resized box loses its conda environments, so repeat the pinning after a resize.
- **Prefetch.** `prefetch.py` downloads each checkpoint before serving, with retries. Otherwise a slow Hub download counts against the serving timeout.
- **Anchor passed.** The published Recal-32B re-scored on this stack differs from its May run by −0.10 [−0.27, +0.06] on $B_{bal}$, −0.07 [−0.28, +0.12] on the ten-test mean and +0.02 [−0.28, +0.34] on MMLU-Pro. All interval checks include zero, so the new checkpoints were scored on this stack as planned.
- **The scorer's exit code does not mean success.** `run_foundation_8b_baselines.sh` logs a failed suite as a warning and exits 0, so `check_scores.py` confirms every expected row exists before a run is marked done. It reads `expected_rows.json` and a 2 MB `published_items.json` of published item ids, which stays with the local records.
- **Qwen3, May against November.** The thinking-on alias has complete May 26 CTI results in the old scoring instance's archive. They carry 159 TI and 64 malware `ctx-overflow` failures, which November's automatic context length does not have (10 TI and 0 malware). That accounts for the +5.15 difference on CyberSOCEval-TI; the other ten CTI tests show no drift.
- **Where results live.** Per-item results are in `SFT/eval/responses/<alias>/` on the analysis machine, gitignored like all per-item files. Training and scoring records are kept locally and are available on request. The scripts that drove the rounds are in `analysis/aaiml_2027/run_scripts/`.

## Putting results into the paper

Page 10 is full, so every addition needs an equal cut. Prefer text to new table rows. The `%% RUN-SLOT` comments in `aaiml2027_camera_ready.tex` mark the places:

| Result | Replace | Location |
|---|---|---|
| G1, G2 | The last sentence of §VII-C ("Because the later layers also add … were not run for this paper") with the paired differences against the single pass and the skip-attribution chain | §VII-C, plus the *Causal attribution* limitation in §VIII-B (delete "were not run") |
| G3 | The clause after "so that loss is not a serving artifact" with the matched CTI comparison | §VII-F |
| G4 | "every checkpoint is one training run, and seeded replication … would test recipe robustness" with the seed mean ± SD of $B_{bal}$ and $A_{bal}$ | §VIII-B *Uncertainty and reproducibility* |

Candidate cuts, if a sentence grows:
- one hosted-model row of Table XII, or a sentence of §VII-G *Hosted Reference Models*;
- one row of Table I (related-work comparison);
- the sentence on the 14B Layer-4 variant in §IV.

After editing, rebuild (`pdflatex`, `bibtex`, then `pdflatex` twice) and confirm 10 pages with no overfull boxes. Then update the status table in `RESPONSE_TO_REVIEW.md`.
