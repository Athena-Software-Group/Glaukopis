# AAIML 2027 (L3544): additional runs on Jarvis Labs

These runs close the review items the existing data cannot: comment 1 (matched single-pass baseline, control without the attribution layer) and comment 5 (matched Qwen3 serving, training seeds). The camera-ready is complete without them. Each run replaces a hedged sentence with a measured result. See `RESPONSE_TO_REVIEW.md` for the comment-by-comment status.

**Deadline:** final paper and registration form to aaiml_conf@163.com by **30 Nov 2026**. Start the training runs by about 18 Nov to leave time for evaluation, analysis and edits.

**Same data, same harness (decided 7 Nov 2026).** Every run trains on the seven v21 shard files already in `SFT/data` (dated 2026-05-18; row counts match the `_v21_*_build` reports) and is scored with the unchanged benchmark code. Do not regenerate the templates or refresh the Ariadne graph for these runs. A trial rebuild of Core from the same graph and the same templates reproduced only 35% of the rows: 0% of MCQ, 2% of TAA, 1.7% of RCM and 4.2% of VSP rows. The generator samples with `ORDER BY rand()`, so a controlled comparison is only possible on the original files. The library versions are pinned to May's below.

## Handoff: working from the GPU host (8 Nov 2026)

For a session running on the GPU box. The planning conversations and their notes are not on this machine, so this section holds what the rest of the plan does not.

**What is where.** The camera-ready is finished; these runs replace hedged sentences in it. `doc/` is no longer in the repo. The paper, `RESPONSE_TO_REVIEW.md`, `doc/AAIML/analysis/` and the published per-item files (`SFT/test/responses/`) exist only on the author's Mac, so analysis and paper edits happen there. This box only trains, scores, and hands results back.

**Check the box first.** It is a resumed instance, so do not assume its state: run `git pull --ff-only`, confirm `SFT/.env` exists (never print it), and compare any existing conda envs against the pinned versions under Setup before reusing them.

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

**Hardware differs from the table below.** The training box is 2×H200 141 GB with about 600 GB of system RAM (confirm with `nvidia-smi` and `free -g`), not 8×H100.
- The launchers detect the GPU count and raise gradient accumulation, so the effective batch stays the same (16 for G1 and G2a, 8 for G2b and G4). They print a warning and continue. Leave CPU offload on, which is automatic below 4 GPUs; without it the per-GPU state is not expected to fit.
- 32B has never been trained on 2 GPUs in this repo. Launcher times scaled by GPU count give: G2a 4–6 h, G2b 8–12 h, each G4 seed 8–12 h, G1 65–90 h.
- Run G2a first as the test of this setup. Watch `nvidia-smi` and `free -g` through its first save, when the model is gathered into system RAM; that step ran a 4×H100 host out of memory.
- **Open decision, ask the user before G1:** every launcher passes `--save_only_model True`, so checkpoints cannot resume, despite what the `--resume` comments say. The proposal is to drop that flag for G1 (about 250 GB per checkpoint, two kept, so at least 1 TB of disk) and prove resume on G2a first.
- **Scoring:** this plan assumes a separate 2-GPU eval box. If this is the only box, scoring waits behind training.
- The paper says "eight 80-GB H100 GPUs", so the new runs need a disclosure clause, and the G4 spread should be reported across the three new seeds only.

**Rules.**
- Same data, same harness: no template regeneration, no Neo4j, no library changes beyond the pins under Setup.
- The repo is public. Never commit `.env`, tokens, shards, per-item responses, checkpoints or logs; stage files by explicit name; never print token values.
- Starting paid training, deleting checkpoints, and pushing to Hugging Face or git beyond what a launcher does are the user's call.

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
    ./setup.sh --cuda cu128 --mode train'    # training box: llm-sft
ssh root@HOST 'cd ~/Glaukopis/SFT/utils && VLLM_SPEC="vllm==0.21.0" \
    ./setup.sh --cuda cu128 --mode vllm'     # eval box: vllm + ctibench + CyberSOCEval fetch
```

On the training box, the in-repo LlamaFactory is unchanged since 18 May. Its capped dependencies resolve to the same releases as in May: transformers 5.2.0, deepspeed 0.18.4, accelerate 1.11.0, peft 0.18.1, trl 0.24.0 and torch 2.8.0. Only liger-kernel and bitsandbytes (which provides `adamw_8bit`) are uncapped, hence the two pins. If the pinned vLLM fails to install, drop `VLLM_SPEC`; the anchor check below then decides whether the results are comparable.

Run long jobs inside `tmux` so they survive a dropped SSH session. Tülu-3, the base model and the chain checkpoints download from the Hugging Face Hub on first use.

## Training (training box: `conda activate llm-sft; cd ~/Glaukopis/SFT/autotrain`)

Dry-run each command first (`--dry-run`): it prints the resolved base model, datasets, hyperparameters and HF repo. Then run it.

```bash
# G1 - matched single pass  ->  asg-ai/athena-cti-sft-qwen25-32b-v21-single-stage
./run_sft_qwen25_32b_v21_single_stage.sh

# G2a - Core -> CSE, attribution layer omitted  ->  asg-ai/athena-cti-sft-qwen25-32b-v21-skiptaa-cse
./run_sft_qwen25_32b_v21_final.sh \
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
