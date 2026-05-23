import os
import re
import sys
import torch
import shutil
import random
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import traceback
from pipelines.api_usage import add_tokens, get_totals

# Load environment variables
load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Login to HuggingFace if token is available
if hf_token:
    try:
        login(token=hf_token)
        print("HuggingFace login successful")
    except Exception as e:
        print(f"HuggingFace login failed: {e}")
else:
    print("HUGGINGFACE_TOKEN not found. Gated models (like Llama) will fail.")

# Set up HuggingFace cache (cross-platform)
# Check if we're in a container/workspace environment
if os.path.exists("/workspace"):
    workspace_cache = "/workspace/.cache/huggingface"
else:
    # Use current working directory for local development
    workspace_cache = os.path.join(os.getcwd(), ".cache", "huggingface")

os.makedirs(workspace_cache, exist_ok=True)
os.environ['HF_HOME'] = workspace_cache
os.environ['TRANSFORMERS_CACHE'] = workspace_cache
os.environ['HF_DATASETS_CACHE'] = workspace_cache
print(f" HuggingFace cache directory: {workspace_cache}")

model_mapping = {
    #'gpt3': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4-turbo-2024-04-09',
    'gpt5': 'gpt-5',
    'gpt5.2': 'gpt-5.2',
    'gpt5.5': 'gpt-5.5',
    'gpt5.5-pro': 'gpt-5.5-pro',
    'gemini-2.5-flash': 'gemini-2.5-flash',
    'gemini-2.5-pro' : 'gemini-2.5-pro',
    'gemini-3-pro' : 'gemini-3-pro-preview',
    'gemini-3-flash' : 'gemini-3-flash-preview',
    'gemini-3.1-pro' : 'gemini-3.1-pro-preview',
    'llama-3-3b': 'meta-llama/Llama-3.2-3B',
    'llama-3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama-primus-8b' : 'trendmicro-ailab/Llama-Primus-Merged',
    'athena-cti-sft-llama31-8b': 'asg-ai/athena-cti-sft-llama31-8b',
    'athena-cti-sft-llama31-8b-mcqfixed': 'asg-ai/athena-cti-sft-llama31-8b-mcqfixed',
    'athena-cti-sft-llama31-8b-abaligned': 'asg-ai/athena-cti-sft-llama31-8b-abaligned',
    'athena-cti-sft-llama31-8b-abaligned-v3': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v3',
    'athena-cti-sft-llama31-8b-abaligned-v4': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v4',
    'athena-cti-sft-llama31-8b-abaligned-v5': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v5',
    'athena-cti-sft-llama31-8b-abaligned-v5-lora': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v5-lora',
    'athena-cti-sft-llama31-8b-abaligned-v6': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v6',
    'athena-cti-sft-llama31-8b-abaligned-v7': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v7',
    'athena-cti-sft-qwen25-14b-abaligned-v7': 'asg-ai/athena-cti-sft-qwen25-14b-abaligned-v7',
    'athena-cti-sft-qwen25-32b-abaligned-v7': 'asg-ai/athena-cti-sft-qwen25-32b-abaligned-v7',
    #'llama-4-17b': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    'minerva' : "xashru/minerva_v0",
    'llama3.3-70b': 'meta-llama/Llama-3.3-70B-Instruct',
    'qwen2.5-14b': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen2.5-32b': 'Qwen/Qwen2.5-32B-Instruct',
    'qwen3-4b': 'Qwen/Qwen3-4B-Instruct-2507',
    'qwen3-8b': 'Qwen/Qwen3-8B',
    'qwen3-14b': 'Qwen/Qwen3-14B',
    'qwen3-32b': 'Qwen/Qwen3-32B',
    'qwen3.5-9b':'Qwen/Qwen3.5-9B',
    'gpt-oss-20b': 'openai/gpt-oss-20b',
    'foundation-8b-reasoning': 'fdtn-ai/Foundation-Sec-8B-Reasoning',       # Cisco Foundation-Sec-8B-Reasoning
    'foundation-8b-instruct': 'fdtn-ai/Foundation-Sec-8B-Instruct',         # Cisco Foundation-Sec-8B-Instruct (SFT+RLHF, custom <|system|>/<|user|>/<|assistant|> template, Aug 2025)
    'foundation-8b': 'fdtn-ai/Foundation-Sec-8B',                           # Cisco Foundation-Sec-8B simple model
    'minerva-llama8b':'athena-security/minerva-llama8b',
    'minerva-llama31-8b': 'asg-ai/minerva-llama3.1-8b',
    'deephat-7b': 'DeepHat/DeepHat-V1-7B',
    'deepseek-r1-14b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',

    # --- HF Inference Providers (hosted; '-hf' suffix routes to HFInferenceModel) ---
    'deepseek-r1-14b-hf':  'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    'deepseek-r1-70b-hf':  'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'qwen3-14b-hf':        'Qwen/Qwen3-14B',
    'qwen3-32b-hf':        'Qwen/Qwen3-32B',
    'qwen2.5-14b-hf':      'Qwen/Qwen2.5-14B-Instruct',
    'qwen2.5-32b-hf':      'Qwen/Qwen2.5-32B-Instruct',
    'llama-3-70b-hf':      'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama3.3-70b-hf':     'meta-llama/Llama-3.3-70B-Instruct',
    'deepseek-v3.1-terminus-hf': 'deepseek-ai/DeepSeek-V3.1-Terminus',
    'deepseek-v3.2-exp-hf': 'deepseek-ai/DeepSeek-V3.2-Exp',
    'deepseek-v4-pro-hf':  'deepseek-ai/DeepSeek-V4-Pro',
    'deepseek-v4-flash-hf': 'deepseek-ai/DeepSeek-V4-Flash',
    'kimi-k2.6-hf':        'moonshotai/Kimi-K2.6',
    'gemma-4-31b-hf':      'google/gemma-4-31B-it',
    'qwen3.5-plus-hf':     'Qwen/Qwen3.5-397B-A17B',
    'athena-cti-cpt-llama31-8b-v1': 'asg-ai/athena-cti-cpt-llama31-8b-v1',
    'llama-3-8b-base': 'meta-llama/Llama-3.1-8B',

    # --- Local vLLM server ('-vllm' suffix routes to VLLMModel). The HF repo
    # id is the same as the non-vllm alias; suffix selects the inference path.
    # VLLM_BASE_URL (default http://localhost:8000/v1) points at a running
    # `vllm serve <repo-id>` process. See SFT/test/utils/serve_vllm.sh.
    'llama-3-8b-base-vllm':                    'meta-llama/Llama-3.1-8B',
    'llama-3-8b-vllm':                         'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen3-4b-vllm':                           'Qwen/Qwen3-4B-Instruct-2507',
    'qwen3-32b-vllm':                          'Qwen/Qwen3-32B',
    # Qwen3-32B served with the hybrid <think> trace disabled. Same HF repo as
    # qwen3-32b-vllm; the '-no-think' substring is detected by VLLMModel which
    # forwards `chat_template_kwargs.enable_thinking=False` on every request so
    # the chat template skips the reasoning preamble. Use this for short-answer
    # MCQ tasks (CKT/ATE/TAA/CyberMetric) where the trace eats the generation
    # budget; keep qwen3-32b-vllm for tasks that benefit from CoT.
    'qwen3-32b-no-think-vllm':                 'Qwen/Qwen3-32B',
    # Qwen3-30B-A3B pure-instruct July 2025 split. MoE architecture: ~30.5B
    # total params, ~3.3B active per token (128 experts, 8 routed). Distinct
    # from the dense Qwen3-32B above. Pure non-thinking by design (no hybrid
    # `<think>` mode), 262K native ctx. Pre-train ~7 months newer than the
    # Qwen2.5 family. Use this as the ~30B-class pure-instruct baseline;
    # weights footprint (~60 GB bf16 resident) is comparable to dense 32B
    # but decode throughput is ~5-7x faster owing to the 3.3B active path.
    # No '-no-think' suffix needed -- the Instruct-2507 chat template has
    # no thinking mode to suppress.
    'qwen3-30b-a3b-instruct-2507-vllm':        'Qwen/Qwen3-30B-A3B-Instruct-2507',
    # Qwen3-30B-A3B pure-thinking July 2025 split. Same MoE base as the
    # Instruct-2507 variant above but post-trained for reasoning-only
    # output -- always emits a <think>...</think> trace before the final
    # answer. Serve with `--reasoning-parser deepseek_r1` (or `qwen3` on
    # vllm>=0.10) so the trace lands in `reasoning_content` and the
    # bench-visible `content` field is just the final answer. Alias
    # contains the 'thinking' substring (and NOT 'no-think') which
    # VLLMModel detects to raise max_new_tokens to a thinking-mode floor
    # (8192) -- the TASK_MAX_NEW_TOKENS table caps MCQ at 128 which would
    # truncate every row mid-trace and collapse accuracy to <random.
    'qwen3-30b-a3b-thinking-2507-vllm':        'Qwen/Qwen3-30B-A3B-Thinking-2507',
    # Same HF repo as qwen3-30b-a3b-thinking-2507-vllm above, served with the
    # pure-thinking trace suppressed at request time via
    # chat_template_kwargs.enable_thinking=False. The '-no-think' substring
    # also opts the alias OUT of VLLMModel's thinking-mode 8192-token floor
    # so the per-task TASK_MAX_NEW_TOKENS caps (e.g. 1024 for MMLU-Pro)
    # apply unmodified. The base Thinking-2507 was NOT trained with the
    # empty-thought pattern that the v21 SFT instills, so under this
    # inference path it will typically emit a substantive trace and
    # truncate mid-reasoning -- which is the point: this alias exists as
    # the matched-conditions baseline against the SFT'd v21-cse-no-think
    # variant, isolating the SFT's contribution to functioning under a
    # no-trace inference budget.
    'qwen3-30b-a3b-thinking-2507-no-think-vllm': 'Qwen/Qwen3-30B-A3B-Thinking-2507',
    'qwen2.5-14b-vllm':                        'Qwen/Qwen2.5-14B-Instruct',
    'qwen2.5-32b-vllm':                        'Qwen/Qwen2.5-32B-Instruct',
    'phi-4-vllm':                              'microsoft/phi-4',
    'gemma-2-9b-vllm':                         'google/gemma-2-9b-it',
    # Gemma 4 31B Dense (IT). Multimodal (text+image), 256K trained context, 60
    # transformer layers + 1024-tok sliding window. Served text-only here:
    # the bench harness only sends /v1/chat/completions with text, and the
    # vLLM serve cmd should pass `--limit-mm-per-prompt image=0` via --extra
    # to skip the vision-encoder KV pre-allocation. Native ctx is 256K but
    # we cap --max-len at 49152 for parity with the cybersoceval-mode auto-pick
    # (TI rows ~32K-32.7K). Weights ~62GB bf16 -> tp=2 on H100 (~31GB per GPU).
    'gemma-4-31b-it-vllm':                     'google/gemma-4-31B-it',
    'ministral-8b-vllm':                       'mistralai/Ministral-8B-Instruct-2410',
    'mistral-7b-vllm':                         'mistralai/Mistral-7B-Instruct-v0.3',
    # Mistral Small 3.2 24B Instruct (June 2025). 24B dense, multimodal
    # (text+vision); served text-only here, so the serve cmd should pass
    # `--limit-mm-per-prompt image=0` via EXTRA_SERVE_FLAGS to skip the
    # vision-encoder KV pre-allocation. Native ctx 128K, weights ~48GB bf16
    # -> tp=2 on H100 80GB (~24GB/rank). Pitched as a 32B-class candidate
    # for the v21 SFT recipe; this alias is for the pre-SFT baseline.
    'mistral-small-3.2-24b-instruct-2506-vllm': 'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
    'athena-cti-cpt-llama31-8b-v1-vllm':       'asg-ai/athena-cti-cpt-llama31-8b-v1',
    'athena-cti-sft-llama31-8b-abaligned-v3-vllm': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v3',
    'athena-cti-sft-llama31-8b-abaligned-v4-vllm': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v4',
    'athena-cti-sft-llama31-8b-abaligned-v5-vllm': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v5',
    'athena-cti-sft-llama31-8b-abaligned-v5-lora-vllm': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v5-lora',
    'athena-cti-sft-llama31-8b-abaligned-v6-vllm': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v6',
    'athena-cti-sft-llama31-8b-abaligned-v7-vllm': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-v7',
    'athena-cti-sft-qwen25-14b-abaligned-v7-vllm': 'asg-ai/athena-cti-sft-qwen25-14b-abaligned-v7',
    'athena-cti-sft-qwen25-32b-abaligned-v7-vllm': 'asg-ai/athena-cti-sft-qwen25-32b-abaligned-v7',
    # v8-small SFT family: 60K stratified mix, 2 epochs, cutoff 8192, full-param.
    # Pushed by SFT/autotrain/run_abaligned_sft_{llama31_8b,foundation_8b,qwen25_14b}_v8*.sh.
    'athena-cti-sft-llama31-8b-abaligned-v8-vllm':             'asg-ai/athena-cti-sft-llama31-8b-abaligned-v8',
    'athena-cti-sft-foundation-8b-instruct-abaligned-v8-vllm': 'asg-ai/athena-cti-sft-foundation-8b-instruct-abaligned-v8',
    'athena-cti-sft-qwen25-14b-abaligned-v8small-vllm':        'asg-ai/athena-cti-sft-qwen25-14b-abaligned-v8small',
    # v8-large recipe on 14B (sized for 32B's parameter budget; pushed by
    # SFT/autotrain/run_abaligned_sft_qwen25_14b_v8.sh after Phase A->B chain).
    'athena-cti-sft-qwen25-14b-abaligned-v8-vllm':             'asg-ai/athena-cti-sft-qwen25-14b-abaligned-v8',
    # v8.1 single-pass recipe on 14B (consolidated 41.8K-row corpus; pushed by
    # SFT/autotrain/run_abaligned_sft_qwen25_14b_v81.sh).
    'athena-cti-sft-qwen25-14b-abaligned-v81-vllm':            'asg-ai/athena-cti-sft-qwen25-14b-abaligned-v81',
    # v9 two-phase 14B (Phase A v8-broad baseline -> Phase B v8.1 RMS slice;
    # pushed by SFT/autotrain/run_abaligned_sft_qwen25_14b_v9.sh).
    'athena-cti-sft-qwen25-14b-abaligned-v9-vllm':             'asg-ai/athena-cti-sft-qwen25-14b-abaligned-v9',
    # v10 single-pass 14B on the 200K-row v10 corpus (TAA cap=20/actor,
    # 50-shared-13gram dedup, <desc> markers; pushed by
    # SFT/autotrain/run_abaligned_sft_qwen25_14b_v10.sh).
    'athena-cti-sft-qwen25-14b-abaligned-v10-vllm':            'asg-ai/athena-cti-sft-qwen25-14b-abaligned-v10',
    # v11 single-pass 14B/32B on the 199K-row v11 corpus (SOC.* / TAA.CANON.*
    # / RMS-paraphrase expansions, F3 anchor-fixation fix, actor cap=40,
    # held-out val slice; pushed by SFT/autotrain/run_sft_qwen25_{14b,32b}_v11.sh).
    # Naming migration per v11_plan.txt §0: "abaligned" suffix dropped.
    'athena-cti-sft-qwen25-14b-v11-vllm':                      'asg-ai/athena-cti-sft-qwen25-14b-v11',
    'athena-cti-sft-qwen25-32b-v11-vllm':                      'asg-ai/athena-cti-sft-qwen25-32b-v11',
    # v12 three-phase 14B/32B (Phase A broad + Phase B RMS/ATE/VSP/RCM drill +
    # Phase C TAA.CANON memorisation; row-count gate on, AB.TAA total cap 3500,
    # stratified shuffle on, athena-cti-taa-canonical bench wired). Pushed by
    # SFT/autotrain/run_sft_qwen25_{14b,32b}_v12.sh. 32B serial after 14B
    # passes v12_plan.txt §8.
    'athena-cti-sft-qwen25-14b-v12-vllm':                      'asg-ai/athena-cti-sft-qwen25-14b-v12',
    'athena-cti-sft-qwen25-32b-v12-vllm':                      'asg-ai/athena-cti-sft-qwen25-32b-v12',
    # v13 two-phase 14B/32B (Phase A broad+canon with TAA.CANON merged in +
    # Phase B axis drill RMS/ATE/VSP/RCM/SOC; v9-shape recipe revert
    # cutoff=8192/packing=ON for both phases; MISP CC-0 TAA expansion;
    # licence-allowlist gate; SOC dual-shard supervision). Pushed by
    # SFT/autotrain/run_sft_qwen25_{14b,32b}_v13.sh. 32B serial after 14B
    # passes v13_plan.txt §8.
    'athena-cti-sft-qwen25-14b-v13-vllm':                      'asg-ai/athena-cti-sft-qwen25-14b-v13',
    'athena-cti-sft-qwen25-32b-v13-vllm':                      'asg-ai/athena-cti-sft-qwen25-32b-v13',
    # v14.1 four-phase 14B narrow-drilling experiment (cutoff-4096 +
    # gradient-checkpointing-off hot-fix of v14; corpus/topology/LR/eff-bs
    # held verbatim). Five sequential passes off Qwen2.5-14B-Instruct:
    #   Phase A    broad re-anchor (no HF push)
    #   Phase B    ATE+VSP+RCM long-context  -> v14p1-ab
    #   Phase D-RMS narrow drill from v14p1-ab -> v14p1-rms (parallel branch)
    #   Phase D-TAA narrow drill from v14p1-ab -> v14p1-taa (parallel branch)
    #   Production D-TAA chained on D-RMS     -> v14p1 (full chain)
    # Pushed by SFT/autotrain/run_sft_qwen25_14b_v14_1.sh. The four
    # checkpoints isolate the narrow-drill effect of each axis (rms vs
    # taa) and let the full-chain production candidate be compared
    # against the parallel-branch checkpoints (decides chained vs
    # parallel deployment per v14_plan.txt §9). 32B repo target reserved
    # pending 14B v14p1 §8 pass. See tmpl_gen/templates/05082026/v14_plan.txt.
    'athena-cti-sft-qwen25-14b-v14p1-ab-vllm':                 'asg-ai/athena-cti-sft-qwen25-14b-v14p1-ab',
    'athena-cti-sft-qwen25-14b-v14p1-rms-vllm':                'asg-ai/athena-cti-sft-qwen25-14b-v14p1-rms',
    'athena-cti-sft-qwen25-14b-v14p1-taa-vllm':                'asg-ai/athena-cti-sft-qwen25-14b-v14p1-taa',
    'athena-cti-sft-qwen25-14b-v14p1-vllm':                    'asg-ai/athena-cti-sft-qwen25-14b-v14p1',
    # v15 W1 experiment: v12+TAA single-phase narrow specialist. Trains the v14 TAA
    # shard (ift_data_2026_05_08_v14_taa, 32,783 rows; CANON excluded) on top of
    # the frozen v12 baseline using the v9 narrow recipe (cutoff=4096, packing=on,
    # lr=5e-6, eff_bs=16). First test of the v15 parallel-branching architecture:
    # does narrow per-axis SFT off a healthy v12 base preserve other-axis capability
    # (vs v14.1's chained five-pass topology, which regressed below v12 across the
    # board). Pushed by SFT/autotrain/run_sft_qwen25_14b_v12_plus_taa.sh. See
    # tmpl_gen/templates/05082026/v15_plan.txt for the W1 decision tree (ship-as-
    # specialist vs escalate-to-merge-sweep vs halt-on-data-issue).
    'athena-cti-sft-qwen25-14b-v12-plus-taa-vllm':             'asg-ai/athena-cti-sft-qwen25-14b-v12-plus-taa',
    # v16: v15 W1-rev. Same v12+TAA single-specialist topology as v12-plus-taa above,
    # rebuilt against the CANON-purged / JSON-bumped v16 shard
    # (ift_data_2026_05_10_v16_taa, 20,339 rows). v15 W1 post-mortem found the v14
    # TAA shard was 68.5% Canonical alias-resolution (a task NOT measured by the
    # formal TAA Classic benchmark) and only ~1.3% JSON-shaped attribution; v16
    # purges CANON entirely and bumps AB.TAA.{1,2,3,5} 1500-2000 -> 3000, JS.TAA.
    # {1,2,3} 400 -> 2500 each, lifting per-actor cap 40 -> 60. Pushed by
    # SFT/autotrain/run_sft_qwen25_14b_v12_plus_v16_taa.sh. See
    # tmpl_gen/templates/05102026/v16_plan.txt and README.md for the W1 post-mortem
    # and v16 design.
    'athena-cti-sft-qwen25-14b-v16-vllm':                      'asg-ai/athena-cti-sft-qwen25-14b-v16',
    # v17: first chained narrow-SFT vintage. Trains on top of v16
    # (asg-ai/athena-cti-sft-qwen25-14b-v16) rather than off the frozen v12
    # baseline, and adds only a CyberSOCEval letter-set JSON output shape
    # (JS.CSE.TI.* / JS.CSE.MAL.*, 14 templates, ~16.5K rows in
    # ift_data_2026_05_11_v17_cse). Hypothesis: the v16 -> CSE accuracy
    # ceiling (TI 30.63% / Malware 10.69% strict accuracy vs avg_score
    # 58.54% / 45.15%) was bound by output shape, not task knowledge --
    # v16 produces prose+letter answers, CyberSOCEval scores Jaccard on
    # {"correct_answers": ["A","C"]} JSON. v17 tests this without
    # re-introducing any TAA-attribution surface (zero AB.TAA / JS.TAA rows
    # in the manifest by design; chained SFT inherits TAA from v16). Pushed
    # by SFT/autotrain/run_sft_qwen25_14b_v16_plus_v17_cse.sh. See
    # tmpl_gen/templates/05112026/v17_plan.txt and README.md for the
    # falsification criteria (Outcomes A/B/C/D in section 4).
    'athena-cti-sft-qwen25-14b-v17-vllm':                      'asg-ai/athena-cti-sft-qwen25-14b-v17',
    # v17.1: data-fix recovery of v17. Same chained-SFT recipe (base model
    # asg-ai/athena-cti-sft-qwen25-14b-v16, lr 5e-6, 1 epoch, eff_bs 16,
    # packing, cutoff 4096) -- only the corpus changes. v17 was Outcome D
    # (regressed every Athena axis simultaneously) because two engine defects
    # (parser drop of multi-paragraph Question bodies + missing multi-select
    # MCQ shuffler) collapsed the v17 corpus to ~4 distinct correct_answers
    # tuples with the dominant ["A","B"] combo carrying 50.7% of rows; the
    # model could satisfy training loss by emitting "A,B" regardless of input.
    # v17.1 isolates the corpus-quality variable: manifest body byte-identical
    # to v17 except every JS.CSE.* template now declares Shuffle: mcq_multi,
    # rebuilt corpus has uniform per-letter coverage (A/B/C/D/E ~20% each)
    # and the combinatorial-ceiling 26 distinct tuples (ift_data_2026_05_12_
    # v17_1_cse, 18,817 train rows). Pushed by
    # SFT/autotrain/run_sft_qwen25_14b_v16_plus_v17_1_cse.sh. See
    # tmpl_gen/templates/05102026/v17_1_plan.txt and README-17-1.md for the
    # falsification criteria overlay (Outcomes A/B/C/D in section 4).
    'athena-cti-sft-qwen25-14b-v17-1-vllm':                    'asg-ai/athena-cti-sft-qwen25-14b-v17-1',
    # v18: chained three-stage SFT off Qwen2.5-14B-Instruct (mirrors the v17.1
    # chain shape: v12-shape Core, then a v16 TAA Classic refresher, then a
    # v17.1 CSE drill). Each stage publishes its own HF repo so any of the
    # three checkpoints can be benchmarked independently. Pushed by
    # SFT/autotrain/run_sft_qwen25_14b_v18_{core,plus_taa,final}.sh. See
    # tmpl_gen/templates/05112026/v18_plan.txt for the chain rationale and
    # tmpl_gen/templates/MASTER_RESULTS.md for the locked naming convention.
    # Per v18.1 plan §2(5) the cumulative-suffix Stage 2/3 repos were renamed
    # on HF to domain-specific names (the chained TAA / CSE checkpoints are
    # the same artefact regardless of which Core base they ride on, and the
    # v18.1 redo intends to re-chain them off v18.1-core):
    #   asg-ai/...-v18-core-plus-taa     -> asg-ai/...-v18-taa
    #   asg-ai/...-v18-core-plus-taa-cse -> asg-ai/...-v18-cse
    # The HF Hub rename returns HTTP 307 from the old paths so older launchers
    # that still hardcode the cumulative names continue to resolve.
    'athena-cti-sft-qwen25-14b-v18-core-vllm':                 'asg-ai/athena-cti-sft-qwen25-14b-v18-core',
    'athena-cti-sft-qwen25-14b-v18-taa-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v18-taa',
    'athena-cti-sft-qwen25-14b-v18-cse-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v18-cse',
    # v18.1: Core-only redo of v18 Stage 1 after the v18 chain regressed
    # CKT (-15pp vs v8small), RMS (-10pp vs v9_rms), and VSP (-10pp vs v10).
    # MCQ axis reverts to the v8small scenario-only recipe; AB.MCQ.EXT.{MITRE,
    # SEC,GLOSS}.1 KB-flashcard families are dropped and three new scenario
    # MCQ families (AB.MCQ.{7,8,9}) are added against high-cardinality MITRE
    # ATT&CK substrates. RMS consolidates AB.RMS.{4a..4j,5a..5j} back to the
    # v9_rms single-template-per-direction shape; VSP caps to v10's 12K shape.
    # Pushed by SFT/autotrain/run_sft_qwen25_14b_v18p1_{core,plus_taa,final}.sh.
    # See tmpl_gen/templates/05112026/v18_1_plan.txt and README-18-1.md for the
    # diagnosis, deltas, and falsification matrix. The chained TAA / CSE stages
    # reuse the v18 TAA / CSE shards verbatim (recipes unchanged); only the
    # base-model pointer changes (each picks up the v18.1 predecessor instead
    # of the v18 predecessor).
    'athena-cti-sft-qwen25-14b-v18-1-core-vllm':               'asg-ai/athena-cti-sft-qwen25-14b-v18-1-core',
    'athena-cti-sft-qwen25-14b-v18-1-taa-vllm':                'asg-ai/athena-cti-sft-qwen25-14b-v18-1-taa',
    'athena-cti-sft-qwen25-14b-v18-1-cse-vllm':                'asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse',
    # v18.2 candidate (single-shard): v18.1-cse + Stage 4 RMS-replay touch-up.
    # Phase B shard (ift_data_2026_05_11_v18p1_core_b_rms_ate_vsp_rcm) replayed
    # at lr 1e-6 over v18.1-cse to reverse the CSE-stage RMS regression. Retained
    # for regression comparison; superseded as the ship candidate by v18-2.
    # See tmpl_gen/templates/05132026/v18_2_plan.txt and
    # SFT/autotrain/run_sft_qwen25_14b_v18p1_rms_replay.sh.
    'athena-cti-sft-qwen25-14b-v18-1-cse-rms-vllm':            'asg-ai/athena-cti-sft-qwen25-14b-v18-1-cse-rms',
    # v18.2 ship candidate (multi-shard): v18.1-cse + Stage 4 3-shard replay
    # interleaving Phase A (MCQ), Phase B (RMS/ATE/VSP/RCM), and standalone TAA
    # at lr 1e-6 with mix_strategy=interleave_under, probs 0.25/0.40/0.35.
    # Designed to recover RMS while protecting MCQ and TAA Classic (which
    # regressed in the cse-rms single-shard touch-up). See
    # tmpl_gen/templates/05132026/v18_2_plan.txt and
    # SFT/autotrain/run_sft_qwen25_14b_v18p2_multi_replay.sh.
    # Bench (2026-05-14): MCQ 62.33 (target >=70.0; -7.67 pp MISS),
    # RMS 54.72 (target >=55.0; -0.28 pp hairline MISS); other axes all PASS
    # (CSE-TI 41.25 / CSE-Mal 24.14 / VSP 83.87 / ATE 63.20 / RCM 72.55 /
    # TAA combined 47.50 / CM-2K 88.95 / CM-10K 83.94). Superseded by v18-2-1.
    'athena-cti-sft-qwen25-14b-v18-2-vllm':                    'asg-ai/athena-cti-sft-qwen25-14b-v18-2',
    # v18.2.1 (multi-shard, rebalanced): v18.1-cse + Stage 4 3-shard replay
    # with PROBS 0.35/0.45/0.20 and --max-samples 3000/dataset. Iteration of
    # v18.2 to recover MCQ (Phase A 0.25 -> 0.35) and close the 0.28 pp RMS
    # hairline (Phase B 0.40 -> 0.45) while dropping TAA share (0.35 -> 0.20;
    # TAA combined was already PASSING and the standalone TAA short-form
    # pattern likely competed with MCQ's letter-decoder). lr 1e-6, cutoff
    # 16384, packing off (UNCHANGED). See plan §7 and
    # SFT/autotrain/run_sft_qwen25_14b_v18p2p1_multi_replay.sh.
    # Bench (2026-05-14): MCQ 63.17 (target >=70.0; -6.83 pp MISS, +0.84 vs
    # v18.2 = within noise), RMS 50.37 (target >=55.0; -4.63 pp MISS, -4.35
    # vs v18.2 = INVERTED), ATE 62.40 (-0.60 MISS), RCM 66.80 (-0.70 MISS);
    # other axes PASS (VSP 82.65 / TAA combined 47.00 / CSE-TI 41.79 /
    # CSE-Mal 23.48 / CM-2K 89.35 / CM-10K 84.17). Strict regression of the
    # §7.4 gate package (4 fails vs v18.2's 2). Trade ratio |dRMS/dMCQ| 0.45
    # vs v18.2's 0.86 -- half as efficient. Superseded by v18-2-2.
    'athena-cti-sft-qwen25-14b-v18-2-1-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v18-2-1',
    # v18.2.2 ship candidate (multi-shard, "smaller, not different"): v18.1-cse
    # + Stage 4 3-shard replay with the v18.2 prob mix REVERTED (0.25/0.40/0.35)
    # and --max-samples 3000 -> 1500 per dataset (-50% steps vs v18.2.1; -38%
    # vs v18.2). Hypothesis: the v18.2 prob mix was correct and the regression
    # is over-exposure damage from too many Stage 4 steps, not an under-exposure
    # of any one shard. lr 1e-6, cutoff 16384, packing off (UNCHANGED). See
    # plan §8 and SFT/autotrain/run_sft_qwen25_14b_v18p2p2_multi_replay.sh.
    'athena-cti-sft-qwen25-14b-v18-2-2-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v18-2-2',
    # v19 ground-up reproducible 5-stage chain off Qwen2.5-14B-Instruct, built
    # on the 2026_05_15 v19 datasets. Stage outputs (each pushed to its own HF
    # repo so any checkpoint can be benchmarked independently; v19-recalibrate
    # is the published headline):
    #   Stage 1+2 Core (broad re-anchor + axis catalog drill)  -> v19-core
    #   Stage 3   TAA Classic refresher (chains off v19-core)  -> v19-taa
    #   Stage 4   CSE letter-set drill (chains off v19-taa)    -> v19-cse
    #   Stage 5   3-shard interleaved replay (probs 0.33/0.33/0.34, lr 1e-6,
    #             cutoff 16384, packing off; chains off v19-cse)
    #                                                          -> v19-recalibrate
    # Recipe is byte-identical to v18.1+TAA+CSE+v18.2 except the Stage 5 prob
    # mix moves from v18.2's 0.25/0.40/0.35 to equal-weight 0.33/0.33/0.34.
    # Headline gate (v19_plan.txt §5.4, carried from v18.2 §7.4): RMS >= 54.0,
    # MCQ >= 62.0, TAA Classic >= 40.0, CSE-TI >= 34.0, CSE-Malware >= 20.0,
    # ATE >= 62.0, RCM >= 67.5, VSP >= 80.0, CM-2K >= 85.5, CM-10K >= 81.0.
    # Pushed by SFT/autotrain/run_sft_qwen25_14b_v19_{core,taa,cse,recalibrate}.sh.
    # See tmpl_gen/templates/05152026/v19_plan.txt.
    'athena-cti-sft-qwen25-14b-v19-core-vllm':                 'asg-ai/athena-cti-sft-qwen25-14b-v19-core',
    'athena-cti-sft-qwen25-14b-v19-taa-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v19-taa',
    'athena-cti-sft-qwen25-14b-v19-cse-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v19-cse',
    'athena-cti-sft-qwen25-14b-v19-recalibrate-vllm':          'asg-ai/athena-cti-sft-qwen25-14b-v19-recalibrate',
    # v19-recalibrate-v18p2mix: Stage 5 isolation variant. v19-cse + 3-shard
    # interleave with the v18.2 prob mix (0.25/0.40/0.35) at the v18.2-matched
    # step count (--max-samples 2400 -> ~1500 steps, byte-identical to v18.2).
    # Only training-recipe delta vs v19-recalibrate is --probs. Bench delta vs
    # v19-recalibrate isolates the prob-mix contribution; bench delta vs v18-2
    # isolates the v19-cse base-checkpoint contribution. Pushed by
    # SFT/autotrain/run_sft_qwen25_14b_v19_recalibrate_v18p2mix.sh.
    'athena-cti-sft-qwen25-14b-v19-recalibrate-v18p2mix-vllm': 'asg-ai/athena-cti-sft-qwen25-14b-v19-recalibrate-v18p2mix',
    # v20 5-stage chain (Qwen2.5-14B-Instruct + v20 Phase A/B Core, +TAA, +CSE,
    # +Recalibrate). v20 carries the v19 chain topology forward with an axis-
    # density rebalance (raised count_max for ATE / RCM / CSE-TI in the catalog
    # build) and a Stage 5 revert to the v18.2-style interleave probabilities.
    # Headline gate (v20_plan.txt §5.4, carried from v19 §5.4): RMS >= 54.0,
    # MCQ >= 62.0, TAA Classic >= 40.0, CSE-TI >= 34.0, CSE-Malware >= 20.0,
    # ATE >= 62.0, RCM >= 67.5, VSP >= 80.0, CM-2K >= 85.5, CM-10K >= 81.0.
    # Pushed by SFT/autotrain/run_sft_qwen25_14b_v20_{core,taa,cse,recalibrate}.sh
    # (or chained via SFT/autotrain/run_sft_qwen25_14b_v20_chain.sh).
    'athena-cti-sft-qwen25-14b-v20-core-vllm':                 'asg-ai/athena-cti-sft-qwen25-14b-v20-core',
    'athena-cti-sft-qwen25-14b-v20-taa-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v20-taa',
    'athena-cti-sft-qwen25-14b-v20-cse-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v20-cse',
    'athena-cti-sft-qwen25-14b-v20-recalibrate-vllm':          'asg-ai/athena-cti-sft-qwen25-14b-v20-recalibrate',
    # v21 3-stage chain (Qwen2.5-14B-Instruct + v21 Phase A/B Core, +TAA, +CSE)
    # plus an off-plan Recalibrate touch-up (Stage 4). v21 is a byte-identical
    # replay of the v18.1 chain on freshly rebuilt datasets (new date stamp
    # 2026_05_18) with the SAME templates, gates, counts, mixes, and
    # hyperparameters as v18.1; only the dataset filenames and HF push targets
    # change. Goal: recover the v18.1 Core optimum and establish whether the
    # v19/v20 regression is data-build variance vs recipe drift. Headline
    # targets (= v18.1 Core gates): CKT 62.6, RMS 55.6, VSP 76.8, plus the
    # v18.1 TAA / CSE downstream parity. Pushed by
    # SFT/autotrain/run_sft_qwen25_14b_v21_{core,plus_taa,final,recalibrate}.sh
    # (or chained via SFT/autotrain/run_sft_qwen25_14b_v21_chain.sh).
    # Recalibrate is off-plan vs v21_plan.txt §3 (which defines only Core/TAA/
    # CSE for v18.1 parity); included for v19/v20 chain compatibility and only
    # benched if the v21-cse sign-off exposes the same Phase B / catalog
    # erosion v20-cse showed against v18.2.
    # See tmpl_gen/templates/05182026/v21_plan.txt for the replication recipe.
    'athena-cti-sft-qwen25-14b-v21-core-vllm':                 'asg-ai/athena-cti-sft-qwen25-14b-v21-core',
    'athena-cti-sft-qwen25-14b-v21-taa-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v21-taa',
    'athena-cti-sft-qwen25-14b-v21-cse-vllm':                  'asg-ai/athena-cti-sft-qwen25-14b-v21-cse',
    'athena-cti-sft-qwen25-14b-v21-recalibrate-vllm':          'asg-ai/athena-cti-sft-qwen25-14b-v21-recalibrate',
    # v21 chain ported to Qwen2.5-32B-Instruct. Same 4-stage recipe (Core
    # Phase A/B -> TAA -> CSE -> Recalibrate) and same datasets as the Qwen
    # 14B v21 chain; only the base model and HF push targets differ (memory
    # deltas: per_device_batch 2->1 on Phase A, --optim adamw_8bit, --gc on
    # default; recipe LRs/cutoffs/probs held verbatim). Pushed by
    # SFT/autotrain/run_sft_qwen25_32b_v21_{core,plus_taa,final,recalibrate}.sh
    # (or chained via SFT/autotrain/run_sft_qwen25_32b_v21_chain.sh).
    # Scale-up probe of the v21 recipe: whether the 14B v21 chain's 62.3
    # Total ship score reproduces (or exceeds) on the dense 32B base under
    # bit-identical templates/gates/mixes.
    'athena-cti-sft-qwen25-32b-v21-core-vllm':                 'asg-ai/athena-cti-sft-qwen25-32b-v21-core',
    'athena-cti-sft-qwen25-32b-v21-taa-vllm':                  'asg-ai/athena-cti-sft-qwen25-32b-v21-taa',
    'athena-cti-sft-qwen25-32b-v21-cse-vllm':                  'asg-ai/athena-cti-sft-qwen25-32b-v21-cse',
    'athena-cti-sft-qwen25-32b-v21-recalibrate-vllm':          'asg-ai/athena-cti-sft-qwen25-32b-v21-recalibrate',
    # 32B-recipe variant of the off-plan Stage 4 Recalibrate touch-up.
    # Parallel branch off v21-cse alongside qwen25-32b-v21-recalibrate
    # (which uses the 14B recipe verbatim); naming reflects RECIPE
    # PROVENANCE, not chain position. Motivated by the 14B-recipe port
    # failing to recover VSP on Qwen2.5-32B-Instruct (post-cse VSP 78.9
    # -> post-recal VSP 75.7, vs the 14B chain's 72.9 -> 83.1 lift).
    # Three coupled deltas vs the standard recal, holding step count +
    # wall-time constant so the only A/B variable is the recipe: lr 1e-6
    # -> 3e-6 (clear the 32B + adamw_8bit optimizer noise floor),
    # interleave probs 0.25/0.40/0.35 -> 0.15/0.60/0.25 (heavier Phase B
    # share for VSP/RMS catalog re-exposure), --max-samples 2400 -> 3600
    # (preserves the 6000 interleaved rows / ~1500 optimizer steps at
    # the new max(P)=0.60). Base = v21-cse. Pushed by
    # SFT/autotrain/run_sft_qwen25_32b_v21_recal_32b.sh. If VSP recovers
    # without sacrificing the cse-stage gains, this becomes the 32B ship
    # candidate; otherwise v21-cse stays the headline.
    'athena-cti-sft-qwen25-32b-v21-recal-32b-vllm':            'asg-ai/athena-cti-sft-qwen25-32b-v21-recal-32b',
    # v21 recal-32b on Qwen3-30B-A3B-Thinking-2507: DEFAULT on-chain Stage
    # 4 of the Qwen3-MoE v21 chain (Core -> TAA -> CSE -> Recal-32b).
    # Chained off asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-
    # cse. The 32B-tuned recipe (lr 3e-6, probs 0.15/0.60/0.25, max-
    # samples 3600, cutoff 16384, packing off, eff_bs 8, adamw_8bit,
    # Liger) replaces the 14B-recipe Recalibrate on this port because
    # the dense Qwen2.5-32B port confirmed the 14B recipe drifts VSP
    # the wrong way at 32B+ scale under adamw_8bit (78.9 -> 75.7); the
    # Qwen3-MoE parent is peer-scale (30.5B total / 3.3B active per
    # token). Held byte-identical to the Qwen2.5-32B sibling recipe
    # (athena-cti-sft-qwen25-32b-v21-recal-32b: Total 66.3, Weighted
    # 65.3) so the Qwen3-MoE outcome is directly comparable.
    # Template: qwen3 (native); run_train.sh defaults --enable_thinking
    # to True so the reasoning template injects <think>\n\n</think> into
    # the loss/response_ids on every sample without a <think> block (i.e.
    # all our CTI rows). The model is trained to autonomously emit an
    # empty 6-token thought followed by the answer for CTI prompts -- the
    # thinking apparatus stays alive as a generation path so OOD (non-
    # CTI) reasoning behaviour can still resurface. Pushed by
    # SFT/autotrain/run_sft_qwen3_30b_a3b_thinking_v21_recal_32b.sh on
    # 8xB300 (no offload), or by the matching _chain.sh wrapper at the
    # end of the TAA -> CSE -> Recal-32b sequence. The '-no-think' alias
    # suffix forwards chat_template_kwargs.enable_thinking=False at
    # serve time as belt-and-suspenders (in case a checkpoint drifts
    # off the empty-thought pattern) and -- more importantly --
    # suppresses VLLMModel's '-thinking' 8192-token floor so the per-
    # task caps in TASK_MAX_NEW_TOKENS (MCQ=128, RCM/RMS/TAA=256) apply
    # correctly. This alias is the Qwen3-MoE v21 chain headline; the
    # 14B-recipe v21-recalibrate alias below is retained for off-chain
    # A/B against it.
    'athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b-no-think-vllm': 'asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recal-32b',
    # Qwen3-30B-A3B-Thinking-2507 full v21 chain (Core -> TAA -> CSE ->
    # Recal-32b), MoE port of the Qwen2.5-32B v21 chain above. Stages
    # 1-3 hold datasets / cutoffs / packing / LRs / eff_bs byte-
    # identical to the 32B chain (only base model + template qwen ->
    # qwen3 + HF push targets differ); Stage 4 ships the 32B-tuned
    # recal-32b recipe instead of the 14B-recipe Recalibrate (see the
    # recal-32b alias comment above for the rationale and the empty-
    # thought training semantic). The 14B-recipe v21-recalibrate alias
    # below is retained for off-chain A/B work against the on-chain
    # recal-32b ship-candidate. Pushed by
    # SFT/autotrain/run_sft_qwen3_30b_a3b_thinking_v21_{core,plus_taa,
    # final,recal_32b}.sh (or chained via the matching _chain.sh
    # wrapper, which now ends at recal_32b). Scale-up + sparse-arch
    # probe of the v21 recipe: whether the dense-32B v21 chain headline
    # reproduces (or exceeds) on the 30.5B-total / 3.3B-active MoE base
    # under bit-identical templates/gates/mixes with thinking-on
    # training semantics.
    'athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-core-no-think-vllm':         'asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-core',
    'athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa-no-think-vllm':          'asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-taa',
    'athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-no-think-vllm':          'asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse',
    # Same HF repo as the -no-think alias above, served WITHOUT
    # chat_template_kwargs.enable_thinking=False -- i.e. the chat template
    # is allowed to inject its <think> preamble and the alias name
    # ('thinking' substring, no '-no-think') opts into VLLMModel's 8192-
    # token thinking floor so traces don't truncate at the per-task cap.
    # The v21 SFT trained this checkpoint with the empty-thought pattern
    # (run_train.sh defaults --enable_thinking True and the template
    # injects <think>\n\n</think> on every CTI row, so the model learns to
    # emit an empty trace then jump straight to the answer); under
    # thinking-on serving the model should keep doing that and the CTI
    # numbers should land within noise of the -no-think alias above. This
    # alias exists to verify that empirically and to provide a matched-
    # conditions comparison against qwen3-30b-a3b-thinking-2507-vllm (the
    # untrained base under the same inference path), isolating the SFT's
    # contribution under identical serving semantics. Pair with
    # EXTRA_SERVE_FLAGS="--reasoning-parser qwen3" so any non-empty trace
    # routes to reasoning_content and the bench-visible content field
    # stays clean.
    'athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-vllm':                   'asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse',
    # Off-chain 14B-recipe Stage-4 A/B variant (retained for comparison
    # against the on-chain v21-recal-32b ship-candidate above; not on
    # the default chain path on Qwen3-MoE).
    'athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recalibrate-no-think-vllm':  'asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-recalibrate',
    # v21 chain ported to Llama-3.1-8B-Instruct. Same 4-stage recipe (Core
    # Phase A/B -> TAA -> CSE -> Recalibrate) and same datasets as the Qwen
    # 14B v21 chain; only the base model, --template (qwen -> llama3), and
    # HF push targets differ. Pushed by
    # SFT/autotrain/run_sft_llama31_8b_v21_{core,plus_taa,final,recalibrate}.sh
    # (or chained via SFT/autotrain/run_sft_llama31_8b_v21_chain.sh).
    # Cross-architecture probe of the v21 recipe: whether the Qwen 14B
    # Stage-3-CSE-erodes-VSP / Stage-4-recovers signature reproduces on the
    # smaller Llama-3.1-8B architecture is the primary open question.
    'athena-cti-sft-llama31-8b-v21-core-vllm':                 'asg-ai/athena-cti-sft-llama31-8b-v21-core',
    'athena-cti-sft-llama31-8b-v21-taa-vllm':                  'asg-ai/athena-cti-sft-llama31-8b-v21-taa',
    'athena-cti-sft-llama31-8b-v21-cse-vllm':                  'asg-ai/athena-cti-sft-llama31-8b-v21-cse',
    'athena-cti-sft-llama31-8b-v21-recalibrate-vllm':          'asg-ai/athena-cti-sft-llama31-8b-v21-recalibrate',
    # v21 chain ported to fdtn-ai/Foundation-Sec-8B-Instruct (Cisco SFT+RLHF
    # cybersecurity model on the Llama-3.1-8B architecture). Same 4-stage
    # recipe and same datasets as the Llama-3.1-8B v21 chain; the only
    # difference is the starting checkpoint -- a domain-anchored Cisco
    # security SFT rather than generic Llama-3.1-8B-Instruct. --template
    # stays llama3 (Foundation shares the Llama-3.1 architecture and LF
    # rewrites the saved chat_template at SFT time, overriding Foundation's
    # custom '<|system|>'/'<|user|>'/'<|assistant|>' template). Pushed by
    # SFT/autotrain/run_sft_foundation_8b_v21_{core,plus_taa,final,recalibrate}.sh
    # (or chained via SFT/autotrain/run_sft_foundation_8b_v21_chain.sh).
    # The post-SFT checkpoints serve via the standard llama3 jinja (vLLM
    # picks it up from the saved tokenizer_config.json); no chat-template
    # override needed at serve time.
    'athena-cti-sft-foundation-8b-v21-core-vllm':              'asg-ai/athena-cti-sft-foundation-8b-v21-core',
    'athena-cti-sft-foundation-8b-v21-taa-vllm':               'asg-ai/athena-cti-sft-foundation-8b-v21-taa',
    'athena-cti-sft-foundation-8b-v21-cse-vllm':               'asg-ai/athena-cti-sft-foundation-8b-v21-cse',
    'athena-cti-sft-foundation-8b-v21-recalibrate-vllm':       'asg-ai/athena-cti-sft-foundation-8b-v21-recalibrate',
    # v21 chain ported to Gemma 4 31B-it. Same 4-stage recipe (Core Phase
    # A/B -> TAA -> CSE -> Recalibrate) and same datasets as the Qwen
    # 14B v21 chain; only the base model, --template (qwen -> gemma4),
    # --flash_attn (auto -> sdpa; head_dim=512 / FA #2427 pending), and
    # HF push targets differ. Pushed by
    # SFT/autotrain/run_sft_gemma4_31b_v21_{core,plus_taa,final,recalibrate}.sh
    # (or chained via SFT/autotrain/run_sft_gemma4_31b_v21_chain.sh).
    # Cross-architecture + scale probe of the v21 recipe: the 8B (Llama)
    # and 31B (Gemma) ports together bracket the scale axis; whether the
    # Qwen 14B Stage-3-CSE-erodes-VSP / Stage-4-recovers signature
    # reproduces is the primary open question.
    # vLLM-serve caveats for the trained checkpoints (text-only inputs):
    #   --limit-mm-per-prompt image=0 (skip vision-encoder KV pre-alloc),
    #   weights ~62GB bf16 -> tp=2 on H100 80GB (~31GB/GPU) or tp=1 on
    #   B300 (288GB/GPU).
    'athena-cti-sft-gemma4-31b-v21-core-vllm':                 'asg-ai/athena-cti-sft-gemma4-31b-v21-core',
    'athena-cti-sft-gemma4-31b-v21-taa-vllm':                  'asg-ai/athena-cti-sft-gemma4-31b-v21-taa',
    'athena-cti-sft-gemma4-31b-v21-cse-vllm':                  'asg-ai/athena-cti-sft-gemma4-31b-v21-cse',
    'athena-cti-sft-gemma4-31b-v21-recalibrate-vllm':          'asg-ai/athena-cti-sft-gemma4-31b-v21-recalibrate',
    # HF Inference Providers route. Custom community fine-tunes are not in the
    # default Together/Fireworks/Novita/etc. catalogs; this alias only resolves
    # if the model is exposed via an HF Inference Endpoint or the legacy
    # 'hf-inference' warm tier picks it up. Set HF_INFERENCE_ENDPOINT_URL to
    # bypass auto-routing and target a dedicated endpoint URL directly.
    'athena-cti-sft-llama31-8b-abaligned-v8-hf':               'asg-ai/athena-cti-sft-llama31-8b-abaligned-v8',
    'athena-cti-sft-llama31-8b-abaligned-lora-vllm': 'asg-ai/athena-cti-sft-llama31-8b-abaligned-lora',
    'minerva-llama31-8b-vllm':                 'asg-ai/minerva-llama3.1-8b',
    # Cisco Foundation-Sec-8B-Reasoning emits <think>...</think> traces; serve
    # with `--reasoning-parser minimax_m2 --trust-remote-code` so vLLM strips
    # the trace into `reasoning_content` and leaves clean answers in `content`.
    'foundation-8b-reasoning-vllm':            'fdtn-ai/Foundation-Sec-8B-Reasoning',
    # Cisco Foundation-Sec-8B-Instruct ships its own jinja chat template
    # ('<|system|>'/'<|user|>'/'<|assistant|>' markers, baked-in Cisco system
    # prompt when no system message is provided). vLLM auto-uses it; do NOT
    # pass `--chat-template` overrides. No reasoning parser needed.
    'foundation-8b-instruct-vllm':             'fdtn-ai/Foundation-Sec-8B-Instruct',
    'foundation-8b-vllm':                      'fdtn-ai/Foundation-Sec-8B',
}


def alias_to_safe_name(alias: str) -> str:
    """Sanitize an alias for use as a cache directory / filename component.

    Returns ``alias.replace('/', '_')`` -- the alias verbatim, NOT the HF
    repo id it maps to via ``model_mapping``. All benchmark cache paths
    (``responses/<safe>/<task>/<task>_..._<safe>_response.<ext>``) and
    the per-model summary aggregation directory MUST go through this
    helper so two aliases that resolve to the same HF repo get isolated
    caches.

    Concrete case this prevents: ``qwen3-30b-a3b-thinking-2507-vllm``
    (thinking-on, 8192 floor, optionally ``--reasoning-parser qwen3``)
    and ``qwen3-30b-a3b-thinking-2507-no-think-vllm`` (per-request
    ``enable_thinking=False``, per-task caps) both resolve to the same
    HF repo ``Qwen/Qwen3-30B-A3B-Thinking-2507`` yet produce very
    different MMLU-Pro / CTI scores. Keying caches by HF id collided
    them -- the second alias to run silently re-scored the first's CSV
    on resume and ``--mode overwrite`` deleted nothing because the
    resolver pointed at the empty alias-keyed slot. Keying by alias
    gives each serving semantic its own cache slot by construction so
    the matched-conditions A/B works without manual ``mv`` dances.

    The matching shell-side convention in ``run_benchmark.sh`` and
    ``run_foundation_8b_baselines.sh`` is the ``SAFE_NAME`` /
    ``SAFE_ALIAS`` bash variable -- ``MODEL_NAME//\\//_``, byte-
    identical to this function's output. Keep both ends in sync.
    """
    return alias.replace("/", "_")


# --- Centralized Helpers ---
def check_disk_space(model_id):
    stat = shutil.disk_usage(workspace_cache)
    available_gb = stat.free / (1024 ** 3)
    required_gb = 5
    if '70b' in model_id.lower(): required_gb = 150
    elif '32b' in model_id.lower(): required_gb = 70
    elif '20b' in model_id.lower(): required_gb = 45
    elif '14b' in model_id.lower(): required_gb = 32
    elif '13b' in model_id.lower(): required_gb = 30
    elif '8b' in model_id.lower(): required_gb = 20
    elif '3b' in model_id.lower(): required_gb = 10
    print(f"Available disk space in {workspace_cache}: {available_gb:.2f} GB")
    print(f"Estimated required space for {model_id}: {required_gb} GB")
    if available_gb < required_gb:
        print(f"WARNING: Not enough disk space for {model_id}!")
        print(f"Required: {required_gb} GB, Available: {available_gb:.2f} GB")
        print("Consider cleaning up the cache directory or using a smaller model.")
        return False
    return True

def get_system_prompt(task):
    if task in ["glue", "superglue"]:
        return "You are an expert classification assistant."
    elif task == "mmlu":
        return "You are a knowledgeable multiple-choice assistant."
    # CTI tasks: CTIBench (bare names) and AthenaBench (`athena-` prefix), plus
    # CyberMetric. Previously only the CTIBench names were matched, so every
    # `athena-*` sweep silently ran with `sys_prompt=None`.
    elif task in ["ate", "rcm", "vsp", "taa", "mcq",
                  "athena-ate", "athena-rcm", "athena-rms",
                  "athena-taa", "athena-taa-canonical",
                  "athena-vsp", "athena-mcq",
                  "cybermetric"]:
        return "You are a cybersecurity expert specializing in cyberthreat intelligence."
    return None


# ----------------- Base Model ----------------- #
class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, question, task=None, cleanup_after=False, use_web_search=False, temperature=0, **kwargs):
        raise NotImplementedError

# ----------------- OpenAI Model ----------------- #
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# OpenAI aliases routed through the v1/responses endpoint with optional
# reasoning_effort (low|medium|high|xhigh). Keep in sync with the
# REASONING_FAMILIES set in test/inference.py and run_benchmark.sh.
REASONING_MODELS = {'gpt5.2', 'gpt5.5', 'gpt5.5-pro'}

class OpenAIModel(BaseModel):
    def __init__(self, model_name, api_key=None):
        super().__init__(model_name)
        if OpenAI is None:
            raise ImportError("openai package required")
        self.api_key = api_key or openai_api_key
        self.client = OpenAI(api_key=self.api_key)
        self.reasoning_effort = None

    def generate(self, question, task=None, cleanup_after=False, use_web_search=False, temperature=0,
                 reasoning_effort=None, **kwargs):
        sys_prompt = get_system_prompt(task)
        model_id = model_mapping[self.model_name]

        if self.model_name == 'gpt5':
            # Determine if grounding should be applied
            use_grounding = task == "cve" and use_web_search

            # If grounding is needed, enable the web search tool
            tools = [{"type": "web_search_preview"}] if use_grounding else []

            # Generate the response
            full_input = f"{sys_prompt}\n\n{question}" if sys_prompt else question
            resp = self.client.responses.create(
                model="gpt-5",
                input=full_input,
                tools=tools
            )

            # Track tokens
            usage = getattr(resp, "usage", None)
            if usage:
                input_tokens = getattr(usage, "input_tokens", 0) 
                output_tokens = getattr(usage, "output_tokens", 0)
            else:
                input_tokens = 0
                output_tokens = 0

            # Add tokens and compute costs
            add_tokens(self.model_name,input_tokens,output_tokens,grounding=use_grounding)
            return resp.output_text

        elif self.model_name in REASONING_MODELS:
            use_grounding = task == "cve" and use_web_search
            tools = [{"type": "web_search_preview"}] if use_grounding else []

            full_input = f"{sys_prompt}\n\n{question}" if sys_prompt else question
            api_kwargs = {
                "model": model_id,
                "input": full_input,
                "tools": tools,
            }

            # Reasoning effort: none (default), low, medium, high, xhigh
            effort = reasoning_effort or self.reasoning_effort
            if effort:
                api_kwargs["reasoning"] = {"effort": effort}

            resp = self.client.responses.create(**api_kwargs)

            usage = getattr(resp, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

            add_tokens(self.model_name, input_tokens, output_tokens, grounding=use_grounding)
            return resp.output_text

        else:
            messages = []
            if sys_prompt:
                messages.append({'role': 'system', 'content': sys_prompt})
            messages.append({'role': 'user', 'content': question})
            resp = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                #max_tokens=2048,
                top_p=1.0
            )
            # Track tokens (grounding=False)
            use_grounding = False 
            usage = getattr(resp, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                add_tokens(self.model_name, input_tokens, output_tokens, grounding=use_grounding)

            return resp.choices[0].message.content


# ----------------- Gemini Model ----------------- #
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

class GeminiModel(BaseModel):
    def __init__(self, model_name, api_key=None):
        super().__init__(model_name)
        if genai is None or types is None:
            raise ImportError("google-genai package required")
        
        # Ensure API key is provided
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or passed as argument")

        # Initialize client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_mapping[model_name]

    def generate(self, question, task=None, cleanup_after=False, use_web_search=False, temperature=0, **kwargs):
        sys_prompt = get_system_prompt(task)
        full_prompt = f"{sys_prompt}\n\n{question}" if sys_prompt else question

        # Determine if grounding is used
        use_grounding = self.model_name == "gemini-2.5-flash" and task == "cve" and use_web_search
        if use_grounding:
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=temperature,
                top_p=1.0,
                #max_output_tokens=2048,
            )

        else:
            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=1.0,
                #max_output_tokens=2048,
            )

        # Generate response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=config,
        )

        if hasattr(response, "usage_metadata"):
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            thoughts_tokens = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
            add_tokens(self.model_name, input_tokens, output_tokens + thoughts_tokens, grounding=use_grounding)

        return response.text


# ----------------- HuggingFace Inference Providers (hosted) ----------------- #
class HFInferenceModel(BaseModel):
    """Run inference against HuggingFace Inference Providers (hosted API).

    Uses the OpenAI-compatible chat.completions endpoint on
    https://router.huggingface.co/v1 with server-side auto-routing to a
    provider (Together, Fireworks, Sambanova, etc.). No local GPU used.

    Requires HUGGINGFACE_TOKEN (with inference scope) and billing enabled
    on the HF account, or a Pro subscription with included credits.

    Model keys use the '-hf' suffix convention so the same base model can
    coexist with its local-GPU variant in model_mapping (e.g. 'qwen3-14b'
    vs 'qwen3-14b-hf').
    """

    _TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, model_name):
        super().__init__(model_name)
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for HF Inference Providers. "
                "Install or upgrade: pip install -U 'huggingface_hub>=0.34'"
            ) from e
        if not hf_token:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN is not set. HF Inference Providers "
                "requires an HF token with inference scope and billing enabled."
            )
        self.hf_model_id = model_mapping.get(model_name, model_name)
        # HF_INFERENCE_ENDPOINT_URL points InferenceClient at a dedicated
        # endpoint (https://<id>.<region>.aws.endpoints.huggingface.cloud) and
        # bypasses the router's catalog lookup. Required for custom community
        # fine-tunes (e.g. asg-ai/*) that are not in any provider's preloaded
        # catalog. Per-model overrides take precedence so multiple endpoints
        # can coexist in the same .env.
        per_model_env = (
            f"HF_INFERENCE_ENDPOINT_URL_{model_name.replace('-', '_').upper()}"
        )
        endpoint_url = (
            os.getenv(per_model_env)
            or os.getenv("HF_INFERENCE_ENDPOINT_URL")
        )
        if endpoint_url:
            self.client = InferenceClient(model=endpoint_url, api_key=hf_token)
            print(f"HF Inference client ready for {self.hf_model_id} "
                  f"(dedicated endpoint: {endpoint_url})")
        else:
            # provider=None => server-side auto-routing on router.huggingface.co/v1
            self.client = InferenceClient(api_key=hf_token)
            print(f"HF Inference client ready for {self.hf_model_id} (provider=auto)")

    def generate(self, question, task=None, cleanup_after=False, use_web_search=False,
                 temperature=0.0, max_new_tokens=2048, **kwargs):
        import time
        sys_prompt = get_system_prompt(task)
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": question})

        # Per-model extra_body knobs. Kimi-K2.6 ships with thinking mode on by
        # default; in that mode the OpenAI-compat path on HF Router leaks the
        # (truncated) <think> trace into `content` instead of returning the
        # final answer. Moonshot's documented API contract (mirrored by Novita
        # and the Moonshot direct endpoint) accepts a top-level `thinking`
        # field; vLLM/SGLang use `chat_template_kwargs.enable_thinking`
        # instead. The `thinking` shape is the one HF Router's K2.6 provider
        # honors; the `chat_template_kwargs` shape gets a 400 from the strict
        # OpenAI validator on Together/Fireworks-flavored providers.
        extra_body = {}
        if "kimi-k2" in self.hf_model_id.lower():
            extra_body["thinking"] = {"type": "disabled"}

        # Raise the max_tokens floor for hosted inference. The per-task cap in
        # TASK_MAX_NEW_TOKENS (e.g. MCQ=128, RMS/RCM=256) is tuned for terse
        # local models that emit just an answer letter and stop on EOS. Hosted
        # models that respond with an analytical preamble before the answer
        # (Gemma 4 31B IT, Kimi K2.6 with thinking, several reasoning-tuned
        # variants on HF Router) get truncated mid-analysis at 128/256 tokens
        # and never emit the final answer, collapsing MCQ accuracy to <25%.
        # Hosted providers stream and stop at EOS, so the floor only costs
        # latency on rows where the model would have rambled past the cap
        # anyway, which is exactly the case where we need the extra room.
        effective_max_tokens = max(int(max_new_tokens or 0), 2048)

        last_err = None
        for attempt in range(5):
            try:
                create_kwargs = dict(
                    model=self.hf_model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                )
                if extra_body:
                    create_kwargs["extra_body"] = extra_body
                resp = self.client.chat.completions.create(**create_kwargs)
                choice = resp.choices[0] if resp.choices else None
                cmsg = choice.message if choice else None
                content = (cmsg.content if cmsg else "") or ""
                # Fallback: some providers split a reasoning model's output
                # into message.content (final answer) + message.reasoning_content
                # (trace). If content is empty but reasoning_content is set,
                # prefer the reasoning text over an empty string so the row is
                # at least scoreable rather than silently dropped.
                if not content and cmsg is not None:
                    rc = getattr(cmsg, "reasoning_content", None) or ""
                    content = rc or ""
                # Track per-request usage so the sweep summary can surface
                # token totals + cost for HF Router models. add_tokens is
                # graceful for models without a PRICING_PER_1K entry (warns
                # once, accumulates tokens, reports cost=0). Mirrors the
                # OpenAIModel.generate pattern.
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    in_tok = getattr(usage, "prompt_tokens", 0) or 0
                    out_tok = getattr(usage, "completion_tokens", 0) or 0
                    add_tokens(self.model_name, in_tok, out_tok, grounding=False)
                return content
            except Exception as e:
                last_err = e
                status = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None)
                msg = str(e)
                retriable = status in self._TRANSIENT_HTTP or any(
                    s in msg.lower() for s in ("timeout", "rate limit", "temporarily", "connection"))
                if not retriable or attempt == 4:
                    raise
                backoff = min(2 ** attempt, 30)
                print(f"HF Inference transient error ({status or 'err'}) on "
                      f"{self.hf_model_id}, retry {attempt+1}/5 in {backoff}s: {msg[:200]}")
                time.sleep(backoff)
        # Unreachable but keeps type-checkers happy
        raise last_err if last_err else RuntimeError("HF Inference: unknown failure")


# ----------------- Local vLLM server (OpenAI-compatible) ----------------- #
class VLLMModel(BaseModel):
    """Run inference against a local `vllm serve` process.

    vLLM exposes an OpenAI-compatible HTTP API; we talk to it via the openai
    SDK by pointing ``base_url`` at the vLLM server (default
    ``http://localhost:8000/v1``) and supplying a dummy api_key. No local
    model load happens in this process: the server keeps weights resident
    and serves concurrent requests, so --batch N in inference.py maps to N
    in-flight HTTP requests from the benchmark's ThreadPoolExecutor.

    Model keys use the '-vllm' suffix convention so the same HF repo id can
    coexist with local-transformers and HF-Inference variants in
    model_mapping (e.g. 'athena-cti-cpt-llama31-8b-v1' vs
    '...-v1-vllm' vs a future '...-v1-hf').

    Configuration via env:
        VLLM_BASE_URL   default http://localhost:8000/v1
        VLLM_API_KEY    default "EMPTY" (vLLM ignores but the SDK requires non-empty)

    Chat template: vLLM uses whatever chat_template is on the HF repo. For
    base (non-Instruct) models without a chat template, start the server
    with `--chat-template <path>` or the chat.completions endpoint errors.
    """

    _TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, model_name):
        super().__init__(model_name)
        if OpenAI is None:
            raise ImportError(
                "openai package required for VLLMModel. "
                "Install into the benchmark env: pip install openai"
            )
        self.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY") or "EMPTY"
        self.hf_model_id = model_mapping.get(model_name, model_name)
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        # Hybrid-thinking models (Qwen3 family, Foundation-Sec-8B-Reasoning,
        # etc.) emit a <think> trace by default. For short-answer benchmarks
        # the trace consumes the generation budget and the final answer never
        # makes it into `content`. Aliases carrying the '-no-think' substring
        # opt out per-request via `chat_template_kwargs.enable_thinking=False`,
        # which Qwen's chat template (and the SGLang/vLLM OpenAI-compat layer)
        # honor by skipping the reasoning preamble entirely.
        self.disable_thinking = "no-think" in model_name.lower()
        # Pure-thinking models (Qwen3-*-Thinking-2507, DeepSeek-R1, etc.)
        # ALWAYS emit a <think>...</think> trace -- there is no off switch.
        # The trace itself counts against max_tokens even when a server-side
        # reasoning-parser routes it to `reasoning_content` for the client.
        # The per-task caps in TASK_MAX_NEW_TOKENS (MCQ=128, RCM/RMS/TAA=256)
        # are tuned for terse non-thinking models and will truncate every row
        # mid-trace, collapsing accuracy to below random. Detect via the
        # explicit 'thinking' substring in the alias (matches the upstream
        # repo naming convention) and raise the per-request floor to 8192.
        # Excludes '-no-think' aliases (which mean the opposite).
        name_lc = model_name.lower()
        self.is_thinking_model = "thinking" in name_lc and "no-think" not in name_lc
        if self.is_thinking_model and self.disable_thinking:
            # Can't be both; -no-think wins by intent (it's the explicit override).
            self.is_thinking_model = False
        flags = []
        if self.disable_thinking:
            flags.append("thinking=disabled")
        if self.is_thinking_model:
            flags.append("thinking=floor8192")
        print(f"vLLM client ready for {self.hf_model_id} (base_url={self.base_url}"
              f"{', ' + ', '.join(flags) if flags else ''})")

    # vLLM 400 message has two forms depending on version:
    #   old: "maximum context length is N tokens. However, you requested
    #        M output tokens and your prompt contains at least P input
    #        tokens, for a total of at least N+1 tokens."
    #   new: "...your prompt contains P input tokens..." (no "at least"
    #        hedge; vLLM 0.7+ reports the exact count once it tokenizes).
    # The "at least" word is therefore optional. Without this the recovery
    # path silently misses the modern format and every overflow row
    # crashes with a full traceback instead of a clean one-line "prompt
    # exceeds served context" log + drop.
    _CTX_OVERFLOW_RE = re.compile(
        r"maximum context length is (\d+) tokens.*?prompt contains (?:at least )?(\d+) input tokens",
        re.DOTALL,
    )
    # The "input_tokens >= P" figure in vLLM's error is NOT the true prompt
    # size; it is a derived lower bound P = N - max_tokens + 1 that just
    # tightens as we shrink max_tokens. Iterative bisection is therefore
    # provably non-convergent: each retry only re-confirms the same overflow
    # at a tighter bound. Strategy is one-shot instead -- on the first 400,
    # drop max_tokens straight to the floor (so the generation slice is the
    # smallest the bench can still post-process). If that still overflows,
    # the prompt itself is >= ctx and no retry can save the row; bail.
    _CTX_OVERFLOW_FLOOR = 64
    _MIN_GENERATION_BUDGET = 32
    # Thinking-mode floor: trace + answer must both fit in max_tokens. The
    # MCQ/short-answer per-task caps (128-512) are too tight by an order of
    # magnitude; bump to a uniform 8192 floor when the alias declares
    # thinking mode. Server-side --reasoning-parser routes the trace to
    # `reasoning_content`, so the bench-visible `content` field is still
    # just the final answer; the floor only affects the generation budget,
    # not what the bench sees.
    _THINKING_MIN_TOKENS = 8192

    def generate(self, question, task=None, cleanup_after=False, use_web_search=False,
                 temperature=0.0, max_new_tokens=2048, **kwargs):
        import time
        sys_prompt = get_system_prompt(task)
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": question})

        effective_max = max_new_tokens
        if self.is_thinking_model:
            effective_max = max(int(effective_max or 0), self._THINKING_MIN_TOKENS)
        shrunk = False  # one-shot drop-to-floor on context overflow per call
        transient_budget = 5
        last_err = None
        # vLLM exposes Qwen3's hybrid-thinking switch through extra_body so we
        # don't have to mutate the prompt; the SDK forwards it as JSON and
        # vLLM surfaces it to the chat template as
        # chat_template_kwargs.enable_thinking.
        create_extra = {}
        if self.disable_thinking:
            create_extra["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        # +1 outer slot to cover the post-shrink retry on top of the transient
        # budget. Shrink is one-shot so this caps total attempts at 7.
        for attempt in range(transient_budget + 2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.hf_model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=effective_max,
                    top_p=1.0,
                    **create_extra,
                )
                # No add_tokens() call here: local vLLM is free and has no
                # entry in api_usage.PRICING_PER_1K. HFInferenceModel does the
                # same. Token counts are still available via resp.usage if a
                # future caller wants to log them.
                choice = resp.choices[0] if resp.choices else None
                content = (choice.message.content if choice and choice.message else "") or ""
                return content
            except Exception as e:
                last_err = e
                status = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None)
                msg = str(e)
                # Context-overflow recovery: drop max_tokens straight to the
                # floor on the first 400 and retry once. The "input_tokens >=
                # P" figure in the error is just N - max_tokens + 1 (a derived
                # lower bound that tightens as we shrink), so iterative
                # shrinking provably can't reveal the true prompt size --
                # one-shot to floor is the only convergent strategy. If the
                # retry also overflows, the prompt itself is >= ctx and no
                # max_tokens value can save the row; bail with a short notice
                # and tag the exception so get_single_prediction can suppress
                # its full traceback.
                if status == 400:
                    m = self._CTX_OVERFLOW_RE.search(msg)
                    if m is not None:
                        ctx_max = int(m.group(1))
                        prompt_lb = int(m.group(2))
                        if not shrunk and self._CTX_OVERFLOW_FLOOR < effective_max:
                            print(f"vLLM ctx-shrink {self.hf_model_id}: "
                                  f"prompt>={prompt_lb}, ctx={ctx_max}, "
                                  f"max_tokens {effective_max}->{self._CTX_OVERFLOW_FLOOR} "
                                  f"(one-shot to floor)",
                                  file=sys.stderr)
                            effective_max = self._CTX_OVERFLOW_FLOOR
                            shrunk = True
                            continue  # immediate retry, no backoff
                        # Either we already shrunk and still overflowed, or
                        # the floor is already >= effective_max. Either way
                        # the prompt is genuinely larger than ctx; mark the
                        # exception so the caller can log a one-liner instead
                        # of a 30-line stack trace per offending row.
                        print(f"vLLM ctx-overflow giving up on "
                              f"{self.hf_model_id}: prompt>={prompt_lb} "
                              f"in ctx={ctx_max} with max_tokens={effective_max}; "
                              f"prompt exceeds served context",
                              file=sys.stderr)
                        try:
                            setattr(e, "_vllm_ctx_overflow", True)
                        except Exception:
                            pass
                        raise
                retriable = status in self._TRANSIENT_HTTP or any(
                    s in msg.lower() for s in ("timeout", "rate limit", "temporarily", "connection"))
                if not retriable or attempt == (transient_budget + 1):
                    raise
                backoff = min(2 ** attempt, 30)
                print(f"vLLM transient error ({status or 'err'}) on "
                      f"{self.hf_model_id}, retry {attempt+1} in {backoff}s: {msg[:200]}")
                time.sleep(backoff)
        raise last_err if last_err else RuntimeError("vLLM: unknown failure")


# Bundled chat templates for base models that ship without `tokenizer.chat_template`.
# Mirrors the auto-apply logic in SFT/test/utils/serve_vllm.sh so the local
# transformers path and the vLLM path produce comparable baselines.
_CHAT_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "utils", "chat_templates",
)
_BUNDLED_CHAT_TEMPLATES = (
    # (substring matched against model_id.lower(), template filename)
    ("llama-3", "llama3.jinja"),
    ("llama3",  "llama3.jinja"),
)

def _maybe_apply_bundled_chat_template(tokenizer, model_id):
    """If the tokenizer has no chat_template and the repo id matches a known
    base-model family, load the bundled jinja template. No-op otherwise."""
    if getattr(tokenizer, "chat_template", None):
        return
    mid = model_id.lower()
    for needle, fname in _BUNDLED_CHAT_TEMPLATES:
        if needle in mid:
            path = os.path.join(_CHAT_TEMPLATE_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
                print(f" Applied bundled chat template '{fname}' to {model_id} "
                      f"(tokenizer had none)")
            except OSError as e:
                print(f" WARN: bundled chat template '{fname}' not loadable: {e}")
            return


# ----------------- HuggingFace Model ----------------- #
class HuggingFaceModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = None
        self.tokenizer = None
        self.current_model_id = None
        self.load_model(model_mapping.get(model_name, model_name))

    def load_model(self, model_id):
        if self.current_model_id == model_id and self.model is not None:
            print(f"Model {model_id} already loaded in memory")
            return

        print(f"Loading HuggingFace model: {model_id}")
        if not check_disk_space(model_id):
            print("Low disk space, but proceeding with load...")

        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        print(f"Loading tokenizer for {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
            cache_dir=workspace_cache,
            local_files_only=False,  # Allow downloading if not cached
        )
        print("Tokenizer loaded and cached")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Base models (e.g. meta-llama/Llama-3.1-8B) ship without a chat_template,
        # which causes generate() to fall through to a raw-prompt path. On
        # chat-shaped CTI prompts the base model then emits <|end_of_text|> as
        # token 1 and produces an empty response. Apply the bundled template
        # when the family matches; instruct/chat-templated repos are untouched.
        _maybe_apply_bundled_chat_template(self.tokenizer, model_id)

        # Model loading settings
        loading_kwargs = {
            "token": hf_token,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "cache_dir": workspace_cache,
            "local_files_only": False,  # Allow downloading if not cached
        }

        if "70b" in model_id.lower():
            loading_kwargs["torch_dtype"] = torch.float16
            print("Loading 70B model in float16")
        else:
            loading_kwargs["torch_dtype"] = torch.bfloat16
            print("Loading model in bfloat16")

        print(f"Loading model {model_id}... (this may take time on first run)")
        # Attention implementation selection.
        # Default is "sdpa" (PyTorch's scaled-dot-product attention) because it
        # dispatches to flash / mem-efficient kernels under the hood without the
        # transformers x flash-attn version mismatch that surfaces on Qwen2-based
        # models (DeepHat, Qwen2.5-*, etc.). Set ATHENA_ATTN_IMPL=flash_attention_2
        # to opt back into FA2 when a compatible flash-attn build is installed.
        preferred_impl = os.environ.get("ATHENA_ATTN_IMPL", "sdpa").strip() or "sdpa"
        impl_order = [preferred_impl]
        for fallback in ("sdpa", "eager"):
            if fallback not in impl_order:
                impl_order.append(fallback)

        last_err = None
        for impl in impl_order:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    attn_implementation=impl,
                    **loading_kwargs,
                )
                print(f" Attention implementation active: {impl}")
                break
            except Exception as e:
                last_err = e
                print(f" attn_implementation='{impl}' failed: {e}")
                continue
        else:
            raise RuntimeError(
                f"All attn_implementation fallbacks failed for {model_id}; "
                f"last error: {last_err}"
            )

        self.current_model_id = model_id
        print(f"Model {model_id} loaded on {next(self.model.parameters()).device}")
   
    def _eos_ids(self):
        """Include EOS and EOT if present (Llama often uses <|eot_id|>)."""
        ids = set()
        for attr in ("eos_token_id", "eot_token_id"):
            val = getattr(self.tokenizer, attr, None)
            if val is not None:
                if isinstance(val, list):
                    ids.update(val)
                else:
                    ids.add(val)
        for attr in ("eos_token_id", "eot_token_id"):
            val = getattr(self.model.config, attr, None)
            if val is not None:
                if isinstance(val, list):
                    ids.update(val)
                else:
                    ids.add(val)
        if not ids:
            return None
        return list(ids) if len(ids) > 1 else next(iter(ids))

    def _cap_new_tokens(self, prompt_str: str, requested: int) -> int:
        """Avoid exceeding the model's context window."""
        cfg_ctx = getattr(self.model.config, "max_position_embeddings", None)
        tok_ctx = getattr(self.tokenizer, "model_max_length", None)
        ctx = None
        for v in (cfg_ctx, tok_ctx):
            if isinstance(v, int) and v > 0:
                ctx = v if ctx is None else min(ctx, v)
        if ctx is None:  # last-resort default
            ctx = 4096

        # Tokenize on CPU just to count; pipeline will retokenize on the right device.
        ids = self.tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt")["input_ids"]
        room = max(1, ctx - ids.shape[-1])
        return max(1, min(requested, room))

    def generate(self, question, task=None, temperature=0.0, max_new_tokens=2048, use_web_search=False, **kwargs):
        """
        Generate a deterministic or sampled response from the model.
        """
        sys_prompt = get_system_prompt(task)
        
        model_id_lower = self.current_model_id.lower()
        is_instruct = "instruct" in model_id_lower
        has_chat_template = getattr(self.tokenizer, "chat_template", None) is not None

        # Use both system and user roles if the model is instruct OR supports chat templates
        if is_instruct or has_chat_template:
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": question})
            try:
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if "qwen3.5" in model_id_lower:
                    template_kwargs["enable_thinking"] = False
                prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except AttributeError:
                prompt = f"{sys_prompt}\n\n{question}" if sys_prompt else question
                # print(f"apply_chat_template not available for model: {self.current_model_id}, fallback used")
        else:
            prompt = f"{sys_prompt}\n\n{question}" if sys_prompt else question
            # print(f"No chat template applied for model: {self.current_model_id}")

        # print(f"Prompt length: {len(prompt)} chars")

        eos = self._eos_ids()
        # Commented out: redundant tokenization pass just to count tokens — adds ~20% overhead
        # max_new = self._cap_new_tokens(prompt, max_new_tokens)
        # print(f"Max new tokens allowed: {max_new}")

        # Tokenize input
        ctx_limit = getattr(self.model.config, "max_position_embeddings", 4096)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_limit)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # print(f"Input token length: {inputs['input_ids'].shape[-1]}")
        # print(f"Using device: {self.model.device}")

        do_sample = True if temperature > 0 else False

        # with torch.no_grad():
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos,
                pad_token_id=self.tokenizer.pad_token_id,
                #repetition_penalty=1.0,
                do_sample=do_sample,
                temperature=temperature,
                #top_p=1,
                #top_k=50,
            )

        gen_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # print(f"Response length: {len(response)} chars")
        return response

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
            self.tokenizer = None
            self.current_model_id = None
            torch.cuda.empty_cache()
            print("Model cleaned up from memory")

# ------------------ Dummy Model --------------------- #
class DummyModel(BaseModel):
    def generate(self, question, **kwargs):
        return "This is a dummy response."

# ----------------- Global Model Cache ----------------- #
_model_cache = {}

def get_cached_model(model_name):
    """Get or create a cached model instance"""
    if model_name not in _model_cache:
        # Suffix-based routing takes precedence over family-substring matches
        # so 'llama-3-8b-base-vllm' hits VLLMModel rather than HuggingFaceModel.
        # '-vllm' => local vLLM OpenAI-compatible server, no local model load.
        # '-hf'   => hosted via HF Inference Providers, not local GPU.
        if model_name.endswith("-vllm"):
            _model_cache[model_name] = VLLMModel(model_name)
        elif model_name.endswith("-hf"):
            _model_cache[model_name] = HFInferenceModel(model_name)
        elif model_name.startswith("gpt-oss"):
            _model_cache[model_name] = HuggingFaceModel(model_name)
        elif model_name.startswith("gpt"):
            _model_cache[model_name] = OpenAIModel(model_name)
        elif model_name.startswith("gemini"):
            _model_cache[model_name] = GeminiModel(model_name)
        elif any(fam in model_name for fam in ("llama", "qwen", "foundation", "minerva", "deephat", "deepseek")):
            _model_cache[model_name] = HuggingFaceModel(model_name)
        else:
            raise ValueError(f"Unknown model type for: {model_name}")
        print(f"Model {model_name} cached and ready for reuse")
    
    return _model_cache[model_name]

def cleanup_model_cache(model_name=None):
    """Clean up cached models"""
    global _model_cache
    if model_name:
        if model_name in _model_cache:
            model = _model_cache[model_name]
            if isinstance(model, HuggingFaceModel):
                model.cleanup()
            del _model_cache[model_name]
            print(f"Model {model_name} removed from cache")
    else:
        # Clean up all models
        for name, model in _model_cache.items():
            if isinstance(model, HuggingFaceModel):
                model.cleanup()
        _model_cache.clear()
        print("All models removed from cache")

# ----------------- Global Function ----------------- #

# Per-task generation-length caps. Tight caps dramatically shorten local HF
# inference because the model stops generating as soon as it emits EOS, but
# wall-clock scales with `max_new_tokens` for any prompt where the model
# fails to emit EOS (rambling, repetition, formatting drift). Keeping MCQ
# at 128 trims 5-10x off runs on abaligned Llama-3.1-8B and has no observed
# accuracy impact for the SFT/test post-processors, which only care
# about the first answer letter / tail of the response.
TASK_MAX_NEW_TOKENS: dict[str, int] = {
    # MCQ-style: one letter + optional "Therefore, X." tail
    "athena-mcq":    128,
    "mcq":           128,
    "cybermetric":   128,
    # Short structured answers (technique ID, role label, short code)
    "athena-rcm":    256,
    "athena-rms":    256,
    "athena-taa":    256,
    "athena-taa-canonical": 256,
    "rcm":           256,
    "rms":           256,
    "taa":           256,
    # Medium free-form (technique extraction lists, severity paragraphs)
    "athena-ate":    512,
    "athena-vsp":    512,
    "ate":           512,
    "vsp":           512,
    # Long / unbounded tasks keep the historical 2048 default
    "cve":          1024,
    "urlhaus":       512,
    "glue":          256,
    "superglue":     256,
    "mmlu":          256,
    # MMLU-Pro is zero-shot CoT: model writes a multi-step chain
    # ("Let's think step by step...") before "The answer is (X)".
    # 256 truncates the chain mid-reasoning on most rows; 1024 leaves
    # comfortable headroom for hard math/physics rows without inflating
    # latency on easy subjects (model EOSes once it emits the answer).
    "mmlu-pro":     1024,
    # CyberSOCEval malware/TI emit a JSON object {"correct_answers":["A","B"]}
    # but Llama-3.1-8B-Instruct (and similar) preface the JSON with a verbose
    # bulleted MITRE/IOC walkthrough. The malware baseline at 256 tokens hit a
    # 30% parse-error rate driven entirely by mid-emit truncation (PE rate
    # jumps from 2% in the 0-500 char bucket to 86% in the 1000-1300 bucket,
    # which is the 256-token ceiling). 1024 leaves headroom for the preamble
    # without changing wall-clock on terse models that stop at EOS.
    "cybersoceval-malware": 1024,
    "cybersoceval-ti":      1024,
}
DEFAULT_MAX_NEW_TOKENS = 2048


def _resolve_max_new_tokens(task: str | None, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    if task and task in TASK_MAX_NEW_TOKENS:
        return TASK_MAX_NEW_TOKENS[task]
    return DEFAULT_MAX_NEW_TOKENS


def get_single_prediction(question, model_name, task=None, cleanup_after=False, use_web_search=False, temperature=0, max_new_tokens=None):
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(model_mapping.keys())}")

    # Resolve the per-task generation-length cap. Callers passing an
    # explicit max_new_tokens override the table; the default (None) falls
    # through to TASK_MAX_NEW_TOKENS[task] and finally to 2048.
    resolved_max = _resolve_max_new_tokens(task, max_new_tokens)

    # Get cached model instance
    model = get_cached_model(model_name)

    try:
        # All models now accept the same parameters (use_web_search is ignored by HuggingFace models)
        response = model.generate(
            question,
            task=task,
            temperature=temperature,
            max_new_tokens=resolved_max,
            use_web_search=use_web_search
        )
        
    except Exception as e:
        # vLLM ctx-overflow already emitted a one-liner from VLLMModel.generate;
        # the full stack here is pure noise (every overflow row would print 30+
        # lines of openai/_base_client internals). Other exceptions still get
        # a traceback for debugging.
        if getattr(e, "_vllm_ctx_overflow", False):
            response = f"Error generating response: ctx-overflow ({e.__class__.__name__})"
        else:
            print(f" Error generating response from {model_name}: {e}")
            traceback.print_exc()
            response = f"Error generating response: {e}"

    # Only cleanup if explicitly requested (this will remove from cache)
    if cleanup_after and isinstance(model, HuggingFaceModel):
        cleanup_model_cache(model_name)

    return response

# Export functions for external use
__all__ = ['get_single_prediction', 'get_cached_model', 'cleanup_model_cache',
           'model_mapping', 'alias_to_safe_name']
