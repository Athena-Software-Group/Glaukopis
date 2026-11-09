#!/bin/bash
# Put both scoring environments on the package versions available just before the May 20-21 2026 scoring runs.
# vllm 0.21.0 alone is not enough: unpinned xgrammar now caps transformers below 5, and transformers 4.x cannot
# read the tokenizers these checkpoints were saved with. The torch family stays as installed.
C=2026-05-20T00:00:00Z
source /root/miniconda3/bin/activate || exit 1

pin() {
    local env=$1
    shift
    conda activate "$env" || return 1
    pip install --dry-run --ignore-installed --uploaded-prior-to=$C "$@" 2>/dev/null \
        | grep '^Would install' | tr ' ' '\n' | tail -n +3 \
        | grep -v -E '^(torch|torchaudio|torchvision|torch_c_dlpack_ext|triton|cuda|cuda-|cuda_|nvidia-|nvidia_)' \
        | sed -E 's/-([^-]+)$/==\1/' > ~/logs/may_pins_$env.txt
    echo "$env: $(wc -l < ~/logs/may_pins_$env.txt) pinned packages"
    [ -s ~/logs/may_pins_$env.txt ] || return 1
    pip install --no-deps --uploaded-prior-to=$C -r ~/logs/may_pins_$env.txt || return 1
    conda deactivate
}

pin vllm "vllm==0.21.0" openai python-dotenv huggingface_hub "pypdf>=5.0.0" || exit 2
pin ctibench -r ~/Glaukopis/SFT/eval/requirements.txt || exit 3

cd ~/Glaukopis/SFT || exit 4
set -a; source ./.env; set +a

conda activate vllm || exit 5
python - <<'EOF' || exit 6
import vllm, torch, transformers, tokenizers, xgrammar
from transformers import AutoTokenizer
print("vllm env:", "vllm", vllm.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__, "| tokenizers", tokenizers.__version__, "| xgrammar", xgrammar.__version__)
for r in ("asg-ai/athena-cti-sft-qwen25-32b-v21-recal-32b", "asg-ai/athena-cti-sft-qwen25-32b-v21-single-stage"):
    t = AutoTokenizer.from_pretrained(r)
    print("tokenizer ok:", r.rsplit("-v21-", 1)[1], type(t).__name__, len(t))
EOF
conda deactivate

conda activate ctibench || exit 7
python - <<'EOF' || exit 8
import numpy, openai, pandas, sklearn, transformers
from transformers import pipeline
print("ctibench env:", "pandas", pandas.__version__, "| numpy", numpy.__version__, "| openai", openai.__version__, "| sklearn", sklearn.__version__, "| transformers", transformers.__version__, "| pipeline import ok")
EOF
