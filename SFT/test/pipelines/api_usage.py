import os
import json
from datetime import datetime

# ---- Global counters ----
_total_input_tokens = 0
_total_output_tokens = 0
_total_grounding_calls = 0
_total_cost = 0.0  # running total in USD
_total_input_cost = 0.0
_total_output_cost = 0.0

# ---- Pricing per 1K tokens (USD), standard and batch pricing ----
PRICING_PER_1K = {
    # Gemini
    "gemini-2.5-flash": {
            "input": [(0, float("inf"), 0.0003)],    
            "output": [(0, float("inf"), 0.0025)],   
            "grounding_per_1k": 35.0,      
    },
    "gemini-2.5-pro": {
            "input": [(0, 200_000, 0.00125), (200_001, float("inf"), 0.00250)],
            "output": [(0, 200_000, 0.01000), (200_001, float("inf"), 0.01500)],
            "grounding_per_1k": 35.0,   # $35 per 1K requests (same as batch)
    },
    "gemini-3-pro": {
            # Deprecated 2026-03-09; entry retained for back-compat with prior
            # response files. Same tier shape as gemini-3.1-pro below.
            "input": [(0, 200_000, 0.00200), (200_001, float("inf"), 0.00400)],
            "output": [(0, 200_000, 0.01200), (200_001, float("inf"), 0.01800)],
            "grounding_per_1k": 35.0,
    },
    "gemini-3.1-pro": {
            # Standard tier. <=200K ctx: $2/$12 per 1M; >200K: $4/$18 per 1M.
            "input": [(0, 200_000, 0.00200), (200_001, float("inf"), 0.00400)],
            "output": [(0, 200_000, 0.01200), (200_001, float("inf"), 0.01800)],
            "grounding_per_1k": 35.0,
    },
    "gemini-3-flash": {
            # Vertex AI listed pricing as of 2026-04: $0.50 input / $3.00 output
            # per 1M tokens, flat (no >200K context tier breakpoint published
            # for Flash). Used as the non-Pro Gemini frontier baseline; there
            # is no general-purpose 'gemini-3.1-flash' (3.1 only ships
            # Flash-Lite, Flash Image, Flash Live, and Flash TTS).
            "input": [(0, float("inf"), 0.00050)],
            "output": [(0, float("inf"), 0.00300)],
            "grounding_per_1k": 35.0,
    },
    # GeminiModel.generate passes the resolved model id (with -preview suffix)
    # to add_tokens, not the alias key, so duplicate the rate cards under the
    # resolved id as well. Keeping both shapes avoids breaking existing
    # checkpoint data keyed by the alias form.
    "gemini-3-pro-preview": {
            "input": [(0, 200_000, 0.00200), (200_001, float("inf"), 0.00400)],
            "output": [(0, 200_000, 0.01200), (200_001, float("inf"), 0.01800)],
            "grounding_per_1k": 35.0,
    },
    "gemini-3.1-pro-preview": {
            "input": [(0, 200_000, 0.00200), (200_001, float("inf"), 0.00400)],
            "output": [(0, 200_000, 0.01200), (200_001, float("inf"), 0.01800)],
            "grounding_per_1k": 35.0,
    },
    "gemini-3-flash-preview": {
            "input": [(0, float("inf"), 0.00050)],
            "output": [(0, float("inf"), 0.00300)],
            "grounding_per_1k": 35.0,
    },
    # OpenAI GPT models
    "gpt4": {
            "input": [(0, float("inf"), 0.01)],       # $10 per 1M tokens
            "output": [(0, float("inf"), 0.03)],      # $30 per 1M tokens tokens
    },
    "gpt5": {
            "input": [(0, float("inf"), 0.00125)],    # $1.25/ 1M tokens
            "output": [(0, float("inf"), 0.01)],      # $10.00 / 1M tokens
            "grounding_per_1k": 10.0,              # $10 per 1k web_search_preview call
    },
    "gpt5.2": {
        "input": [(0, float("inf"), 0.00175)],    # $1.75 per 1M tokens
        "output": [(0, float("inf"), 0.014)],     # $14 per 1M tokens
    },
    "gpt5.5": {
        # Released 2026-04-23. $5 input / $30 output per 1M tokens (<272K ctx).
        "input": [(0, float("inf"), 0.00500)],
        "output": [(0, float("inf"), 0.03000)],
        "grounding_per_1k": 10.0,             # web_search_preview, same as gpt5
    },
    "gpt5.5-pro": {
        # Released 2026-04-23. $30 input / $180 output per 1M tokens.
        "input": [(0, float("inf"), 0.03000)],
        "output": [(0, float("inf"), 0.18000)],
        "grounding_per_1k": 10.0,
    },
}

RESPONSES_DIR = os.path.join(os.getcwd(), "responses")
os.makedirs(RESPONSES_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(RESPONSES_DIR, "api_usage_checkpoint.json")

# ---- Helpers ----
def _get_tiered_rate(rates, tokens):
    """Pick correct per-1K rate for this request's token count."""
    for low, high, price in rates:
        if low <= tokens <= high:
            return price
    raise ValueError("No matching pricing tier found")

def add_tokens(model_name: str, input_tokens: int, output_tokens: int, grounding: bool = False,
               grounding_calls: int = None, cost_override: float = None):
    """
    Add usage for a request and compute cost.
    Supports Gemini, GPT, and GROQ models.
    Only Gemini models have grounding cost.
    """
    global _total_input_tokens, _total_output_tokens, _total_cost, _total_grounding_calls, _total_input_cost, _total_output_cost

    if model_name not in PRICING_PER_1K:
        raise ValueError(f"Unknown model pricing: {model_name}")

    rates = PRICING_PER_1K[model_name]
    input_rate = _get_tiered_rate(rates["input"], input_tokens)
    output_rate = _get_tiered_rate(rates["output"], output_tokens)

    input_cost = (input_tokens / 1000) * input_rate
    output_cost = (output_tokens / 1000) * output_rate
    request_cost = input_cost + output_cost

    # apply grounding cost if applicable
    if grounding and "grounding_per_1k" in rates:
        grounding_cost = (1 / 1000) * rates["grounding_per_1k"]
        request_cost += grounding_cost
        _total_grounding_calls += 1
    elif grounding_calls is not None:
        _total_grounding_calls += grounding_calls

    # update totals
    _total_input_tokens += input_tokens
    _total_output_tokens += output_tokens
    _total_input_cost += input_cost
    _total_output_cost += output_cost
    if cost_override is not None:
        _total_cost = cost_override
    else:
        _total_cost += request_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "grounding_cost": (1 / 1000) * rates.get("grounding_per_1k", 0) if grounding else 0,
        "request_cost": request_cost,
    }

def get_totals():
    """Return cumulative usage totals."""
    return {
        "input_tokens": _total_input_tokens,
        "output_tokens": _total_output_tokens,
        "total_tokens": _total_input_tokens + _total_output_tokens,
        "grounding_calls": _total_grounding_calls,
        "input_cost": _total_input_cost,
        "output_cost": _total_output_cost,
        "total_cost": _total_cost,
    }


# ---- Checkpoint functions ----
def save_checkpoint(task, model_name, version=1):
    """Save current totals to checkpoint file"""
    totals = get_totals()
    checkpoint = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "model_name": model_name,
        "version": version,
        "input_tokens": totals['input_tokens'],
        "output_tokens": totals['output_tokens'],
        "input_tokens_cost": totals['input_cost'],
        "output_tokens_cost": totals['output_cost'],
        "grounding_calls": totals['grounding_calls'],
        "total_cost": totals['total_cost']
    }

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Remove old entry for same task+model+version
    data = [d for d in data if not (d["task"] == task and d["model_name"] == model_name and d.get("version", 1) == version)]
    data.append(checkpoint)

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Checkpoint saved for {task} / {model_name}")

def restore_checkpoint(task, model_name, version=1):
    """Restore previous totals from checkpoint (idempotent)."""
    global _total_input_tokens, _total_output_tokens, _total_grounding_calls, _total_cost
    global _total_input_cost, _total_output_cost 

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)

        for entry in data:
            if entry["task"] == task and entry["model_name"] == model_name and entry.get("version", 1) == version:
                print(f"Restoring totals for {task} / {model_name} / v{version}")
                # Directly restore totals instead of re-adding
                _total_input_tokens = entry["input_tokens"]
                _total_output_tokens = entry["output_tokens"]
                _total_input_cost = entry.get("input_tokens_cost", 0.0)
                _total_output_cost = entry.get("output_tokens_cost", 0.0)
                _total_grounding_calls = entry["grounding_calls"]
                _total_cost = entry["total_cost"]

                return True

    return False
