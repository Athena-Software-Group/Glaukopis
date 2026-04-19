import os
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
    'gemini-2.5-flash': 'gemini-2.5-flash',
    'gemini-2.5-pro' : 'gemini-2.5-pro',
    'gemini-3-pro' : 'gemini-3-pro-preview',
    'llama-3-3b': 'meta-llama/Llama-3.2-3B',
    'llama-3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama-primus-8b' : 'trendmicro-ailab/Llama-Primus-Merged',
    #'llama-4-17b': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    'minerva' : "xashru/minerva_v0",
    'llama3.3-70b': 'meta-llama/Llama-3.3-70B-Instruct',
    'qwen2.5-14b': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen3-4b': 'Qwen/Qwen3-4B-Instruct-2507',
    'qwen3-8b': 'Qwen/Qwen3-8B',
    'qwen3-14b': 'Qwen/Qwen3-14B',
    'qwen3.5-9b':'Qwen/Qwen3.5-9B',
    'gpt-oss-20b': 'openai/gpt-oss-20b',
    'foundation-8b-reasoning': 'fdtn-ai/Foundation-Sec-8B-Reasoning',       # Cisco Foundation-Sec-8B-Reasoning
    'foundation-8b': 'fdtn-ai/Foundation-Sec-8B',                           # Cisco Foundation-Sec-8B simple model
    'minerva-llama8b':'athena-security/minerva-llama8b',
    'deephat-7b': 'DeepHat/DeepHat-V1-7B',
    'deepseek-r1-14b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',

    # --- HF Inference Providers (hosted; '-hf' suffix routes to HFInferenceModel) ---
    'deepseek-r1-14b-hf':  'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    'deepseek-r1-70b-hf':  'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'qwen3-14b-hf':        'Qwen/Qwen3-14B',
    'qwen2.5-14b-hf':      'Qwen/Qwen2.5-14B-Instruct',
    'llama-3-70b-hf':      'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama3.3-70b-hf':     'meta-llama/Llama-3.3-70B-Instruct',
    'deepseek-v3.2-exp-hf': 'deepseek-ai/DeepSeek-V3.2-Exp',
}

# --- Centralized Helpers ---
def check_disk_space(model_id):
    stat = shutil.disk_usage(workspace_cache)
    available_gb = stat.free / (1024 ** 3)
    required_gb = 5
    if '70b' in model_id.lower(): required_gb = 150
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
    elif task in ["ate", "rcm", "vsp", "taa", "mcq"]: # CTIBench tasks
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

        elif self.model_name == 'gpt5.2':
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

        last_err = None
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.hf_model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                choice = resp.choices[0] if resp.choices else None
                content = (choice.message.content if choice and choice.message else "") or ""
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
        # '-hf' suffix => hosted via HF Inference Providers, not local GPU
        if model_name.endswith("-hf"):
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

def get_single_prediction(question, model_name, task=None, cleanup_after=False, use_web_search=False, temperature=0, max_new_tokens=2048):
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(model_mapping.keys())}")

    # Set model-specific max_new_tokens if not explicitly provided
    # Llama 3.3 70b max_new_tokens = 1000    
    #if max_new_tokens == 512 and model_name == 'gpt-oss-20b' and task == 'athena-vsp':
    #    max_new_tokens = 1000

    # Get cached model instance
    model = get_cached_model(model_name)

    try:
        # All models now accept the same parameters (use_web_search is ignored by HuggingFace models)
        response = model.generate(
            question,
            task=task,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_web_search=use_web_search
        )
        
    except Exception as e:
        # Print detailed error info for debugging
        print(f" Error generating response from {model_name}: {e}")
        traceback.print_exc()
        response = f"Error generating response: {e}"

    # Only cleanup if explicitly requested (this will remove from cache)
    if cleanup_after and isinstance(model, HuggingFaceModel):
        cleanup_model_cache(model_name)

    return response

# Export functions for external use
__all__ = ['get_single_prediction', 'get_cached_model', 'cleanup_model_cache', 'model_mapping']
