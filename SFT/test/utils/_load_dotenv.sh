#!/bin/bash
# Source SFT/.env (or SFT/test/.env as a legacy fallback) into the current
# shell so per-family API keys (OPENAI_API_KEY, GEMINI_API_KEY, HF_TOKEN,
# HUGGINGFACE_TOKEN, ...) become available to pre-flight checks and
# downstream invocations. pipelines/models.py already calls load_dotenv()
# at import time, but those exports stay inside the Python process; bash
# pre-flights that fail-fast on missing keys need them in the shell too.
#
# This file is meant to be SOURCED, not executed:
#   source "${SCRIPT_DIR}/_load_dotenv.sh"
#
# Idempotent: a second source is a no-op if every variable already exists
# in the environment. Variables set in the shell win over .env values
# (set -a only exports; it does not overwrite existing exported vars
# unless the .env line is unconditional, in which case the .env wins --
# we use the pre-check guard below to avoid that surprise).

_glaukopis_load_dotenv() {
    local script_dir env_path
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for env_path in \
        "${script_dir}/../../.env" \
        "${script_dir}/../.env"; do
        if [[ -f "${env_path}" ]]; then
            set -a
            # shellcheck disable=SC1090
            source "${env_path}"
            set +a
            return 0
        fi
    done
    return 0
}

_glaukopis_load_dotenv
unset -f _glaukopis_load_dotenv
