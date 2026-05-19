#!/usr/bin/env bash
# bootstrap_remote.sh -- one-shot remote-host bootstrap for Glaukopis.
#
# Configures global git identity, wires up GitHub HTTPS credentials for
# the private Glaukopis repo, and clones the repo to a target directory.
# Run this ONCE per fresh remote box BEFORE any other Glaukopis tooling.
# After it completes, follow up with:
#   1. rsync SFT/data/ift_data_*.json shards (gitignored datasets)
#   2. cd <target>/SFT/utils && ./setup.sh --cuda cu128   (or cu130)
# (A co-located .env beside this script is auto-moved into <target>/SFT/.env
#  after a successful clone, so no manual rsync of credentials is needed.)
#
# Usage (env vars; values are NOT echoed):
#   GITHUB_TOKEN=ghp_xxx \
#   GIT_USER_NAME="Your Name" \
#   GIT_USER_EMAIL=you@example.com \
#       ./bootstrap_remote.sh
#
# Usage (.env file; recommended on Verda/RunPod/etc. that pre-stage creds):
#   # 1. scp this script + your .env into the SAME directory on the host:
#   #      scp SFT/utils/bootstrap_remote.sh root@HOST:/root/
#   #      scp /path/to/.env                  root@HOST:/root/.env
#   # 2. On the remote host:
#   ./bootstrap_remote.sh
#   # The .env in the script's directory is auto-detected; pass
#   # --env-file PATH to override or ${HOME}/.env as a fallback.
#
# Usage (interactive; token read silently):
#   ./bootstrap_remote.sh
#
# Usage (flags):
#   ./bootstrap_remote.sh --git-user-name "..." --git-user-email "..." \
#                         --git-token ghp_xxx \
#                         [--env-file PATH] [--target ~/Glaukopis] \
#                         [--branch main] [--repo-url https://github.com/ORG/REPO.git]
#
# Precedence (highest -> lowest) for GITHUB_TOKEN / GIT_USER_NAME /
# GIT_USER_EMAIL: CLI flag > pre-existing shell export > --env-file value
# > interactive prompt. So passing --git-token always overrides whatever
# the .env file says, which matches the principle of least surprise.
#
# Re-running is safe: existing git identity is overwritten, the github.com
# line in ~/.git-credentials is rotated (not stacked), and an existing
# checkout is fetched + fast-forwarded instead of re-cloned.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_URL="${REPO_URL:-https://github.com/Athena-Software-Group/Glaukopis.git}"
TARGET_DIR="${TARGET_DIR:-${HOME}/Glaukopis}"
BRANCH="${BRANCH:-main}"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
ENV_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --git-user-name)  GIT_USER_NAME="$2"; shift 2 ;;
        --git-user-email) GIT_USER_EMAIL="$2"; shift 2 ;;
        --git-token)      GITHUB_TOKEN="$2"; shift 2 ;;
        --env-file)       ENV_FILE="$2"; shift 2 ;;
        --target)         TARGET_DIR="$2"; shift 2 ;;
        --branch)         BRANCH="$2"; shift 2 ;;
        --repo-url)       REPO_URL="$2"; shift 2 ;;
        -h|--help)        sed -n '3,44p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

command -v git >/dev/null 2>&1 \
    || { echo "git not installed on this host; install it before re-running." >&2; exit 1; }

# .env auto-detect: if no --env-file was passed, search (in order):
#   1. ${SCRIPT_DIR}/.env   -- co-located with this script (recommended:
#      scp bootstrap_remote.sh + .env into the same directory on the
#      fresh host, then just run ./bootstrap_remote.sh).
#   2. ${HOME}/.env         -- legacy fallback for boxes that pre-stage
#      creds in the home dir (Verda / RunPod / Modal / etc.).
# Skipped if the operator explicitly passed all three creds via flags /
# shell exports, since there'd be nothing to read.
if [[ -z "${ENV_FILE}" ]]; then
    if [[ -z "${GIT_USER_NAME}" || -z "${GIT_USER_EMAIL}" || -z "${GITHUB_TOKEN}" ]]; then
        for _candidate in "${SCRIPT_DIR}/.env" "${HOME}/.env"; do
            if [[ -f "${_candidate}" ]]; then
                ENV_FILE="${_candidate}"
                echo "=== Auto-detected ${ENV_FILE} (use --env-file '' to opt out) ==="
                break
            fi
        done
    fi
fi

# Source --env-file so values persisted there fill in anything still empty
# after CLI parsing + shell exports. Values that came from CLI flags or
# pre-existing shell exports take precedence (snapshot-and-restore pattern).
# `set -a` exports every assignment in the sourced file so subprocesses
# (e.g. the upcoming `git clone`) inherit them too.
if [[ -n "${ENV_FILE}" ]]; then
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "--env-file not found: ${ENV_FILE}" >&2
        exit 2
    fi
    _name_pre="${GIT_USER_NAME}"
    _email_pre="${GIT_USER_EMAIL}"
    _token_pre="${GITHUB_TOKEN}"
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    set +a
    [[ -n "${_name_pre}"  ]] && GIT_USER_NAME="${_name_pre}"
    [[ -n "${_email_pre}" ]] && GIT_USER_EMAIL="${_email_pre}"
    [[ -n "${_token_pre}" ]] && GITHUB_TOKEN="${_token_pre}"
fi

# Interactive fill-in for anything still missing. Token read with -s so it
# never lands on the terminal or in shell history. Skipped automatically
# when all three values were satisfied by CLI flags / shell exports / .env.
if [[ -z "${GIT_USER_NAME}"  ]]; then read -rp  "Git user.name : " GIT_USER_NAME; fi
if [[ -z "${GIT_USER_EMAIL}" ]]; then read -rp  "Git user.email: " GIT_USER_EMAIL; fi
if [[ -z "${GITHUB_TOKEN}"   ]]; then read -rsp "GitHub PAT (ghp_...): " GITHUB_TOKEN; echo; fi

[[ -n "${GIT_USER_NAME}"  ]] || { echo "GIT_USER_NAME required"  >&2; exit 2; }
[[ -n "${GIT_USER_EMAIL}" ]] || { echo "GIT_USER_EMAIL required" >&2; exit 2; }
[[ -n "${GITHUB_TOKEN}"   ]] || { echo "GITHUB_TOKEN required"   >&2; exit 2; }

echo "=== Global git identity ==="
git config --global user.name  "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"
echo "  user.name : $(git config --global --get user.name)"
echo "  user.email: $(git config --global --get user.email)"

# Persist the PAT in ~/.git-credentials with credential.helper=store so
# subsequent git pulls/pushes don't prompt. Mirrors SFT/utils/setup.sh's
# behaviour (lines 278-293): strip any pre-existing github.com line first
# so re-runs rotate the token cleanly instead of stacking duplicates.
echo "=== GitHub HTTPS credential helper ==="
git config --global credential.helper store
CRED_FILE="${HOME}/.git-credentials"
if [[ -f "${CRED_FILE}" ]]; then
    grep -v '@github.com' "${CRED_FILE}" > "${CRED_FILE}.tmp" 2>/dev/null || true
    mv "${CRED_FILE}.tmp" "${CRED_FILE}"
fi
( umask 077 && \
  printf 'https://x-access-token:%s@github.com\n' "${GITHUB_TOKEN}" >> "${CRED_FILE}" )
chmod 600 "${CRED_FILE}" 2>/dev/null || true
echo "  ok (file: ${CRED_FILE}, perms 0600)"

echo "=== Cloning ${REPO_URL} -> ${TARGET_DIR} (branch ${BRANCH}) ==="
if [[ -d "${TARGET_DIR}/.git" ]]; then
    echo "  ${TARGET_DIR} already a git checkout; fetching + fast-forwarding."
    git -C "${TARGET_DIR}" fetch origin
    git -C "${TARGET_DIR}" checkout "${BRANCH}"
    git -C "${TARGET_DIR}" pull --ff-only origin "${BRANCH}"
else
    mkdir -p "$(dirname "${TARGET_DIR}")"
    git clone --branch "${BRANCH}" "${REPO_URL}" "${TARGET_DIR}"
fi

# Promote the co-located .env (if any) into the cloned tree so downstream
# tooling (SFT/utils/setup.sh, run_train.sh, run_benchmark.sh, ...) can
# pick up HF / wandb / API creds from the canonical SFT/.env path. Only
# the sibling .env is moved; --env-file targets and ${HOME}/.env are left
# in place so they remain reusable for re-runs / sibling hosts.
SRC_ENV="${SCRIPT_DIR}/.env"
DEST_ENV="${TARGET_DIR}/SFT/.env"
if [[ -f "${SRC_ENV}" ]]; then
    echo "=== Promoting co-located .env -> ${DEST_ENV} ==="
    mkdir -p "$(dirname "${DEST_ENV}")"
    if [[ -e "${DEST_ENV}" ]]; then
        _backup="${DEST_ENV}.bak.$(date +%Y%m%d-%H%M%S)"
        echo "  ${DEST_ENV} already exists; backing up to ${_backup}"
        mv "${DEST_ENV}" "${_backup}"
    fi
    mv "${SRC_ENV}" "${DEST_ENV}"
    chmod 600 "${DEST_ENV}" 2>/dev/null || true
    echo "  ok (perms 0600)"
else
    echo "=== No co-located .env at ${SRC_ENV}; skipping promotion ==="
fi

# Scrub the token from this shell.
unset GITHUB_TOKEN

echo
echo "=== Done ==="
echo "Repo at : ${TARGET_DIR}"
echo "HEAD    : $(git -C "${TARGET_DIR}" log --oneline -1)"
echo "Branch  : $(git -C "${TARGET_DIR}" rev-parse --abbrev-ref HEAD)"
echo
echo "Next steps:"
echo "  1. From the source box, rsync the v21 dataset shards in parallel"
echo "     (see the xargs -P4 rsync recipe in the v21 launch notes)."
echo "  2. On THIS host, install the conda env + CUDA-matched torch:"
echo "       cd ${TARGET_DIR}/SFT/utils && ./setup.sh --cuda cu128"
echo "       # (use cu130 if nvcc reports CUDA 13.x; cu128 is the floor for"
echo "       #  Blackwell sm_120 / RTX PRO 6000)"
