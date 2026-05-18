#!/usr/bin/env bash
# bootstrap_remote.sh -- one-shot remote-host bootstrap for Glaukopis.
#
# Configures global git identity, wires up GitHub HTTPS credentials for
# the private Glaukopis repo, and clones the repo to a target directory.
# Run this ONCE per fresh remote box BEFORE any other Glaukopis tooling.
# After it completes, follow up with:
#   1. rsync SFT/.env from the source box -> <target>/SFT/.env
#   2. rsync SFT/data/ift_data_*.json shards (gitignored datasets)
#   3. cd <target>/SFT/utils && ./setup.sh --cuda cu128   (or cu130)
#
# Usage (env vars; values are NOT echoed):
#   GITHUB_TOKEN=ghp_xxx \
#   GIT_USER_NAME="Your Name" \
#   GIT_USER_EMAIL=you@example.com \
#       ./bootstrap_remote.sh
#
# Usage (interactive; token read silently):
#   ./bootstrap_remote.sh
#
# Usage (flags):
#   ./bootstrap_remote.sh --git-user-name "..." --git-user-email "..." \
#                         --git-token ghp_xxx \
#                         [--target ~/Glaukopis] [--branch main] \
#                         [--repo-url https://github.com/ORG/REPO.git]
#
# Re-running is safe: existing git identity is overwritten, the github.com
# line in ~/.git-credentials is rotated (not stacked), and an existing
# checkout is fetched + fast-forwarded instead of re-cloned.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Athena-Software-Group/Glaukopis.git}"
TARGET_DIR="${TARGET_DIR:-${HOME}/Glaukopis}"
BRANCH="${BRANCH:-main}"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --git-user-name)  GIT_USER_NAME="$2"; shift 2 ;;
        --git-user-email) GIT_USER_EMAIL="$2"; shift 2 ;;
        --git-token)      GITHUB_TOKEN="$2"; shift 2 ;;
        --target)         TARGET_DIR="$2"; shift 2 ;;
        --branch)         BRANCH="$2"; shift 2 ;;
        --repo-url)       REPO_URL="$2"; shift 2 ;;
        -h|--help)        sed -n '3,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

command -v git >/dev/null 2>&1 \
    || { echo "git not installed on this host; install it before re-running." >&2; exit 1; }

# Interactive fill-in for anything still missing. Token read with -s so it
# never lands on the terminal or in shell history.
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

# Scrub the token from this shell.
unset GITHUB_TOKEN

echo
echo "=== Done ==="
echo "Repo at : ${TARGET_DIR}"
echo "HEAD    : $(git -C "${TARGET_DIR}" log --oneline -1)"
echo "Branch  : $(git -C "${TARGET_DIR}" rev-parse --abbrev-ref HEAD)"
echo
echo "Next steps:"
echo "  1. From the source box, rsync SFT/.env over (creds for HF/wandb):"
echo "       rsync -avP -e 'ssh -p <PORT>' SFT/.env <THIS_HOST>:${TARGET_DIR}/SFT/.env"
echo "  2. From the source box, rsync the v21 dataset shards in parallel"
echo "     (see the xargs -P4 rsync recipe in the v21 launch notes)."
echo "  3. On THIS host, install the conda env + CUDA-matched torch:"
echo "       cd ${TARGET_DIR}/SFT/utils && ./setup.sh --cuda cu128"
echo "       # (use cu130 if nvcc reports CUDA 13.x; cu128 is the floor for"
echo "       #  Blackwell sm_120 / RTX PRO 6000)"
