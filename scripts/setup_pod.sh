#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu+CUDA pod for the SELF-HOSTED prime-rl path.
#
# You only need this if you are not using Prime Hosted Training (configs/hosted-rl.toml),
# which needs no pod at all. Idempotent: safe to re-run, and re-run it if a bare
# `uv run` ever prunes the environment package back out of prime-rl's venv.
#
#   git clone <this repo> ~/pan-gan && bash ~/pan-gan/scripts/setup_pod.sh
#
# Override the prime-rl location with:  PRIME_RL_DIR=/path bash scripts/setup_pod.sh
#
# prime-rl needs 2 NVIDIA GPUs minimum for the RL loop (1 trainer + 1 inference),
# and Python ~=3.12 (uv provisions it; this repo's own venv is 3.14 and is unrelated
# - the environment package is installed into prime-rl's venv, not ours).

set -euo pipefail

PAN_GAN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
ENV_PKG="$PAN_GAN_DIR/environments/pangram_creative_writing"

if [ ! -d "$ENV_PKG" ]; then
  echo "error: environment package not found at $ENV_PKG" >&2
  exit 1
fi

# --- 1. prime-rl, per its README (uv + submodules + locked sync) --------------

if [ ! -d "$PRIME_RL_DIR/.git" ]; then
  git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_RL_DIR"
fi
cd "$PRIME_RL_DIR"

# prime-rl's .gitmodules uses SSH URLs (git@github.com:PrimeIntellect-ai/...), which
# fail on a fresh pod with no SSH key loaded. Rewrite them to HTTPS for this repo
# only. These are all public.
git config url."https://github.com/".insteadOf "git@github.com:"
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck source=/dev/null
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"

uv sync --all-extras

# --- 2. our environment package, editable, into prime-rl's venv ---------------
#
# Ordering matters. This must run AFTER `uv sync`, because sync prunes anything
# not in prime-rl's lockfile - which includes our package. That is also why every
# launch below uses `uv run --no-sync`: a bare `uv run` re-syncs and uninstalls it
# again. prime-rl documents the same escape hatch for flash-attn.

uv pip install -e "$ENV_PKG"
uv run --no-sync python -c "import pangram_creative_writing; print('env package import OK')"

# --- 3. secrets checklist -----------------------------------------------------
#
# The orchestrator process (and the env server it spawns as a child) inherits the
# launching shell's environment, so exporting these before `uv run rl` is enough.
# Never put a key in a config file.

missing=0
require() {
  if [ -z "${!1:-}" ]; then
    echo "  MISSING  $1  - $2"
    missing=1
  else
    echo "  ok       $1"
  fi
}

echo
echo "Environment variables:"
require PANGRAM_API_KEY "required: the env server calls the detector once per rollout"
require WANDB_API_KEY   "required for wandb logging (project 'pan-gan')"
require PRIME_API_KEY   "eval only: the craft-rubric judge on https://api.pinference.ai/api/v1"
require HF_TOKEN        "optional: only for gated HF models; Qwen3.5 is public"

if [ "$missing" -ne 0 ]; then
  echo
  echo "Export the missing variables, then re-run this script to re-check."
fi

# --- 4. what to run -----------------------------------------------------------

cat <<EOF

Setup complete.  prime-rl: $PRIME_RL_DIR   configs: $PAN_GAN_DIR/configs

  # 0. THE GATE. Do this before anything that costs real money.
  #    ~\$3.20 of Pangram; tells you whether the reward has any signal at all.
  bash $PAN_GAN_DIR/scripts/calibrate.sh

  cd $PRIME_RL_DIR

  # 1. validate the configs without touching a GPU. Free, and the first thing to
  #    run after editing any config in $PAN_GAN_DIR/configs.
  uv run --no-sync rl @ $PAN_GAN_DIR/configs/rl.toml --dry-run --output-dir /tmp/dry
  uv run --no-sync rl @ $PAN_GAN_DIR/configs/rl.toml \\
                     @ $PAN_GAN_DIR/configs/eval.toml --dry-run --output-dir /tmp/dry-eval

  # 2. smoke test: 2 steps, 32 rollouts, ~\$1.60 of Pangram
  uv run --no-sync rl @ $PAN_GAN_DIR/configs/rl-debug.toml \\
    --output-dir outputs/debug --clean-output-dir

  # 3. the real run (~\$160 of Pangram; the eval overlay adds ~\$9.60 + the judge)
  uv run --no-sync rl @ $PAN_GAN_DIR/configs/rl.toml \\
                     @ $PAN_GAN_DIR/configs/eval.toml \\
    --output-dir outputs/pangram-v1

  # watch it
  tail -F outputs/pangram-v1/logs/orchestrator.log
EOF
