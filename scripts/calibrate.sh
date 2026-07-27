#!/usr/bin/env bash
# THE CALIBRATION GATE. Run this before spending money on any training run.
#
# Pangram's score is a step function, not a gradient: every AI text we measured
# scored 0.9930-0.9936 and every human text scored 0.017-0.163, with nothing in
# between. So `reward = 1 - ai_score` is not a hill to climb, it is a cliff to
# clear, and RL can only work here if the BASE MODEL already clears it sometimes.
#
# This script samples N*R rollouts from the base model on real task prompts,
# scores every one through the real environment, and prints the ai_score
# distribution.
#
# Cost: one Pangram call per rollout at $0.05. The defaults below are 64 rollouts
# = $3.20 of Pangram, plus a few cents of Prime inference. Pangram is billed on
# the Pangram account; the $50 Prime wallet does not cover it.
#
# DECISION RULE
#   >= ~5% of rollouts below 0.9  -> sparse-binary reward is viable. Proceed to
#                                    configs/hosted-rl-smoke.toml.
#   ~0% below 0.9                 -> STOP. Every group will be uniform, every
#                                    advantage zero. Change the TASK first (see
#                                    README.md -> "If the gate fails").
#
# Usage:
#   bash scripts/calibrate.sh                      # defaults below
#   MODEL=Qwen/Qwen3.5-2B N=16 R=8 bash scripts/calibrate.sh
#   DRY_RUN=1 bash scripts/calibrate.sh            # resolve the config, spend nothing

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
N="${N:-8}"    # prompts
R="${R:-8}"    # rollouts per prompt  ->  N*R total Pangram calls
OUT="${OUT:-$REPO/outputs/calibration}"

# The v1 eval CLI. `prime eval run` is NOT this - in prime CLI 0.6.20 it still
# dispatches to the legacy v0 `python -m verifiers.cli.commands.eval`, which has
# no --dry-run, no --harness.*, no --taskset.*, and silently defaults the model to
# openai/gpt-4.1-mini. Same stale-scaffold trap as `prime env init`. The v1
# entrypoints are the `eval` and `validate` console scripts that verifiers
# installs into this repo's venv.
EVAL="${EVAL:-$REPO/.venv/bin/eval}"
if [ ! -x "$EVAL" ]; then
  echo "error: $EVAL not found. Create the venv and install the environment package:" >&2
  echo "  uv sync && uv pip install -e $REPO/environments/pangram_creative_writing" >&2
  exit 1
fi

TOTAL=$((N * R))
COST=$(python3 -c "print(f'{$TOTAL * 0.05:.2f}')")

# --harness.id null is the tool-less chat loop. It is NOT optional: HarnessConfig's
# default id is "bash", and a dry-run silently resolves to it. There is also no
# "default" harness in verifiers - import_harness("default") raises
# ModuleNotFoundError - despite what `prime train init` and the skill docs say.
#
# sampling.max-tokens 3072: at ~1600 the model truncated ~97% of rollouts
# mid-sentence (it overshoots the 400-700 word target to a median of ~1,250
# words). Truncated stories still get scored, which would corrupt the gate.
#
# It must be --sampling.max-tokens, NOT --max-output-tokens. The latter is the
# framework's per-rollout budget, enforced only *between turns*; a single-turn
# null harness never reaches a second turn, so it never fires. Setting it and
# assuming generation was capped produced a 52,802-word rollout in the clean
# calibration run. Only max_scored_words kept that from billing 53 units.
EVAL_ARGS=(
  --taskset.id pangram-creative-writing
  --taskset.split train
  --harness.id null
  --harness.runtime.type subprocess
  -m "$MODEL"
  -n "$N" -r "$R"
  --shuffle
  --no-push
  -c 8
  --sampling.max-tokens 3072
  -o "$OUT"
)

if [ -n "${DRY_RUN:-}" ]; then
  echo "dry run: resolving the config only, no rollouts and no Pangram spend"
  "$EVAL" "${EVAL_ARGS[@]}" --dry-run
  exit 0
fi

echo "calibration gate"
echo "  model      $MODEL"
echo "  rollouts   $N prompts x $R = $TOTAL"
echo "  cost       ~\$$COST of Pangram (billed on the Pangram account, not the Prime wallet)"
echo
read -r -p "proceed? [y/N] " reply
[[ "$reply" == [yY] ]] || { echo "aborted"; exit 1; }

for var in PANGRAM_API_KEY PRIME_API_KEY; do
  if [ -z "${!var:-}" ]; then
    echo "error: $var is not exported" >&2
    exit 1
  fi
done

"$EVAL" "${EVAL_ARGS[@]}"

TRACES="$(find "$OUT" -name traces.jsonl -print0 | xargs -0 ls -t | head -1)"
echo
echo "traces: $TRACES"

python3 - "$TRACES" <<'PY'
import json, sys

def walk(obj):
    """ai_score is recorded as a metric; find it wherever the trace schema puts it."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "ai_score" and isinstance(v, (int, float)):
                yield float(v)
            else:
                yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

scores = [s for line in open(sys.argv[1]) if line.strip() for s in walk(json.loads(line))]
if not scores:
    sys.exit("no ai_score found in traces - check the metric name against CONTRACT.md")

scores.sort()
n = len(scores)
pct = lambda q: scores[min(n - 1, int(q * n))]

print(f"\nn = {n}")
print(f"min    {scores[0]:.6f}")
for q in (0.1, 0.25, 0.5, 0.75, 0.9):
    print(f"p{int(q*100):<5}{pct(q):.6f}")
print(f"max    {scores[-1]:.6f}")

for thr in (0.5, 0.9, 0.99):
    c = sum(s < thr for s in scores)
    print(f"\nbelow {thr}: {c}/{n} = {100*c/n:.1f}%")

rate = sum(s < 0.9 for s in scores) / n
print()
print("VIABLE - sparse-binary reward has signal, proceed to the smoke config."
      if rate >= 0.05 else
      "NOT VIABLE at this setting - reward is effectively constant. Do NOT launch\n"
      "a training run. Change the task (human-prefix continuation, longer outputs\n"
      "scored on the minimum window, higher temperature) and re-run this gate.")
PY
