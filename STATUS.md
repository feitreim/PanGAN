# pan-gan

RL-train a small LLM to write prose that the [Pangram](https://www.pangram.com/) AI-writing
detector scores as human-written.

The name is the framing: a GAN where the discriminator is frozen, commercial, and
adversarial by construction. Pangram is a production detector trained to catch exactly the
model family we are optimizing. We do not get to update it, we do not get its gradients, and
we pay $0.05 every time we ask it a question. The only thing we control is the generator.

---

## Status: the whole thing is provisional on one unanswered measurement

**Do not launch a paid training run until the calibration gate passes.** Every config in
this repo is sized and written as if the gate passes; none of it is validated by a real run.

### The step-function finding

We measured Pangram's live API on 2026-07-27 (model version 3.3.2) against six very
different AI-written texts and three human-written literary passages:

| sample | words | `ai_assistance_score` |
|---|---|---|
| Austen, *Pride and Prejudice* | 154 | **0.017309** |
| Twain, *Huckleberry Finn* (vernacular) | 134 | **0.021076** |
| Melville, *Moby-Dick* | 142 | **0.163394** |
| *— gap: nothing at all between 0.164 and 0.993 —* | | |
| AI, roughened literary | 174 | **0.993015** |
| AI, archetypal LLM prose | 136 | **0.993230** |
| AI, mediocre small-model prose | 133 | **0.993255** |
| AI, 626-word crafted literary story | 626 | **0.993330** |
| AI, *deliberately trying to sound human* | 199 | **0.993324** |
| AI, flat "Sarah went to the store" prose | 143 | **0.993597** |

Total spread across six wildly different AI texts: **5.8e-4**.

Read the last three rows together. A careful, deliberate attempt at human-sounding informal
prose — mundane detail, digressions, sentence fragments, a dropped thread — scored 0.993324.
Robotic supermarket prose scored 0.993597. The human-mimicking sample is not meaningfully
closer to human than the robotic one, and it sits mid-pack among the AI samples. Literary
quality, register, roughness and deliberate mimicry all moved the score by less than 0.001,
while the distance to real human text is ~0.98.

The scores are also **bit-exactly deterministic** — the same text submitted three times
returned `0.9932552576065063` every time, spread 0.000e+00. So the tiny differences are real,
they are just real *and meaningless*.

### What that does to the experiment

1. **`reward = 1 - ai_score` is nearly flat.** Every rollout gets ≈0.0067 with a within-group
   spread around 1e-4. GRPO standardizes within the group, so it would faithfully amplify
   4th-decimal variation that — per the table above — does not track human-likeness at all.
   That is optimizing noise with extra steps.
2. **A logit transform does not rescue it.** It is monotone; it cannot manufacture signal
   that is not in the data. AI-side logit spread is ~0.09 against a ~9.0 chasm to human.
3. **There is no partial credit, so there is no gradient to climb.** The policy has to clear
   the entire cliff at once to be paid anything.

This does not necessarily kill the project. It changes the question from *"can the model
climb a humanness gradient"* (it cannot; there is no such gradient) to **"what fraction of
base-model rollouts land below the cliff at all?"** If that fraction is nonzero, the reward
is a **sparse binary** reward, which GRPO handles fine — it is exactly how math and code RL
work. If it is zero, every group is uniform, every advantage is zero, and no amount of
training moves anything.

All nine samples above were written or selected by a large, heavily-RLHF'd model. A 0.8B
model at temperature 1.0 produces far more erratic text, and erratic text is exactly what
might fall off the AI plateau. That is an empirical question, and it is cheap to answer.

### The base model does not obey the word target

Measured while running the gate, and it changed the configs. Qwen3.5-0.8B asked for 400-700
words writes a **median of ~1,171 words** — roughly 2x the ceiling in the prompt. At the
`max_tokens = 1600` an earlier draft used, that truncated about **97% of rollouts mid-sentence**,
and a truncated story still gets sent to the detector. The configs now sample at
`max_tokens = 3072` so a story can actually finish.

Two consequences worth holding onto:

- **Prompt adherence is not a given at this scale.** The prompt's word range is a suggestion the
  model largely ignores. `word_count` is recorded as a metric for exactly this reason — read it,
  do not assume it.
- **Overshoot is not a cost problem, because it was made not to be.**
  `PangramConfig.max_scored_words = 1000` head-truncates before the API call, capping every
  rollout at one billable unit. Without that guard, billing is per 1,000 words: one runaway
  32,034-word rollout cost **$1.65 by itself**.

### The calibration gate

```bash
DRY_RUN=1 bash scripts/calibrate.sh   # resolve the config, spend nothing
bash scripts/calibrate.sh             # 64 rollouts from the base model, ~$3.20
```

It samples the real base model on real task prompts, scores every rollout through the real
environment, and prints the `ai_score` distribution: min, deciles, max, and the count below
0.9 and below 0.5.

| result | verdict |
|---|---|
| **≥ ~5%** of rollouts below 0.9 | Sparse-binary reward is viable. Proceed to the smoke config. |
| **~0%** below 0.9 | **Stop.** The reward is constant. Do not launch. Change the task first. |

### If the gate fails

Do not turn up the learning rate; there is nothing to learn from. Change the task so that
partial success becomes *payable*:

- **Human-prefix continuation.** Prompt the model to continue a human-written opening. The
  human prefix drags some windows down, producing genuinely intermediate document scores and
  therefore a real gradient.
- **Longer outputs, scored on the worst window.** Pangram windows are ~250-370 words, so a
  600-word rollout yields only ~2 windows. Longer outputs give more windows; rewarding the
  *minimum* window score, or the human-segment fraction, pays partial success. Note this lever
  fights the cost guard: `max_scored_words = 1000` caps the scored text at ~3 windows to keep a
  rollout to one billable unit. Raising it to get more windows raises the per-rollout price
  proportionally — 2,000 scored words is $0.10 a rollout, and the whole cost table doubles.
- **Higher sampling temperature**, to widen the rollout distribution and find the tail.

Then re-run the gate. Never skip it.

---

## Architecture

```
prompt (procedurally generated from 10 story elements)
   |
   v
policy: Qwen/Qwen3.5-0.8B  ->  400-700 words of prose
   |
   +--> Pangram detector   ->  ai_score  ->  reward = 1 - ai_score    [TRAINING]
   |
   +--> craft rubric judge ->  craft metric                           [EVAL ONLY]
```

- **Environment:** `environments/pangram_creative_writing/`, a native **verifiers v1**
  taskset (typed `TaskData`/`Task`/`Taskset`, no `load_environment()` legacy bridge). Taskset
  id `pangram-creative-writing`. Built by a separate agent against `CONTRACT.md`.
- **Reward:** exactly one `@vf.reward`, `humanness = 1.0 - ai_score`, or `0.0` when the
  word-count floor is not met (in which case Pangram is never called, so a short rollout is
  free as well as worthless).
- **Craft rubric:** the original lechmazur creative-writing rubric, run as a **metric only**.
  It is never summed into the reward. Turned on during eval by setting a judge model; left
  unset during training, so training makes zero judge calls and costs zero judge dollars.
- **Trainer:** Prime Hosted Training (primary) or self-hosted prime-rl 0.7.0 (fallback). Same
  environment, same reward, same cost model either way.

### Why the craft rubric is an instrument and not a target

Adding it to the reward would make it a second thing to hack, and the two terms would trade
off in ways we could not read. Keeping it out means the eval plot answers one clean question:
*did the prose fall apart while the detector score improved?* `escaped` up and `craft` down is
the signature of a reward hack. That distinction is only legible if `craft` never touches the
gradient.

---

## Cost model

Pangram bills **$0.05 per 1,000 words realtime, minimum one billable unit per item**. A
400-700 word rollout is one unit, so:

> **one rollout = one Pangram call = $0.05**

That equality is only true because the environment enforces it. It bills *per 1,000 words*,
so a long rollout is several units — and one runaway 32,034-word rollout cost **$1.65 on its
own** before the guard existed. `PangramConfig.max_scored_words = 1000` head-truncates the
story before the API call, so a rollout can never exceed one billable unit no matter how long
it runs. Every config in `configs/` sets it explicitly rather than relying on the default.

The floor is enforced from the other side: below `min_words = 400` the reward is 0 **and
Pangram is never called**, so a degenerate short rollout is free as well as unpaid.

Prime Hosted Training charges **$0.06 per 1M train tokens** for Qwen3.5-0.8B. A rollout is
~1.1k tokens, so a rollout costs **$0.000066** of compute. The detector is roughly **750x**
the cost of the compute that produced the text.

### The two shipped configs, arithmetic shown

**`configs/hosted-rl-smoke.toml`** — plumbing check, 1 step.

```
training   1 step  x 16 rollouts x $0.05  = $0.80
eval       1 eval  x  8 examples x $0.05  = $0.40
                                  Pangram = $1.20

compute    24 rollouts x ~1.6k tok = 38k tok @ $0.06/1M  = ~$0.002
judge      8 rubric calls on gpt-5.4-mini                = ~$0.04
                                            Prime wallet = ~$0.05
```

There is **no documented floor** of `batch_size >= 64` or `group_size >= 8`. An earlier draft
of this README asserted one and priced the smoke at $3.60 on that basis; it was wrong. Read off
the real schema: both fields are `ge=1`, the only enforced rule is
`batch_size % group_size == 0` (verified — 9/8 rejected, 8/8 and 16/8 accepted), and prime-rl's
own debug configs ship `batch_size = 2`. 16/8 here is two real groups, the cheapest shape that
still exercises multi-group batching. The hosted backend does its own validation that cannot be
checked without launching; if it rejects 16/8, raise to 64/16 and the smoke costs $3.60 again —
a rejected launch is free.

**`configs/hosted-rl.toml`** — main run, 50 steps.

```
training  50 steps x 64 rollouts x $0.05  = $160.00
eval       6 evals x 32 examples x $0.05  =   $9.60
                                  Pangram = $169.60

compute   3392 rollouts x ~1.6k tok = 5.4M tok @ $0.06/1M    = ~$0.33
judge     192 rubric calls, ~4k in / 0.5k out, gpt-5.4-mini  = ~$1.00
                                                Prime wallet = ~$1.35
```

The 64/16 sizing there *is* a deliberate experiment choice — see the config's own comment on
P(at least one success per group) — not a minimum. The judge arithmetic uses `openai/gpt-5.4-mini`
at the price `prime inference models` lists: $0.75/1M in, $4.50/1M out, so
192 × (4k in + 0.5k out) ≈ $0.58 + $0.43 ≈ $1.01.

**Pangram is 99.3% of the bill.** Cutting cost means cutting *rollouts* — `max_steps` or
`batch_size` — and nothing else. Model size, sequence length and judge choice are all noise
against $0.05/rollout.

### Two separate wallets

| | pays for | balance |
|---|---|---|
| **Prime** | GPU/compute, the rubric judge via Prime Inference | **$50.00** |
| **Pangram** | every detector call | *separate account, separately billed* |

The $50 Prime wallet does **not** pay for Pangram. A 50-step run costs about $1.35 of Prime
and about $170 of Pangram. Budget on the Pangram side. The $50 balance is comfortably enough
for the compute and the judge, and pays for none of the thing that actually costs money.

### Scaling table

| rollouts/step | $/step | 25 steps | 50 steps | 100 steps |
|---|---|---|---|---|
| 64 | $3.20 | $80 | **$160** | $320 |
| 128 | $6.40 | $160 | $320 | $640 |
| 256 | $12.80 | $320 | $640 | $1280 |

Bulk scoring (`POST /bulk`, $0.04/1k words, up to 1000 units per request) would be a ~20%
saving but is not wired up. Scores are deterministic, so caching identical completions is also
safe — and also unused.

---

## Setup

```bash
# 1. secrets. never in a config file.
export PANGRAM_API_KEY=...       # the detector
export PRIME_API_KEY=...         # Prime itself, and the craft-rubric judge
export WANDB_API_KEY=...         # optional, but you will want the plots

# 2. install the environment package locally
uv sync && uv pip install -e environments/pangram_creative_writing

# 3. sanity-check the environment before it costs anything (no model, no Pangram)
uv run validate --taskset.id pangram-creative-writing -n 20 --runtime.type subprocess

# 4. resolve the gate's config without spending anything
DRY_RUN=1 bash scripts/calibrate.sh

# 5. THE GATE  (~$3.20 — read "Status" above before running this)
bash scripts/calibrate.sh
```

### Use `uv run eval` / `uv run validate`, never `prime eval`

`prime eval run` and `prime eval validate` in prime CLI **0.6.20** still dispatch to
the **legacy v0** entrypoint, `python -m verifiers.cli.commands.eval`. Against this
package they fail outright:

```
python -m verifiers.cli.commands.eval: error: unrecognized arguments: --dry-run
```

That argparse has no `--dry-run`, no `--harness.*`, no `--taskset.*`, and it silently
defaults the model to `openai/gpt-4.1-mini`. This is the same stale-scaffold trap as
`prime env init`. The v1 entrypoints are the console scripts verifiers installs into
the venv: **`eval`**, **`validate`**, `serve`, `debug`, `replay`, `init`, `gepa` — i.e.
`uv run eval`, `uv run validate`. `scripts/calibrate.sh` uses `.venv/bin/eval`.

Always pass **`--harness.id null`**. `HarnessConfig.id` defaults to `"bash"`, and a
dry-run resolves to it silently — a bash agent loop is wrong for a prose task. There is
no `"default"` harness either; `import_harness("default")` raises `ModuleNotFoundError`
despite what `prime train init`'s template and the `train-with-environments` skill say.

Note: `.env` in this repo stores `PANGRAM_API_KEY` **double-quoted**. Anything reading it must
strip the quotes (`.strip().strip("\"'")`) or Pangram returns
`401 {"detail":"Invalid API key"}`.

### Path A — Prime Hosted Training (primary)

No pod, no GPUs, no prime-rl checkout.

```bash
prime train configs/hosted-rl-smoke.toml -e PANGRAM_API_KEY -e PRIME_API_KEY   # ~$3.60
prime train configs/hosted-rl.toml       -e PANGRAM_API_KEY -e PRIME_API_KEY   # ~$170
```

`-e KEY` reads the value from your local `$KEY` and injects it into the training container;
`--env-file .env` and a top-level `env_files = [...]` also work. The key has to reach the
**env-server** container, which is where the taskset's Pangram client runs.

Watch it:

```bash
prime train list
prime train logs <run-id>
prime train metrics <run-id>
prime train distributions <run-id>    # reward histogram — this is the one that matters
prime train rollouts <run-id>         # read the actual prose
```

Hosted Training **does** support native v1 `taskset`/`harness` environment blocks. The
`train-with-environments` skill doc claims it is v0-only; that claim is stale. Verify with
`prime train configs --plain`, which lists `taskset` and `harness` as object fields on
`env[]`, and with `prime train init`, whose template documents the v1 shape verbatim.

Hosted runs are the **LoRA/adapter** path for this account: `prime train gpus` reports "No GPU
types available", and the schema's `lora_alpha` and `[adapters]` section corroborate that.

### Path B — self-hosted prime-rl 0.7.0 (fallback)

Needs **2 NVIDIA GPUs minimum** (1 trainer + 1 inference); prime-rl has no single-GPU RL
layout. 2x24GB is enough at 0.8B.

```bash
bash scripts/setup_pod.sh            # clones prime-rl, syncs, installs the env package
cd ~/prime-rl

uv run --no-sync rl @ ../pan-gan/configs/rl.toml --dry-run --output-dir /tmp/dry
uv run --no-sync rl @ ../pan-gan/configs/rl-debug.toml --clean-output-dir     # ~$1.60
uv run --no-sync rl @ ../pan-gan/configs/rl.toml @ ../pan-gan/configs/eval.toml
```

The self-hosted env block is shaped **differently** from the hosted one and this is not
cosmetic. prime-rl 0.7.0 pins a newer verifiers than PyPI's 0.2.1, in which the orchestrator
env entry nests everything under `env.` and hangs the harness off a named agent role:

```toml
[[orchestrator.train.env]]
env.taskset       = { id = "pangram-creative-writing", ... }
env.agent.harness = { id = "null", runtime = { type = "subprocess" } }
env.agent.timeout = { rollout = 600, finalize = 600, scoring = 600 }
pool              = { type = "static", num_workers = 1 }   # on the ENTRY, not env.pool
```

The flat `taskset =` / `harness =` shape shown in the `train-with-environments` skill doc is
what **Hosted Training** takes; prime-rl rejects it outright with
`orchestrator.train.env.0.env / Extra inputs are not permitted`.

`--no-sync` is load-bearing: `uv sync` prunes everything outside prime-rl's lockfile, which
includes our editable environment package, so a bare `uv run` silently uninstalls it. The
install order in `setup_pod.sh` is `uv sync` **then** `uv pip install -e`.

---

## Reading the metrics

The environment records one reward and a set of metrics (names fixed by `CONTRACT.md`):

| name | kind | what it tells you |
|---|---|---|
| `humanness` | **reward** | `1 - ai_score`, or 0 if the word floor failed |
| `escaped` | metric | `1.0` if `ai_score < 0.5` — **the headline number** |
| `escaped_soft` | metric | `1.0` if `ai_score < 0.9` — the early-warning version |
| `ai_score` | metric | word-count-weighted mean over windows |
| `ai_score_logit` | metric | `log(p/(1-p))`, the better-conditioned view of the plateau |
| `fraction_human`, `fraction_ai` | metric | Pangram's own hard document labels |
| `word_count` | metric | words in the extracted story — watch it, the model overshoots ~2x |
| `gated` | metric | `1.0` if the rollout fell below `min_words` and was never scored |
| `num_windows` | metric | how many ~250-370 word windows Pangram split the story into |
| `craft` | metric | rubric score in [0,1]; eval only, never in the reward |

(`gated` is emitted by the taskset but is not listed in `CONTRACT.md`. It is useful — a rising
`gated` rate is the word-count hack being attempted — so it is documented here.)

### Watch `escaped`, not mean `humanness`

This is the single most important thing to know about reading this run. Because the detector
is a step function, **mean `humanness` will look flat and boring even in a successful run.**
Going from a 0% to a 5% escape rate moves the mean from 0.0067 to 0.056 — a change you would
squint at — while representing the entire difference between "the experiment failed" and "the
experiment worked".

Read, in order:

1. **`escaped` / `escaped_soft` rate.** The real signal. Compare every step against the step-0
   baseline, which is why `skip_first_step = false` in every config here.
2. **The zero-advantage / uniform-group rate.** With a sparse binary reward this is the
   signal-density readout. If it is 100%, no gradient is flowing and the run is dead
   regardless of what the loss curve does. This is why `zero_advantage` is configured as a
   **monitor-only** filter rather than an enforcing one — enforcing it would hide the number.
3. **`craft`, against `escaped`.** Both up: real progress. `escaped` up and `craft` down: a
   reward hack. `craft` is the only thing in the setup that can tell you the difference.
4. **Entropy and `mismatch_kl`.** Collapsing entropy on a single-scalar adversarial reward is
   the classic precursor to degenerate output.
5. **The actual prose** (`prime train rollouts`). Read it. Metrics will not tell you the model
   has started writing in Cyrillic.

### `fraction_ai` is not a reward signal

It was **exactly 0.0 or exactly 1.0** in every sample tested, and
`fraction_ai + fraction_ai_assisted + fraction_human == 1.0` exactly. It is a hard label. Using
it as reward gives an all-identical group and zero advantage. The only continuous signal in
the API is `windows[].ai_assistance_score`.

---

## Reward hacking: what will go wrong, and what stops it

A pure-detector reward with no quality term **will** degenerate. The cheapest way to stop
looking like an LLM is to stop looking like language. Known channels:

| channel | defense |
|---|---|
| **Short outputs.** A 16-word input returns `STAGE_SUCCESS` with score 0.786 and `confidence: "Medium"` — it does not error, so it is a silent free win. `1 - 0.786 = 0.214` against `1 - 0.993 = 0.007`: a one-line stub pays **~30x** what a real 600-word story pays. This is by far the strongest gradient in the naive reward, and it points straight at writing nothing. | Hard `min_words = 400` gate in the taskset, checked **before** the API call. Below it, reward is 0 **and Pangram is never called**, so the hack is both unpaid and free. |
| **Gibberish / non-language.** | `gibberish` rollout filter, `enforce = true`. |
| **Degenerate repetition.** | `repetition` rollout filter, `enforce = true`. (Its threshold field is `prob_threshold`, not `threshold` — the upstream docs are wrong.) |
| **Style collapse into an unreadable but low-scoring register.** | The `craft` metric at eval. It cannot *prevent* this, only make it visible — which is why it must be read, not just logged. |
| **Runaway policy drift.** | `kl_tau = 1e-2`, 10x the prime-rl default, on the self-hosted path. Note this is a trust region against the *sampling* policy, not a KL to the base model; there is no reference model in this algorithm. |

Filters are configured **post-batch**, not pre-batch, on purpose: a pre-batch drop frees the
slot and gets resampled, and a resample is another $0.05. Post-batch keeps the rollout visible
in wandb but out of the gradient, for free.

There is no defense against the deepest failure mode: that the model finds some artifact of
Pangram 3.3.2 that is not "sounding human" at all. If it does, we will only find out by
reading the prose. **Read the prose.**

---

## Repo layout

```
configs/
  hosted-rl.toml         Hosted Training, main run       (~$170  Pangram)
  hosted-rl-smoke.toml   Hosted Training, plumbing check (~$1.20 Pangram)
  rl.toml                self-hosted prime-rl, main run  (~$160  Pangram)
  rl-debug.toml          self-hosted prime-rl, smoke     (~$1.60 Pangram)
  eval.toml              periodic-eval overlay for the self-hosted path
  {rl,eval,gepa}/*.toml  stock Prime Lab templates, untouched, unrelated to this experiment
scripts/
  calibrate.sh           THE GATE. Run first.            (~$3.20 Pangram)
  setup_pod.sh           self-hosted pod bootstrap
environments/
  pangram_creative_writing/    the verifiers v1 taskset
```

---

## Verified vs. assumed

Verified by running commands on 2026-07-27:

- Pangram's wire protocol, its determinism, the step-function score distribution, the $0.05
  unit price and the 5 QPS limit.
- The Hosted Training config schema (`prime train configs --plain`), and that it accepts v1
  `taskset`/`harness` objects.
- The hosted model list and pricing; Prime wallet balance $50.00.
- verifiers 0.2.1's v1 API surface, read from installed source.
- That harness id **`"default"` does not exist** — `import_harness("default")` raises
  `ModuleNotFoundError`; `null` and `bash` both resolve. `prime train init`'s template comment
  and the `train-with-environments` skill doc both say `"default"`; both are wrong. The
  tool-less chat loop is **`"null"`**. `HarnessConfig.id` defaults to `"bash"`.
- That `prime eval run` / `prime eval validate` in CLI 0.6.20 dispatch to the **v0** CLI and
  reject every v1 flag. The v1 entrypoints are `uv run eval` / `uv run validate`.
- That the default env-server pool is **elastic**, not static: a resolved config prints
  `type='elastic' max_workers=None multiplex=128`, i.e. unbounded workers added on demand. The
  taskset's `max_concurrent` limiter is per worker process, so the pool must be pinned to
  `{ type = "static", num_workers = 1 }` or the 5 QPS limit is exceeded. (An earlier draft said
  the default was "4 workers" — that is `StaticPoolConfig`'s own `num_workers` default, which
  only applies once you have already chosen a static pool.)
- **The self-hosted `configs/rl.toml`, `rl-debug.toml` and `eval.toml` schemas.** All three were
  constructed by `prime_rl.configs.rl.RLConfig` and accepted, including the composed
  `rl.toml + eval.toml` overlay. Method, reproducible with no GPU: sparse-clone prime-rl 0.7.0
  @ `3b22dd9`, install its slim `packages/prime-rl-configs` (explicitly "no GPU/ML deps") into a
  Python 3.12 venv, install the **pinned** `deps/verifiers` submodule commit, then call
  `RLConfig(**tomllib.load(...))`. The header of `configs/rl.toml` carries the exact commands.
- That the self-hosted and hosted env shapes genuinely **differ**, and neither doc is right about
  both. prime-rl 0.7.0 pins verifiers at submodule commit `b13ba60` (reports `0.2.2.dev17`),
  where the orchestrator env entry is `{ env, pool, id, args, extra_env_kwargs, name, address,
  ratio, sampling, group_size, algo }` — so it is `env.taskset`, `env.agent.harness` (the harness
  hangs off a named *agent role*), `env.agent.timeout`, and `pool` on the entry, **not**
  `env.pool`. Passing the flat `taskset =` / `harness =` shape from the skill doc is rejected with
  `orchestrator.train.env.0.env / Extra inputs are not permitted`. Hosted Training uses verifiers
  0.2.1 and really is flat. Do not unify them.
- That `enable_thinking` does **not** exist on `AutoRendererConfig` (fields: `thinking_retention`,
  `name`) but **does** exist on `Qwen35RendererConfig` (fields: `thinking_retention`, `name`,
  `enable_thinking`, `add_vision_id`, `image_cache_max`). That is why the configs name the
  concrete renderer and set the flag explicitly.
- That there is **no** `batch_size >= 64` / `group_size >= 8` floor; the only enforced rule is
  `batch_size % group_size == 0`.
- That the repetition filter's field is `prob_threshold` (resolved default:
  `window=3000, prob_threshold=0.99`), not `threshold`.
- That the taskset/task fields the configs set all exist with these names: `split`, `num_tasks`,
  `seed`, `min_words`, `max_words`, and `task.pangram.{base_url, api_key_var, max_concurrent,
  timeout, poll_interval, max_scored_words}`, `task.judge`. Confirmed by resolving a real config
  through the installed package, not by reading `CONTRACT.md`.
- That `openai/gpt-5.4-mini` is listed by `prime inference models` at $0.75/1M in, $4.50/1M out.

Assumed, not verified here:

- **Anything the Hosted Training backend validates server-side.** `prime train configs --plain`
  lists the key names and types, and every key in `configs/hosted-rl*.toml` is in that list; the
  `[[env]]` blocks were additionally accepted by `vf.EnvServerConfig` (verifiers 0.2.1). But no
  hosted run was launched, so backend-only constraints — minimum `batch_size`, whether
  `rollouts_per_example` is bounded, adapter/LoRA specifics — are unchecked.
- That the self-hosted configs *run*. They **parse**; they have never been through
  `uv run rl --dry-run` on a real prime-rl install, let alone onto a GPU. Schema validity is not
  runtime validity.
- The escape-rate assumptions every cost and sizing decision rests on. That is what the gate is
  for.

## What could invalidate the whole approach

1. **The calibration gate returns 0%.** The most likely outcome, and the reason nothing has
   been launched. Fix the task, not the hyperparameters.
2. **The escape rate is nonzero but every escape is degenerate** — the only rollouts that fool
   the detector are exactly the ones the gibberish filter drops. Then the reward is real but
   uncorrelated with anything we want.
3. **Pangram updates its model mid-run.** The API reports a version (3.3.2 as measured); a
   bump silently changes the reward function underneath the policy. Log the version.
4. **The policy learns a Pangram-3.3.2-specific artifact.** It would look like success in
   every metric here and generalize to nothing. Only reading the prose catches this.
