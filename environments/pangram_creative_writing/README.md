# pangram-creative-writing

Write a short story from ten randomly drawn narrative elements. The reward is how *human* the
[Pangram](https://pangram.com) AI-writing detector judges the prose to be.

```bash
uv pip install -e .
export PANGRAM_API_KEY=...        # detector; billed separately from Prime
uv run eval pangram-creative-writing --harness.id null -m Qwen/Qwen3.5-0.8B -n 8 -r 2
```

`--harness.id null` matters. The v1 default harness is `bash`, an agentic loop in a sandbox;
this is a single-turn writing task and `null` is the plain chat harness it wants.

## Layout

| file | what |
| --- | --- |
| `taskset.py` | task data, scoring, taskset |
| `pangram.py` | async detector client (rate limiter, retries, excerpting) |
| `prompts.py` | element pools, seeded train/eval split, prompt + story parsing |
| `judge.py` | the lechmazur craft rubric, eval-only |
| `data/` | the ten vendored element pools and both prompt templates |

`data/` is vendored on purpose. Upstream `primeintellect/creative-writing` fetched all twelve
files from `raw.githubusercontent.com/lechmazur/writing` at load time and every one of those
URLs now 404s. Nothing here touches the network at import.

## Config

`--taskset.*`

| field | type | default | meaning |
| --- | --- | --- | --- |
| `split` | `train \| eval` | `train` | disjoint prompt sets |
| `num_tasks` | `int` | `200` | prompts to generate |
| `seed` | `int` | `0` | element sampling seed |
| `min_words` | `int` | `400` | hard floor: below it reward is 0 and the detector is **not** called |
| `max_words` | `int` | `700` | ceiling asked for in the prompt text. Soft — nothing enforces it |

The two splits partition every element pool, so a value that appears in a `train` prompt can
never appear in an `eval` one. Sampling is seeded and reproducible; upstream's was neither.

`--taskset.task.*`

| field | type | default | meaning |
| --- | --- | --- | --- |
| `pangram.base_url` | `str` | `https://text.external-api.pangram.com` | |
| `pangram.api_key_var` | `str` | `PANGRAM_API_KEY` | read at taskset load, so a missing key fails immediately |
| `pangram.max_concurrent` | `int` | `4` | in-flight detections process-wide, under the 5 QPS ceiling |
| `pangram.timeout` | `float` | `300.0` | seconds to wait for a terminal `stage` |
| `pangram.poll_interval` | `float` | `0.5` | |
| `pangram.excerpt_words` | `int \| None` | `None` | score an N-word excerpt; `None` scores the whole story |
| `pangram.max_scored_words` | `int` | `1000` | cost guard: exactly one billable unit |
| `judge` | `JudgeConfig \| None` | `None` | when set, the craft rubric runs as a metric |

Training leaves `judge` unset and pays nothing for it. Eval turns it on with
`--taskset.task.judge.model <model>`; it defaults to the Prime Inference gateway and
`PRIME_API_KEY`, and its tokens and cost land in `trace.extra_usage`.

`excerpt_words` picks a contiguous window whose offset is derived from the **task index**, so
every rollout in a group scores the same region. A per-rollout offset would inject selection
variance far larger than the ~6e-4 real signal and GRPO advantages would be mostly noise. The
offset varies across tasks, which stops the policy from humanizing only the opening. The
word-count gate always applies to the **full** story, never the excerpt. The offset and excerpt
length are recorded in `trace.info`.

## Scoring

One reward:

| reward | value |
| --- | --- |
| `humanness` | `1 - ai_score`, or `0.0` when the word-count gate fails |

Metrics (recorded, never summed into reward):

| metric | value |
| --- | --- |
| `ai_score` | word-count-weighted mean of `windows[].ai_assistance_score` |
| `ai_score_logit` | `log(p/(1-p))`, clamped — the better-conditioned view of the plateau |
| `escaped` | `1.0` if `ai_score < 0.5` — **the metric that says whether the run is working** |
| `escaped_soft` | `1.0` if `ai_score < 0.9` |
| `fraction_human`, `fraction_ai` | Pangram's own document labels |
| `word_count` | words in the extracted story |
| `gated` | `1.0` when the rollout was gated and no detector call was made |
| `num_windows` | detector windows returned |
| `craft` | lechmazur rubric in [0,1]; only when `judge` is configured |

Gated rollouts record `word_count`, `gated`, and `escaped`/`escaped_soft` as `0.0`, but **no**
`ai_score`. So a collapse to stubs shows up as `gated` rising, never as a fake `ai_score`
distribution — and because a gated rollout demonstrably did not escape, the escape *rate* stays
over every rollout rather than only the scored ones, where dropping out could inflate it.

Three measured facts the scoring is built around:

- **The detector is close to a step function.** Six very different AI texts scored within
  5.8e-4 of 0.9933; human prose scored 0.017-0.163; nothing in between. Mean `humanness` will
  look flat even in a successful run. Watch the `escaped` *rate*.
- **Short text scores less AI.** A 16-word stub scored 0.786 against ~0.993 for real prose, so
  `1 - ai_score` pays ~30x more for a stub. The word-count gate runs *before* the API call:
  it closes that reward-hacking channel and avoids paying $0.05 for a degenerate rollout.
- **`fraction_ai` is a hard 0/1 label**, never intermediate. It is recorded, never rewarded.

## Cost

Pangram bills $0.05 per 1,000 words with a minimum of one unit per item, so a compliant rollout
is one unit. `max_scored_words` is what keeps it there: an early smoke run produced a degenerate
32,034-word rollout, which is 33 billable units — $1.65 for a single rollout. The cap is set at
exactly one billable unit, and a 0.8B model overshoots the prompt's 700-word ceiling routinely
(measured: 59 of 63 rollouts ran long, median 1,171 words), so the guard earns its keep.

The detector dominates the bill. At 64 rollouts per step that is $3.20/step against roughly
$0.01/step of Prime compute for a 0.8B model.

## Differences from upstream `primeintellect/creative-writing`

- Element pools and rubric are vendored, not fetched (upstream's URLs 404).
- Sampling is seeded, and `train`/`eval` are genuinely disjoint.
- The prompt no longer asks for a cumulative word count `[N]` after every sentence. That text
  would go straight to the detector as something no human writes. Removing it also drops
  upstream's word-count bug.
- The story is actually extracted from `<story>...</story>`. Upstream passes a Python dict repr
  to its judge. An *unclosed* `<story>` still opens the body, which is the common case rather
  than an edge case: a small model that hits `max_tokens` is cut off before the closing tag.
  Measured on 63 real 0.8B rollouts, only 2 closed it and 55 would otherwise have shipped the
  literal tag to the detector as text no human wrote.
- One judge model, not seven. The ensemble buys leaderboard stability at seven times the price.
- The rubric is a metric, never a reward.
