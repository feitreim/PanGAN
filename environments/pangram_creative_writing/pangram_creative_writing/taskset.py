"""Write prose that the Pangram detector scores as human-written.

The single reward is `1 - ai_score`. Everything else is a metric, because the detector's output
is close to a step function — measured AI texts cluster within 6e-4 of 0.9933 while human prose
sits at 0.02-0.16 with nothing in between — so mean reward is flat and uninformative and the
`escaped` *rate* is what actually says whether the run is working.

Two rules the scoring depends on, both measured:
  - Short text scores less AI (a 16-word stub scored 0.786), so `1 - ai_score` pays ~30x more
    for a stub than for real prose. The word-count gate closes that reward-hacking channel, and
    running it before the API call also avoids paying $0.05 for a degenerate rollout.
  - `fraction_ai` is a hard 0/1 label, never an intermediate value, so it is recorded but never
    used as reward.
"""

from __future__ import annotations

import verifiers
import verifiers.v1 as vf

if not hasattr(vf, "TaskData"):
    # Prime's Hosted Training image raised `no attribute 'TaskData'` here while the same
    # package imports fine locally and passes the Hub's own install-and-import CI. Its
    # vendored verifiers also satisfies `>=0.2.1` without exposing the attribute, which no
    # published release does (0.2.1 and every 0.2.2.dev have it). A bare AttributeError says
    # nothing about which build is actually loaded, so report the build itself.
    raise RuntimeError(
        f"verifiers {getattr(verifiers, '__version__', '?')} at {verifiers.__file__} has no "
        f"verifiers.v1.TaskData. Public exports: "
        f"{sorted(n for n in dir(vf) if not n.startswith('_'))}"
    )

from pangram_creative_writing.judge import CraftJudge
from pangram_creative_writing.pangram import (
    PangramClient,
    PangramConfig,
    PangramResult,
    excerpt,
    read_api_key,
    truncate,
)
from pangram_creative_writing.prompts import Split, build_prompt, extract_story, sample_elements
from pangram_creative_writing.quality import coherence

ESCAPE_THRESHOLD = 0.5
SOFT_ESCAPE_THRESHOLD = 0.9


class PangramCreativeWritingData(vf.TaskData):
    elements: dict[str, str]
    """One value drawn from each of the ten lechmazur element pools."""
    min_words: int
    """Below this the reward is 0 and the detector is not called."""
    max_words: int
    """The ceiling the prompt asks for. Soft: nothing enforces it, and a 0.8B model overshoots it
    routinely. The cost cap on what reaches the detector is `PangramConfig.max_scored_words`."""


class PangramCreativeWritingTaskConfig(vf.TaskConfig):
    pangram: PangramConfig = PangramConfig()
    judge: vf.JudgeConfig | None = None
    """When set, the craft rubric runs as a metric. Training leaves it unset and pays nothing."""
    coherence_floor: float = 0.5
    """Reward is 0 below this `quality.coherence`, and the detector is not called.

    The detector's blind spot is weirdness, so ascent on `1 - ai_score` alone walks toward
    gibberish: calibration's best escape paid ~55x the median for prose that had degenerated.
    A floor removes that payout entirely, where a weighted craft term merely discounts it (see
    `quality`).

    0.5 was chosen by replaying the 62 scored calibration rollouts: it gates both degenerate
    escapes (the "core concept" leak and the all-lowercase one) while sparing the third, which
    reads as real prose at 0.757. The sweep is flat from 0.3 to 0.7 — 15 vs 16 ordinary
    rollouts gated — and 0.8 starts taking legitimate escapes. Set 0.0 to disable the gate and
    observe the unfiltered detector distribution, which is what calibration wants."""


class PangramCreativeWritingTask(
    vf.Task[PangramCreativeWritingData, vf.State, PangramCreativeWritingTaskConfig]
):
    async def finalize(self, trace: vf.Trace) -> None:
        """One detector call per rollout, its result shared by every signal below and left in
        `trace.info` so `traces.jsonl` is inspectable.

        Both gates run before the call, never after: a gated rollout scores 0 whatever the
        detector would have said, so paying $0.05 to find out buys nothing."""
        story = extract_story(trace.last_reply)
        word_count = len(story.split())
        scores = coherence(story)
        trace.info.update(story=story, word_count=word_count, **scores)
        if word_count < self.data.min_words:
            trace.info["gate"] = "word_count"
            return
        if scores["coherence"] < self.config.coherence_floor:
            trace.info["gate"] = "coherence"
            return
        config = self.config.pangram
        if config.excerpt_words:
            scored, offset = excerpt(story, config.excerpt_words, self.data.idx)
        else:
            scored, offset = truncate(story, config.max_scored_words), 0
        trace.info.update(excerpt_offset=offset, excerpt_word_count=len(scored.split()))
        result = await PangramClient(config).detect(scored)
        trace.info["pangram"] = result.model_dump()

    def _result(self, trace: vf.Trace) -> PangramResult | None:
        payload = trace.info.get("pangram")
        return PangramResult.model_validate(payload) if payload else None

    @vf.reward
    async def humanness(self, trace: vf.Trace) -> float:
        result = self._result(trace)
        return 1.0 - result.ai_score if result else 0.0

    @vf.metric
    async def ai_score(self, trace: vf.Trace) -> dict[str, float]:
        result = self._result(trace)
        if result is None:
            # A gated rollout was never scored, so it has no `ai_score` — but it demonstrably did
            # not escape. Recording that keeps the escape *rate* over every rollout instead of
            # only the scored ones, so it cannot be inflated by rollouts dropping out of the mean.
            return {"escaped": 0.0, "escaped_soft": 0.0}
        return {
            "ai_score": result.ai_score,
            "ai_score_logit": result.ai_score_logit,
            "escaped": float(result.ai_score < ESCAPE_THRESHOLD),
            "escaped_soft": float(result.ai_score < SOFT_ESCAPE_THRESHOLD),
            "fraction_human": result.fraction_human,
            "fraction_ai": result.fraction_ai,
            "num_windows": float(len(result.windows)),
        }

    @vf.metric
    async def word_count(self, trace: vf.Trace) -> dict[str, float]:
        gate = trace.info.get("gate")
        return {
            "word_count": float(trace.info.get("word_count", 0)),
            "gated": float(gate is not None),
            # Split by cause: a rising `gated_word_count` is the short-stub hack being tried,
            # a rising `gated_coherence` is the degeneracy hack. They call for opposite fixes.
            "gated_word_count": float(gate == "word_count"),
            "gated_coherence": float(gate == "coherence"),
        }

    @vf.metric
    async def coherence(self, trace: vf.Trace) -> dict[str, float]:
        keys = ("coherence", "capitalization", "scaffold_clean", "trigram_variety")
        return {key: float(trace.info[key]) for key in keys if key in trace.info}

    @vf.metric
    async def craft(self, trace: vf.Trace) -> dict[str, float]:
        if self.config.judge is None:
            return {}
        judged = await CraftJudge(self.config.judge).evaluate(
            trace=trace, story=trace.info.get("story", ""), elements=self.data.elements
        )
        return {"craft": judged.parsed}


class PangramCreativeWritingConfig(vf.TasksetConfig):
    split: Split = "train"
    """`train` and `eval` partition every element pool, so they share no element value."""
    num_tasks: int = 200
    seed: int = 0
    min_words: int = 400
    max_words: int = 700
    task: PangramCreativeWritingTaskConfig = PangramCreativeWritingTaskConfig()


class PangramCreativeWritingTaskset(
    vf.Taskset[PangramCreativeWritingTask, PangramCreativeWritingConfig]
):
    def load(self) -> list[PangramCreativeWritingTask]:
        read_api_key(self.config.task.pangram.api_key_var)
        return [
            PangramCreativeWritingTask(self._data(idx), self.config.task)
            for idx in range(self.config.num_tasks)
        ]

    def _data(self, idx: int) -> PangramCreativeWritingData:
        elements = sample_elements(idx, self.config.seed, self.config.split)
        return PangramCreativeWritingData(
            idx=idx,
            prompt=build_prompt(elements, self.config.min_words, self.config.max_words),
            elements=elements,
            min_words=self.config.min_words,
            max_words=self.config.max_words,
        )
