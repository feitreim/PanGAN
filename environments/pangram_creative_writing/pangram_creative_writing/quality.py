"""A free, deterministic degeneracy check, standing where an LLM judge is too slow and too
expensive to run on every training rollout.

Calibration measured what escaping the detector actually looks like for a 0.8B policy, and it is
not human-like prose. The best escape (`ai_score` 0.543, paying ~55x the median reward) read
"...the fracturing skeletons of the core concept...", leaking the prompt's own element-category
vocabulary into the story; the second best was written entirely in lowercase. Both are cheap to
catch without a model.

A weighted blend of humanness and craft does not close this. With humanness spanning ~55x
between an escape and a typical rollout while craft spans ~5x, a 0.2 craft weight still leaves
gibberish ahead 0.386 to 0.107. A floor does close it: below the threshold the reward is 0, so
degeneracy earns nothing no matter how well it fools the detector.

Each check returns [0, 1] and the score is the weakest of them, so one severe failure gates the
rollout on its own rather than being averaged away by the others.
"""

from __future__ import annotations

import re

SCAFFOLD = (
    "core concept",
    "timeframe",
    "required element",
    "narrative element",
    "word count",
    "<story>",
)
"""Phrases from the prompt that no story narrates. Deliberately *not* the full category list:
six of the ten labels — setting, action, object, method, character, tone — are ordinary English,
and matching on them gated 56 of 59 ordinary calibration rollouts. Only the distinctive
multi-word labels survive, which is what the measured leak ("the fracturing skeletons of the
core concept") actually contained."""

SENTENCE = re.compile(r"[.!?]+(?:\s|$)")
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
MIN_TRIGRAM_WORDS = 30


def capitalization(story: str) -> float:
    """Fraction of sentences opening with a capital. All-lowercase output scores ~0."""
    sentences = [s.strip() for s in SENTENCE.split(story) if s.strip()]
    if not sentences:
        return 0.0
    return sum(s[:1].isupper() for s in sentences) / len(sentences)


def scaffold_clean(story: str) -> float:
    """0 if the story names any element category verbatim. Binary because it is unambiguous —
    no story legitimately narrates the words "core concept" or "timeframe"."""
    lowered = story.lower()
    return float(not any(label in lowered for label in SCAFFOLD))


def trigram_variety(story: str) -> float:
    """Distinct trigrams over total. Coherent prose sits near 1.0; looping text collapses.
    Returns 1.0 below `MIN_TRIGRAM_WORDS`, where the ratio is dominated by sample size — the
    word-count gate is what handles short output."""
    words = [w.lower() for w in WORD.findall(story)]
    if len(words) < MIN_TRIGRAM_WORDS:
        return 1.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    return len(set(trigrams)) / len(trigrams)


def coherence(story: str) -> dict[str, float]:
    """The three checks plus their minimum, all recorded so a gated run is diagnosable."""
    parts = {
        "capitalization": capitalization(story),
        "scaffold_clean": scaffold_clean(story),
        "trigram_variety": trigram_variety(story),
    }
    return {**parts, "coherence": min(parts.values())}
