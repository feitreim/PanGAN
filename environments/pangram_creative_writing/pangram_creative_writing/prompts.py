"""Prompt construction from the vendored lechmazur/writing element pools.

Upstream `primeintellect/creative-writing` fetched these ten pools and both prompt templates
from raw.githubusercontent.com at load time; the upstream repository has since deleted them and
all twelve URLs 404. They are vendored under `data/` and read from disk, never over the network.

Two deliberate departures from upstream: sampling is seeded (upstream used the global `random`
and was not reproducible), and the train/eval splits partition every pool so the two splits
share no element value at all.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

DATA = Path(__file__).parent / "data"

ELEMENTS = (
    "character",
    "object",
    "core_concept",
    "attribute",
    "action",
    "method",
    "setting",
    "timeframe",
    "motivation",
    "tone",
)

Split = Literal["train", "eval"]

TRAIN_FRACTION = 0.8

STORY = re.compile(r"<story>(.*?)</story>", re.DOTALL | re.IGNORECASE)
STORY_OPEN = re.compile(r"<story>(.*)", re.DOTALL | re.IGNORECASE)
STORY_TAG = re.compile(r"</?story>", re.IGNORECASE)
WORD_COUNT_MARKER = re.compile(r"\s*\[\d+\]")


@lru_cache(maxsize=None)
def _pool(name: str) -> tuple[str, ...]:
    text = (DATA / "elements" / f"{name}.txt").read_text(encoding="utf-8")
    # The pools ship with duplicate lines; dict.fromkeys dedupes them, otherwise a value could
    # land in both splits and the two would not be disjoint.
    return tuple(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))


@lru_cache(maxsize=None)
def _split_pool(name: str, seed: int, split: Split) -> tuple[str, ...]:
    entries = list(_pool(name))
    random.Random(f"{seed}/{name}").shuffle(entries)
    cut = int(len(entries) * TRAIN_FRACTION)
    return tuple(entries[:cut] if split == "train" else entries[cut:])


@lru_cache(maxsize=None)
def create_template() -> str:
    return (DATA / "prompts" / "prompt_create.txt").read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def grading_template() -> str:
    return (DATA / "prompts" / "prompt_grading.txt").read_text(encoding="utf-8").strip()


def sample_elements(idx: int, seed: int, split: Split) -> dict[str, str]:
    """One value per pool, drawn only from this split's half of each pool."""
    rng = random.Random(f"{seed}/{split}/{idx}")
    return {name: rng.choice(_split_pool(name, seed, split)) for name in ELEMENTS}


def build_prompt(elements: dict[str, str], min_words: int, max_words: int) -> str:
    required = "\n".join(f"* {name}: {value}" for name, value in elements.items())
    return create_template().format(
        min_count=min_words, max_count=max_words, required_elements=required
    )


def extract_story(reply: str) -> str:
    """The `<story>` body, else the whole reply.

    An unclosed `<story>` is the common case, not an edge case: a small model that runs into
    `max_tokens` is cut off mid-sentence and never emits the closing tag. Measured on 63 real
    0.8B rollouts, only 2 closed the tag while 55 carried the literal opening tag straight
    through to the detector as text no human wrote. So an opening tag alone still opens the
    body, and any surviving tag is stripped.

    Cumulative word-count markers go the same way: the prompt does not ask for them, but they
    are the upstream habit and they are likewise not prose.
    """
    match = STORY.search(reply) or STORY_OPEN.search(reply)
    body = match.group(1) if match else reply
    return WORD_COUNT_MARKER.sub("", STORY_TAG.sub("", body)).strip()
