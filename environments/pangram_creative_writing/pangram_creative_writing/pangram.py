"""Async client for the Pangram AI-detection API.

The public API is asynchronous only: `POST /task` hands back a `task_id`, and the caller
polls `GET /task/{task_id}` until `stage` is terminal. There is no synchronous endpoint,
and `STAGE_FAILED` arrives with HTTP 200 — so failure is read off `stage`, never off the
status code.

The official `pangram-sdk` is synchronous `requests`, which would block the rollout event
loop, hence this module.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
from functools import lru_cache

import httpx
from pydantic import BaseModel, ConfigDict
from pydantic_config import BaseConfig

QPS_LIMIT = 5.0
"""Documented Pangram rate limit (pangram.com/solutions/api)."""

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
"""Transient. Everything else — 400/401/402/413/422 — is terminal and must not be retried."""

MAX_ATTEMPTS = 5


class PangramError(RuntimeError):
    """The detector could not produce a score."""


class PangramCreditsError(PangramError):
    """HTTP 402: the Pangram account is out of credit. Nothing will succeed until it is topped up."""


class PangramConfig(BaseConfig):
    base_url: str = "https://text.external-api.pangram.com"
    api_key_var: str = "PANGRAM_API_KEY"
    max_concurrent: int = 4
    """In-flight detections process-wide, held under the documented 5 QPS ceiling."""
    timeout: float = 300.0
    poll_interval: float = 0.5
    excerpt_words: int | None = None
    """Score only an N-word contiguous excerpt instead of the whole story (`None` = whole story)."""
    max_scored_words: int = 1000
    """Cost guard, not a scoring knob: Pangram bills per 1,000 words with a one-unit minimum, so
    this is exactly one billable unit. A degenerate rollout that runs to 32k words would otherwise
    cost $1.65 to score by itself. Stories the prompt asked for (400-700 words) never reach it."""


class PangramWindow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    ai_assistance_score: float
    word_count: int = 0
    label: str = ""
    confidence: str = ""
    start_index: int = 0
    end_index: int = 0


class PangramResult(BaseModel):
    """The fields of a `STAGE_SUCCESS` payload that scoring reads."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    version: str = ""
    prediction: str = ""
    prediction_short: str = ""
    fraction_ai: float = 0.0
    fraction_ai_assisted: float = 0.0
    fraction_human: float = 0.0
    windows: list[PangramWindow] = []

    @property
    def ai_score(self) -> float:
        """Word-count-weighted mean window score — the only continuous signal Pangram returns.
        `fraction_ai` is a hard 0/1 label and is useless as a reward."""
        if not self.windows:
            raise PangramError("Pangram returned no windows, so there is no score to read")
        total = sum(window.word_count for window in self.windows)
        if total <= 0:
            return sum(w.ai_assistance_score for w in self.windows) / len(self.windows)
        return sum(w.ai_assistance_score * w.word_count for w in self.windows) / total

    @property
    def ai_score_logit(self) -> float:
        p = min(max(self.ai_score, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))


def read_api_key(var: str) -> str:
    """The key as written in `.env` is double-quoted; unquoted it is a 401."""
    key = os.environ.get(var, "").strip().strip("\"'")
    if not key:
        raise PangramError(
            f"${var} is unset or empty; export the Pangram API key (e.g. via .env) before running"
        )
    return key


def excerpt(text: str, num_words: int, seed: int) -> tuple[str, int]:
    """A contiguous `num_words`-word window of `text`, with its word offset.

    The window sits at a *fraction* of the way through the story that depends on `seed` alone,
    so every rollout of a task scores the same relative region however long it ran. Drawing an
    absolute offset instead would make it a function of the rollout's own length, i.e. a fresh
    draw per rollout — and with the AI-side score spread measured at ~6e-4, that selection noise
    would dwarf the signal GRPO compares within a group.
    """
    words = text.split()
    if len(words) <= num_words:
        return text, 0
    offset = round(random.Random(seed).random() * (len(words) - num_words))
    return " ".join(words[offset : offset + num_words]), offset


def truncate(text: str, num_words: int) -> str:
    """The first `num_words` words, for the cost guard. Unlike `excerpt` this is not a sampling
    decision — it only stops a runaway rollout from being billed as several units."""
    words = text.split()
    return text if len(words) <= num_words else " ".join(words[:num_words])


class _Gate:
    """Process-wide concurrency + QPS ceiling shared by every client in this process."""

    def __init__(self, max_concurrent: int, qps: float) -> None:
        self.slots = asyncio.Semaphore(max_concurrent)
        self.spacing = 1.0 / qps
        self.lock = asyncio.Lock()
        self.next_at = 0.0

    async def pace(self) -> None:
        async with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.spacing
        if wait:
            await asyncio.sleep(wait)


@lru_cache(maxsize=None)
def _gate(max_concurrent: int, qps: float) -> _Gate:
    return _Gate(max_concurrent, qps)


class PangramClient:
    """One detection per `detect()` call. Cheap to construct; build it where you use it."""

    def __init__(self, config: PangramConfig | None = None) -> None:
        self.config = config or PangramConfig()
        self.key = read_api_key(self.config.api_key_var)

    async def detect(self, text: str) -> PangramResult:
        gate = _gate(self.config.max_concurrent, QPS_LIMIT)
        async with gate.slots:
            async with httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(60.0),
                headers={"x-api-key": self.key},
            ) as http:
                task_id = (await self._request(http, gate, "POST", "/task", json={"text": text}))[
                    "task_id"
                ]
                return await self._poll(http, gate, task_id)

    async def _poll(self, http: httpx.AsyncClient, gate: _Gate, task_id: str) -> PangramResult:
        deadline = time.monotonic() + self.config.timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(self.config.poll_interval)
            payload = await self._request(http, gate, "GET", f"/task/{task_id}")
            stage = payload.get("stage", "")
            if stage == "STAGE_SUCCESS":
                result = PangramResult.model_validate(payload)
                if not result.windows:
                    raise PangramError(f"Pangram task {task_id} returned no windows: {payload}")
                return result
            if stage == "STAGE_FAILED":
                raise PangramError(f"Pangram task {task_id} failed: {payload}")
        raise PangramError(f"Pangram task {task_id} did not finish in {self.config.timeout}s")

    async def _request(
        self, http: httpx.AsyncClient, gate: _Gate, method: str, path: str, **kwargs
    ) -> dict:
        for attempt in range(MAX_ATTEMPTS):
            await gate.pace()
            try:
                response = await http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise PangramError(f"Pangram {method} {path} unreachable: {exc}") from exc
                await asyncio.sleep(2.0**attempt)
                continue
            if response.status_code == 402:
                raise PangramCreditsError(
                    f"Pangram returned 402 INSUFFICIENT CREDITS — top up the Pangram account: "
                    f"{response.text}"
                )
            if response.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(2.0**attempt)
                continue
            if response.status_code >= 400:
                raise PangramError(
                    f"Pangram {method} {path} -> {response.status_code}: {response.text}"
                )
            return response.json()
        raise PangramError(f"Pangram {method} {path} exhausted {MAX_ATTEMPTS} attempts")
