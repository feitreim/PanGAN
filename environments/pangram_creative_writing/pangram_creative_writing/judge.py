"""The lechmazur craft rubric, as an eval-only metric.

`vf.RubricJudge` does not fit: it asks for choice-based verdicts, while this rubric asks for
eighteen continuous 0.0-10.0 grades (Q1-Q8 craft, Q9 A-J element integration). The vendored
template is upstream's verbatim, but this runs a single judge model rather than upstream's
seven-model ensemble — the ensemble buys leaderboard stability we do not need, at seven times
the price per rollout.
"""

from __future__ import annotations

import re

import verifiers.v1 as vf

from pangram_creative_writing.prompts import ELEMENTS, grading_template

GRADE = re.compile(r"<question>(.*?)</question>\s*<grade>(.*?)</grade>", re.DOTALL)

CRAFT_KEYS = tuple(str(i) for i in range(1, 9))
ELEMENT_KEYS = tuple(f"9 {letter}" for letter in "ABCDEFGHIJ")

CRAFT_WEIGHT = 0.6
ELEMENT_WEIGHT = 0.4
POWER = 0.5
"""Upstream's Hölder mean exponent."""

MAX_GRADE = 10.0


def parse_grades(text: str) -> dict[str, float]:
    """Numeric grades by question key. `N/A` is dropped, and the block weights renormalize
    over whatever is left."""
    grades = {}
    for question, grade in GRADE.findall(text):
        grade = grade.strip()
        if grade.upper() != "N/A":
            grades[question.strip()] = float(grade)
    return grades


def power_mean(grades: dict[str, float]) -> float:
    """Weighted power mean on the 0-10 scale: 60% craft, 40% elements, uniform within block."""
    craft = [grades[key] for key in CRAFT_KEYS if key in grades]
    elements = [grades[key] for key in ELEMENT_KEYS if key in grades]
    if not craft:
        raise ValueError("rubric judge returned no Q1-Q8 grades")
    total = CRAFT_WEIGHT + (ELEMENT_WEIGHT if elements else 0.0)
    weighted = sum(CRAFT_WEIGHT / len(craft) * grade**POWER for grade in craft)
    if elements:
        weighted += sum(ELEMENT_WEIGHT / len(elements) * grade**POWER for grade in elements)
    return (weighted / total) ** (1 / POWER)


class CraftJudge(vf.Judge[float]):
    def build_messages(self, story: str, elements: dict[str, str]) -> str:
        return grading_template().format(
            story=story, **{name: elements.get(name, "None") for name in ELEMENTS}
        )

    def parse(self, response: vf.JudgeResponse[float]) -> float:
        grades = parse_grades(response.text)
        if not grades:
            raise ValueError(f"rubric judge returned no <grade> tags: {response.text[:400]!r}")
        return power_mean(grades) / MAX_GRADE
