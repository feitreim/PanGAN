"""Regression tests for the two pieces whose behavior was measured rather than assumed.

Every case here is anchored to a real observation from run 1 or from calibration, not to an
invented example — the failures these guard against all actually happened once.

    uv run --with pytest pytest environments/pangram_creative_writing/tests
"""

import pytest

from pangram_creative_writing.prompts import extract_story
from pangram_creative_writing.quality import MARKUP, coherence
from pangram_creative_writing.taskset import (
    ESCAPE_THRESHOLD,
    SOFT_ESCAPE_THRESHOLD,
    shape_reward,
)

CAP = SOFT_ESCAPE_THRESHOLD


class TestRewardCap:
    """The uncapped reward paid most for the worst writing (run 1: within escapes,
    corr(ai_score, craft) = +0.524). Capping is what removes that."""

    def test_escaping_harder_pays_nothing_extra(self):
        # The whole point: 0.64 was the best-written escape in the run, 0.11 the worst-written.
        # Uncapped they were worth 0.36 and 0.89; capped they must be identical.
        assert shape_reward(0.64, CAP) == shape_reward(0.11, CAP) == 1.0

    def test_reaching_the_cap_is_exactly_one(self):
        assert shape_reward(CAP, CAP) == pytest.approx(1.0)

    def test_below_the_cap_the_gradient_survives(self):
        # 5% -> 53% came from ordering in the detected region. Capping must not flatten it.
        detected, closer = shape_reward(0.993, CAP), shape_reward(0.95, CAP)
        assert 0.0 < detected < closer < 1.0

    def test_ordering_is_strictly_monotone_below_the_cap(self):
        scores = [0.999, 0.99, 0.97, 0.95, 0.91]
        rewards = [shape_reward(s, CAP) for s in scores]
        assert rewards == sorted(rewards)

    def test_none_restores_the_uncapped_reward(self):
        assert shape_reward(0.11, None) == pytest.approx(0.89)
        assert shape_reward(0.993, None) == pytest.approx(0.007)

    def test_hard_escape_is_not_special_cased(self):
        # `escaped` (<0.5) is a metric only; it must earn no more than a soft escape.
        assert shape_reward(ESCAPE_THRESHOLD - 0.01, CAP) == shape_reward(CAP - 0.01, CAP) == 1.0


class TestMarkupGate:
    """Both directions cost real money to learn: the first regex gated 56 of 59 ordinary
    rollouts, and a later one false-positived on a legitimate escape."""

    @pytest.mark.parametrize(
        "story",
        [
            "<center STYLE=sorrel tranquility>. In the summer of 1985, during the training",
            "<div class='story'>The rain fell.</div>",
            "</p> The wind howled.",
            "<br>She left.",
        ],
    )
    def test_markup_is_caught(self, story):
        assert MARKUP.search(story), f"missed markup in {story!r}"

    @pytest.mark.parametrize(
        "story",
        [
            # From a real escape at ai_score 0.880 -- gating this would delete signal.
            "<Right now, not once before... not ever!> she thought, bitterly.",
            "He filed it under <miscellaneous items, mostly broken> and forgot.",
            "The clock read 5 < 10 minutes to midnight.",
            "He wrote a < b on the board, then erased it.",
            "The angle was < 90 degrees, she noted.",
        ],
    )
    def test_prose_is_not_markup(self, story):
        assert not MARKUP.search(story), f"false positive on {story!r}"


class TestCoherence:
    def test_weakest_link_not_average(self):
        """One severe failure must gate on its own rather than be averaged away — a 0.2 craft
        weight left gibberish 3.6x ahead, which is why this is a floor."""
        scores = coherence("</div> " + "The bell rang. Alice turned. " * 20)
        assert scores["markup_free"] == 0.0
        assert scores["coherence"] == 0.0

    def test_ordinary_prose_clears_the_floor(self):
        """SCAFFOLD deliberately omits the six element categories that are ordinary English
        (setting, action, object, method, character, tone); matching them gated 56/59."""
        story = (
            "The character of the room changed. She took the object from the table, "
            "considered her method, and set the tone for what came next. Her action was "
            "quiet. The setting held its breath."
        )
        assert coherence(story)["coherence"] >= 0.5

    def test_prompt_vocabulary_leak_is_gated(self):
        """The measured leak: "the fracturing skeletons of the core concept"."""
        assert coherence("the fracturing skeletons of the core concept")["scaffold_clean"] == 0.0

    def test_all_lowercase_is_penalized(self):
        assert coherence("she walked home. it was raining. nobody spoke.")["capitalization"] == 0.0

    def test_looping_text_collapses_trigram_variety(self):
        assert coherence("the wind blew cold " * 40)["trigram_variety"] < 0.2

    def test_token_soup_is_gated_even_when_every_other_check_passes(self):
        """A real 4B temperature-1.3 escape (ai_score 0.048). It has high trigram variety by
        construction, no markup, no scaffold leak and correct capitalization — seven such
        rollouts cleared the 0.5 floor with coherence up to 1.00 before `english` existed."""
        soup = (
            "The fog in Grafton hadn't organized into carrying sea. Elara shaved cream-heavy "
            "dairy sugterr-subsidence all together into one透气, Parking "
            "Pan侵权责任密 together building triples中考"
            "国家一级工作日 shorter than_shop vêm juge."
        )
        assert coherence(soup)["english"] == 0.0
        assert coherence(soup)["coherence"] == 0.0

    def test_ordinary_punctuation_is_not_foreign(self):
        """Curly quotes and em dashes are why this is a density threshold, not a presence one —
        54% of run-1 rollouts contained some non-ASCII."""
        story = "“Don’t,” she said — and the door closed behind her forever."
        assert coherence(story)["english"] == 1.0

    def test_short_text_is_left_to_the_word_count_gate(self):
        """Below MIN_TRIGRAM_WORDS the ratio is dominated by sample size, not repetition."""
        assert coherence("A short opening line.")["trigram_variety"] == 1.0


class TestExtractStory:
    def test_unclosed_tag_still_yields_the_story(self):
        """55 of 63 calibration rollouts never closed the tag, and a parser requiring the closer
        shipped literal `<story>` markup to the detector."""
        assert extract_story("<story>\nThe bell rang.").strip() == "The bell rang."

    def test_wrapper_is_stripped_so_it_cannot_trip_the_markup_gate(self):
        story = extract_story("<story>The bell rang.</story>")
        assert "<story>" not in story
        assert coherence(story)["markup_free"] == 1.0
