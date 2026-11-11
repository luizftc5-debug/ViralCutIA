"""The metric the whole speech-quality claim rests on must itself be correct."""

import pytest

from viralcut.evaluation import normalize_text, timing_report, top_errors, wer


def test_identical_text_is_zero():
    assert wer("O gabinete afirmou que 19 dos feridos", "o gabinete afirmou que 19 dos feridos.").wer == 0.0


def test_counts_substitution_deletion_insertion():
    r = wer("a b c d", "a x c d e")
    assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 1)
    assert r.wer == pytest.approx(2 / 4)
    assert ("sub", "b", "x") in r.errors and ("ins", "", "e") in r.errors


def test_accents_are_part_of_the_word():
    assert wer("não tem", "nao tem").wer == pytest.approx(0.5)
    assert wer("não tem", "nao tem", keep_accents=False).wer == 0.0


def test_empty_reference_counts_invented_words():
    assert wer("", "").wer == 0.0
    assert wer("", "you").wer == 1.0  # a hallucination on silence is a failure, not a rounding error


def test_numbers_and_units_are_normalised_both_ways():
    ref = "stretching 80 km (50 mi) with one hundred percent certainty, nineteen of 3,000"
    hyp = "stretching 80 kilometers (50 miles) with 100% certainty, 19 of 3000"
    assert wer(ref, hyp, lang="en").wer == 0.0
    assert normalize_text("one hundred and five", lang="en") == ["105"]


def test_top_errors_aggregates():
    results = [wer("a b", "a c"), wer("b d", "c d")]
    assert top_errors(results)[0] == (("sub", "b", "c"), 2)


def test_timing_report_flags_words_on_silence():
    import numpy as np

    env = np.full(100, -80.0)
    env[20:40] = -20.0  # speech at 1.0 - 2.0 s
    rep = timing_report([("ok", 1.1, 1.8), ("early", 0.1, 0.5)], env, 0.05, -50.0)
    assert rep["on_silence"] == 1 and rep["words"] == 2
