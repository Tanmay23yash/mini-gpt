"""Tests for the Stage 0 data pipeline (parsing and cleaning, no downloads)."""

import pytest

from minigpt.data import clean_tweet, code_mixing_index, leakage_key, normalize_text, parse_conll

CONLL_SAMPLE = (
    "meta\t101\tpositive\n"
    "@\tO\nuser\tEng\n_\tO\nname\tEng\n"
    "bahut\tHin\nbadhiya\tHin\nmovie\tEng\n!\tO\n"
    "\n"
    "meta\t102\n"
    "kya\tHin\nbaat\tHin\n"
)


def test_parse_conll_reads_uid_label_tokens() -> None:
    tweets = parse_conll(CONLL_SAMPLE)
    assert [t.uid for t in tweets] == ["101", "102"]
    assert tweets[0].label == "positive"
    assert tweets[1].label is None  # the test file has no labels
    assert tweets[0].tokens[-4:] == ["bahut", "badhiya", "movie", "!"]
    assert tweets[0].langs[-4:] == ["Hin", "Hin", "Eng", "O"]


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ("@ tha _ rami I like iTunes . Better than others I ’ ve tried .".split(),
         "I like iTunes. Better than others I've tried."),
        ("RT @ gfriend __ phrp Happy birthday ! https // t . co / 1TcBUH6Jp1".split(), "Happy birthday!"),
        ("@_ Shahbazkhan1 sai khel gya # cwc19 # PAKvWI".split(), "sai khel gya #cwc19 #PAKvWI"),
        ("Fir se PM banne ke liye … https // t co / Mrvgz4dHqE".split(), "Fir se PM banne ke liye"),
        ("tou ðŸ˜… ( haha )".split(), "tou 😅 (haha)"),
        ("Satguru ji https // t c".split(), "Satguru ji"),
        ("for interviews ! https // tco".split(), "for interviews!"),
        ("Kisan ' - Archives https".split(), "Kisan ' - Archives"),
        ("as a # videsh _ mantri".split(), "as a #videsh_mantri"),
        ("slogan of ' Jai Jawan ' , don ' t".split(), "slogan of ' Jai Jawan ', don't"),
    ],
)
def test_clean_tweet(tokens: list[str], expected: str) -> None:
    assert clean_tweet(tokens) == expected


def test_normalize_text_collapses_whitespace_and_quotes() -> None:
    assert normalize_text("  kya   “baat”\t hai  ") == 'kya "baat" hai'


def test_code_mixing_index() -> None:
    assert code_mixing_index(["Hin", "Hin", "O"]) == 0.0  # monolingual
    assert code_mixing_index(["Hin", "Eng", "O", "EMT"]) == pytest.approx(50.0)  # even mix
    assert code_mixing_index(["Hin", "Hin", "Hin", "Eng"]) == pytest.approx(25.0)
    assert code_mixing_index(["O", "EMT"]) == 0.0  # no language tokens


def test_leakage_key_ignores_case_and_punctuation() -> None:
    assert leakage_key("Bhai, KYA baat hai!!", 10) == leakage_key("bhai kya baat hai", 10) == "bhaikyabaa"
