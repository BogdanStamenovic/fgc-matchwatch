from __future__ import annotations

from fgc_matchwatch import align


def W(t0: float, text: str, step: float = 0.4) -> list[list[object]]:
    return [[t0 + i * step, t0 + i * step + 0.3, w] for i, w in enumerate(text.split())]


def test_parse_number_words_and_digits() -> None:
    toks = ["one", "hundred", "twenty", "three"]
    assert align.parse_number(toks, 0) == (123, 4)
    assert align.parse_number(["37"], 0) == (37, 1)
    assert align.parse_number(["forty", "eight", "and"], 0) == (48, 2)
    assert align.parse_number(["hello"], 0) == (None, 0)
    assert align.parse_number(["one", "hundred", "and", "eight"], 0) == (108, 4)
    assert align.parse_number(["three", "two", "one"], 0) == (3, 1)
    assert align.parse_number(["twenty", "one"], 0) == (21, 2)


def test_countdown_tolerates_misheard_one() -> None:
    toks = align.tokens(W(100, "everyone ready in three two wide we have team kenya"))
    cds = align.countdowns(toks)
    assert len(cds) == 1 and 100 < cds[0] < 105


def test_two_one_go() -> None:
    toks = align.tokens(W(50, "hands off there we go two one go robots in motion"))
    assert align.countdowns(toks)


def test_match_mentions() -> None:
    toks = align.tokens(W(0, "field number five match number three in three two one"))
    assert (1.2, 3) in [(round(t, 1), n) for t, n in align.match_mentions(toks)]


def test_country_spotting_fuzzy_and_aliases() -> None:
    idx = align.CountryIndex.build({"ANT": "Antigua and Barbuda", "CHN": "People's Republic of China",
                                    "PER": "Peru", "USA": "United States of America"})
    hits = idx.spot(align.tokens(W(0, "team antiga and barbuda then china and the usa not peruvian")))
    codes = [c for _, c in hits]
    assert "ANT" in codes and "CHN" in codes and "USA" in codes and "PER" not in codes


def _roll(t: float, a: list[str], b: list[str]) -> list[list[object]]:
    return W(t, "on the red alliance " + " ".join(a) + " and blue " + " ".join(b)
             + " three two one go", step=1.0)


def test_align_orders_matches_and_uses_teams() -> None:
    names = {c: c.lower() for c in ["kenya", "germany", "chile", "peru", "italy", "spain",
                                    "japan", "nepal", "ghana", "egypt", "chad", "mali"]}
    names = {k.upper(): v for k, v in names.items()}
    words = (_roll(1000, ["kenya", "germany", "chile"], ["peru", "italy", "spain"])
             + _roll(1500, ["japan", "nepal", "ghana"], ["egypt", "chad", "mali"]))
    specs = [
        align.MatchSpec("t2-1", 1, 1000.0 + 700, ["KENYA", "GERMANY", "CHILE", "PERU", "ITALY", "SPAIN"]),
        align.MatchSpec("t2-7", 7, 1500.0 + 700, ["JAPAN", "NEPAL", "GHANA", "EGYPT", "CHAD", "MALI"]),
    ]
    res = align.align(words, 700.0, specs, names)
    by = {p.key: p for p in res}
    assert by["t2-1"].start is not None and abs(by["t2-1"].start - 1012) < 5
    assert by["t2-7"].start is not None and abs(by["t2-7"].start - 1512) < 5
    assert by["t2-1"].confident and by["t2-7"].confident


def test_align_empty_transcript() -> None:
    res = align.align([], 0.0, [align.MatchSpec("t2-1", 1, 10.0, ["A"])], {"A": "a"})
    assert res[0].start is None and not res[0].confident


def test_rock_paper_scissors_countdown_is_not_a_start() -> None:
    toks = align.tokens(W(0, "we're gonna do three two one rock paper scissors shoot who won"))
    assert align.countdowns(toks) == []


def test_match_followed_by_comma_is_not_a_callout() -> None:
    toks = align.tokens(W(0, "less than a minute on the match, 22 points on the board"))
    assert align.match_mentions(toks) == []
