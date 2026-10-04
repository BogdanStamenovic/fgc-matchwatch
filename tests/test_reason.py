from __future__ import annotations

import pytest

from fgc_matchwatch import reason


def W(t0: float, text: str) -> list[list[object]]:
    return [[t0 + i * 0.5, t0 + i * 0.5 + 0.4, w] for i, w in enumerate(text.split())]


WORDS = W(0, "team kenya already going towards the accelerator and scoring lots of units "
             "while the blue side waits") + W(200, "nothing here about anyone at all for a while")
NAMES = {"KEN": "Kenya", "PAR": "Paraguay"}


def test_parse_json_variants() -> None:
    assert reason.parse_json('```json\n{"teams": {}}\n```') == {"teams": {}}
    assert reason.parse_json('sure! {"teams": {"A": {}}} done') == {"teams": {"A": {}}}
    with pytest.raises(ValueError):
        reason.parse_json("no json")
    with pytest.raises(TypeError):
        reason.parse_json('{"x": 1}')


def test_verify_keeps_backed_drops_invented_and_misattributed() -> None:
    raw = {"teams": {
        "KEN": {"summary": "Kenya went for the accelerator.",
                "goodAt": [{"claim": "accelerator", "quote": "already going towards the accelerator"},
                           {"claim": "climbing", "quote": "climbed all the way to level four"}],
                "climbZone": {"value": "3", "quote": "made up"}},
        "PAR": {"summary": "Paraguay did stuff.",
                "goodAt": [{"claim": "scoring", "quote": "scoring lots of units while the"}]},
    }}
    ver = reason.Verifier(WORDS, NAMES)
    out = reason.verify(raw, ["KEN", "PAR"], ver)
    assert out["KEN"].good_at == ["accelerator"]
    assert out["KEN"].climb_zone is None
    assert out["KEN"].dropped == 2 and out["KEN"].kept == 1
    assert out["KEN"].evidence == ["already going towards the accelerator"]
    # Paraguay is never named near that quote: dropped, and its summary with it.
    assert out["PAR"].good_at == [] and out["PAR"].summary == ""


def test_prompt_is_year_aware() -> None:
    ctx = reason.MatchContext("2026", "t2-1", "Ranking Match 1", [("SRB", "Serbia")],
                              [("KAZ", "Kazakhstan")], 10, 12, None, WORDS)
    p = reason.build_prompt(ctx, 0.0)
    assert "SUPPRESSION UNIT" in p and "BRACE" in p and "biodiversity" not in p.lower().split("transcript")[0]
    ctx.year = "2025"
    assert "ROPES" in reason.build_prompt(ctx, 0.0)
