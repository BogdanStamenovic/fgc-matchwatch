"""Turn one match's commentary into per-team scouting facts with an LLM.

The LLM is a cheap, fallible component; the code around it is the judge.
It must return, per team, claims that each carry a verbatim commentator quote.
Every quote is then checked against the transcript (fuzzy, to forgive
punctuation), and the team must be named within ~30 s of where the quote was
said, so a claim about "they" cannot be pinned on the wrong robot. Claims that
fail either check are dropped and counted. Invalid JSON gets one retry with the
parser error fed back; a second failure skips the match with a logged error.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from .align import CountryIndex, Tok, norm, tokens
from .errors import MatchwatchError

GAME_2026 = """\
GAME (FIRST Global Challenge 2026, "Igniting Innovation"): 2:30 matches, fully driver-controlled, no
autonomous. Two 3-team regional alliances (red, blue) and all six together as a global alliance.
500 orange foam balls (WILDFIRE) pour out from under the central EXTINGUISHER at the start. Robots
INTAKE balls from the floor and score them in their alliance's SUPPRESSION UNIT (tall, opening at
~165 cm under a canopy; usually needs a SHOOTER or a lift), 1 pt each. Robots can push balls into the
FIRE SHIELD port, where the HUMAN PLAYER feeds them back through a chute or throws them into the
EXTINGUISHER (1 pt to all six teams). Endgame: robots CLIMB the alliance's BRACE, a sloped steel pipe
with ZONE 1, 2, 3 (3 is highest); climbing multiplies suppression points. A PARTNER CLIMB is a robot
carrying another robot while climbing (+25). COOPERTITION bonus when 4-6 robots reach ZONE 3.
Typical failure words: tipped, stuck, disconnected, dead, not moving, jammed, fell off the brace."""

GAME_2025 = """\
GAME (FIRST Global Challenge 2025, "Eco-Equilibrium"): 2:30 matches, driver-controlled. Two 3-team
regional alliances (red, blue) plus a global alliance of all six. Robots remove BARRIERS from
ecosystems, spin up the ACCELERATOR to make the dispenser release BIODIVERSITY UNITS (balls), and
score them into the ECOSYSTEMS (often with a launcher/shooter or by a HUMAN PLAYER). Endgame: robots
CLIMB ROPES to levels (up to level 4) for a multiplier. Typical failure words: tipped, stuck,
disconnected, not moving, fell."""

GAMES = {"2026": GAME_2026, "2025": GAME_2025}

# How the 2026 contract fields map onto the 2025 test game.
FIELD_NOTES = {
    "2026": 'climbZone: highest BRACE zone the robot reached as "1", "2" or "3", else null. '
            "shooterWorks: true if the robot is seen scoring into the SUPPRESSION UNIT with a "
            "shooter/launcher, false if its shooter is said to fail or miss repeatedly, else null.",
    "2025": 'climbZone: the rope level the robot reached as "1".."4", else null. '
            "shooterWorks: true if the robot is seen launching biodiversity units into an "
            "ecosystem, false if its launcher is said to fail, else null.",
}

SYSTEM = (
    "You are a scouting analyst for a robotics team. You read live commentary transcripts and "
    "extract facts about specific teams. You never invent: every claim must be supported by a "
    "verbatim quote from the transcript. When the commentators say nothing about a team, return "
    "empty lists and nulls for it. Output JSON only."
)

SCHEMA = """{
  "teams": {
    "<CODE>": {
      "goodAt":   [{"claim": "short", "quote": "verbatim words from the transcript"}],
      "badAt":    [{"claim": "short", "quote": "..."}],
      "problems": [{"claim": "short", "quote": "..."}],
      "strategy": {"claim": "what they tried to do this match", "quote": "..."} or null,
      "climbZone": {"value": "1"|"2"|"3"|"4", "quote": "..."} or null,
      "shooterWorks": {"value": true|false, "quote": "..."} or null
    }
  }
}"""


@dataclass
class TeamFacts:
    code: str
    summary: str
    good_at: list[str] = field(default_factory=list)
    bad_at: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    strategy: str = ""
    climb_zone: str | None = None
    shooter_works: bool | None = None
    evidence: list[str] = field(default_factory=list)
    dropped: int = 0
    kept: int = 0

    def facts(self) -> dict[str, Any]:
        return {
            "goodAt": self.good_at, "badAt": self.bad_at, "strategy": self.strategy,
            "climbZone": self.climb_zone, "shooterWorks": self.shooter_works,
            "problems": self.problems, "evidence": self.evidence,
        }


@dataclass
class MatchContext:
    year: str
    key: str
    name: str
    red: list[tuple[str, str]]  # (code, name)
    blue: list[tuple[str, str]]
    red_score: int | None
    blue_score: int | None
    details: dict[str, Any] | None
    excerpt_words: list[list[Any]]  # [start, end, word], seconds into the stream


def excerpt_text(words: Sequence[Sequence[Any]], t0: float) -> str:
    """Transcript with a [m:ss] stamp every ~10 s, relative to t0."""
    out: list[str] = []
    last = -1e9
    for s, _e, w in words:
        if s - last >= 10:
            rel = max(0.0, s - t0)
            out.append(f"\n[{int(rel // 60)}:{int(rel % 60):02d}]")
            last = s
        out.append(str(w))
    return " ".join(out).strip()


def build_prompt(ctx: MatchContext, t0: float) -> str:
    def side(xs: list[tuple[str, str]]) -> str:
        return ", ".join(f"{c} ({n})" for c, n in xs)

    result = ""
    if ctx.red_score is not None:
        result = f"Official result: red {ctx.red_score}, blue {ctx.blue_score}.\n"
    det = ""
    if ctx.details:
        keep = {k: v for k, v in ctx.details.items()
                if isinstance(v, (int, float)) and k not in ("id",)}
        det = ("Official per-match scoring details (counts/multipliers; 'RobotOne/Two/Three' "
               "are stations 11/12/13 for red and 21/22/23 for blue): " + json.dumps(keep) + "\n")
    codes = [c for c, _ in ctx.red + ctx.blue]
    return f"""{GAMES.get(ctx.year, GAME_2026)}

MATCH: {ctx.name} ({ctx.key})
RED alliance: {side(ctx.red)}
BLUE alliance: {side(ctx.blue)}
{result}{det}
FIELD NOTES: {FIELD_NOTES.get(ctx.year, FIELD_NOTES['2026'])}

TRANSCRIPT of the field commentary around this match (auto-transcribed, may misspell names; the
match starts at about [1:00]):
<<<
{excerpt_text(ctx.excerpt_words, t0)}
>>>

For each of these teams: {", ".join(codes)} — say what it is good at, bad at, its strategy this
match, observed problems, the highest climb level reached and whether its shooter works, using ONLY
what the commentators said. A quote must be copied verbatim (5-25 words) from the transcript and must
be about that team. The official result is context; do not make claims from it alone. Prefer a few
solid claims over many weak ones. Return exactly this JSON shape and nothing else:
{SCHEMA}"""


def call_llm(cmd: Sequence[str], model: str, prompt: str, timeout: float = 300
             ) -> tuple[str, str]:
    """Run the keyless CLI. Returns (text, model id actually used)."""
    args = [*cmd, "--model", model, "--no-session-persistence", "--tools", "",
            "--strict-mcp-config", "--system-prompt", SYSTEM, "--output-format", "json"]
    p = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=timeout,
                       check=False, cwd="/tmp")
    if p.returncode != 0:
        raise MatchwatchError(f"LLM call failed ({p.returncode}): {p.stderr[-300:]}")
    try:
        env = json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise MatchwatchError(f"LLM CLI returned non-JSON envelope: {p.stdout[:200]}") from exc
    used = model
    mu = env.get("modelUsage")
    if isinstance(mu, dict) and mu:
        used = max(mu, key=lambda k: (mu[k] or {}).get("outputTokens", 0))
    return str(env.get("result", "")), used


def parse_json(text: str) -> dict[str, Any]:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j < 0:
            raise ValueError("no JSON object in output")
        t = t[i:j + 1]
    data = json.loads(t)
    if not isinstance(data, dict) or not isinstance(data.get("teams"), dict):
        raise TypeError('top level must be {"teams": {...}}')
    return data


class Verifier:
    """Checks a quote is in the transcript and near a mention of the team."""

    def __init__(self, words: Sequence[Sequence[Any]], names: dict[str, str]):
        self.toks: list[Tok] = tokens(words)
        self.words = [t.w for t in self.toks]
        self.hits = CountryIndex.build(names).spot(self.toks)

    def locate(self, quote: str) -> float | None:
        q = [norm(x) for x in re.split(r"[\s\-,]+", quote) if norm(x)]
        if len(q) < 3:
            return None
        qs = " ".join(q)
        n = len(q)
        best, at = 0.0, None
        for i in range(max(1, len(self.words) - n + 1)):
            if self.words[i][:2] != q[0][:2] and fuzz.ratio(self.words[i], q[0]) < 70:
                continue
            seg = " ".join(self.words[i:i + n])
            r = fuzz.ratio(seg, qs)
            if r > best:
                best, at = r, self.toks[i].t
        return at if best >= 88 else None

    def near_team(self, t: float, code: str, before: float = 30, after: float = 12) -> bool:
        return any(c == code and t - before <= ht <= t + after for ht, c in self.hits)


def _claims(v: Any) -> list[dict[str, Any]]:
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    if isinstance(v, dict):
        return [v]
    return []


def _join(xs: list[str]) -> str:
    return "; ".join(x.rstrip(". ") for x in xs)


def compose_summary(tf: TeamFacts) -> str:
    """2-4 plain sentences from verified facts only; '' when there are none."""
    parts: list[str] = []
    if tf.strategy:
        parts.append(f"Strategy: {tf.strategy.rstrip('. ')}.")
    if tf.good_at:
        parts.append(f"Good at: {_join(tf.good_at)}.")
    weak = tf.bad_at + [p for p in tf.problems if p not in tf.bad_at]
    if weak:
        parts.append(f"Problems: {_join(weak)}.")
    extra = []
    if tf.climb_zone:
        extra.append(f"climbed to zone/level {tf.climb_zone}")
    if tf.shooter_works is not None:
        extra.append("shooter works" if tf.shooter_works else "shooter not working")
    if extra:
        parts.append(f"Observed: {', '.join(extra)}.")
    return " ".join(parts[:4])


def _check(c: dict[str, Any], code: str, tf: TeamFacts, ver: Verifier) -> bool:
    q = str(c.get("quote") or "")
    at = ver.locate(q)
    good = at is not None and ver.near_team(at, code)
    if good:
        tf.kept += 1
        if q not in tf.evidence:
            tf.evidence.append(q)
    else:
        tf.dropped += 1
    return good


def verify(raw: dict[str, Any], codes: Sequence[str], ver: Verifier) -> dict[str, TeamFacts]:
    out: dict[str, TeamFacts] = {}
    teams = raw.get("teams", {})
    for code in codes:
        t = teams.get(code) or {}
        tf = TeamFacts(code=code, summary="")

        def ok(c: dict[str, Any], code: str = code, tf: TeamFacts = tf) -> bool:
            return _check(c, code, tf, ver)

        for src, dst in (("goodAt", tf.good_at), ("badAt", tf.bad_at),
                         ("problems", tf.problems)):
            for c in _claims(t.get(src)):
                if c.get("claim") and ok(c):
                    dst.append(str(c["claim"]))
        for c in _claims(t.get("strategy")):
            if c.get("claim") and ok(c):
                tf.strategy = str(c["claim"])
        for c in _claims(t.get("climbZone")):
            val = str(c.get("value") or "")
            if val in ("1", "2", "3", "4") and ok(c):
                tf.climb_zone = val
        for c in _claims(t.get("shooterWorks")):
            if isinstance(c.get("value"), bool) and ok(c):
                tf.shooter_works = bool(c["value"])
        # The summary is built from the verified claims only. The LLM's own prose
        # summary is not used: in the 2025 test it repeated claims whose quotes
        # had been dropped (e.g. a climb that no verified quote supported).
        tf.summary = compose_summary(tf)
        out[code] = tf
    return out


def analyse(ctx: MatchContext, t0: float, names: dict[str, str], llm_cmd: Sequence[str],
            model: str, log: Callable[[str], None]) -> tuple[dict[str, TeamFacts], str, dict[str, int]]:
    """Returns (facts per team, model used, stats). Raises MatchwatchError when it gives up."""
    prompt = build_prompt(ctx, t0)
    codes = [c for c, _ in ctx.red + ctx.blue]
    stats = {"calls": 0, "invalid": 0}
    last_err = ""
    used = model
    for attempt in range(2):
        p = prompt if attempt == 0 else (
            prompt + f"\n\nYour previous answer was not valid: {last_err}. Return ONLY the JSON.")
        stats["calls"] += 1
        text, used = call_llm(llm_cmd, model, p)
        try:
            raw = parse_json(text)
        except (ValueError, TypeError) as exc:
            stats["invalid"] += 1
            last_err = str(exc)[:200]
            log(f"  {ctx.key}: LLM output invalid ({last_err}); "
                + ("retrying" if attempt == 0 else "giving up"))
            continue
        ver = Verifier(ctx.excerpt_words, names)
        return verify(raw, codes, ver), used, stats
    raise MatchwatchError(f"{ctx.key}: LLM output invalid twice: {last_err}")
