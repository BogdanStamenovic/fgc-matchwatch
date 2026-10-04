"""Find where each match sits inside a field stream's transcript.

Pure logic, no I/O. Inputs are a word-timestamped transcript of one stream and
the official schedule of matches played on that stream's field that day.

How it works:
1. Anchors. Every "3, 2, 1, go" countdown is a candidate match start; so is
   every "match N" call-out (the start is assumed a little after it).
2. Scoring. For a (match, anchor) pair: how many of the match's countries the
   commentators name in [anchor - 75 s, anchor + 165 s], minus the ones named
   that are NOT in the match, plus a bonus if "match N" is said just before,
   plus a prior that prefers anchors near the scheduled time + drift.
3. Assignment. Matches on one field happen in schedule order, so matches are
   assigned to anchors by dynamic programming with strictly increasing anchor
   times. That stops two matches with overlapping teams sharing one countdown.
4. Feedback. Drift (how late the event runs vs. the schedule, and the offset of
   the stream clock) is fitted from the confident assignments and the DP is run
   again with the tighter prior.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

MATCH_LEN = 150.0
# Play-onset anchors (see align()). Tuned head-to-head in bench/onset_sweep.py.
ONSET = True
ONSET_BACK = 300.0
ONSET_FWD = 400.0
ONSET_PENALTY = 0.75
PRE = 200.0
POST = 165.0

UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90}

# Commentators' names for countries whose official name is not what is said.
ALIASES: dict[str, list[str]] = {
    "CHN": ["china"], "USA": ["usa", "united states", "america"], "GBR": ["great britain",
    "united kingdom", "uk", "britain"], "KOR": ["korea", "south korea"], "PRK": ["north korea"],
    "RUS": ["russia"], "IRI": ["iran"], "TPE": ["chinese taipei", "taipei"], "HKG": ["hong kong"],
    "CIV": ["ivory coast", "cote d'ivoire"], "COD": ["dr congo", "drc", "democratic republic"],
    "CGO": ["congo"], "TTO": ["trinidad", "trinidad and tobago"], "ANT": ["antigua"],
    "SKN": ["st kitts", "saint kitts"], "LCA": ["st lucia", "saint lucia"],
    "VIN": ["st vincent", "saint vincent"], "BIH": ["bosnia"], "MKD": ["north macedonia",
    "macedonia"], "CZE": ["czech", "czechia"], "UAE": ["emirates", "uae"],
    "PNG": ["papua new guinea", "papua"], "SRB": ["serbia"], "KAZ": ["kazakhstan"],
    "MDA": ["moldova"], "LAO": ["laos"], "VIE": ["vietnam"], "SYR": ["syria"],
    "TAN": ["tanzania"], "GAM": ["gambia"], "BAH": ["bahamas"], "FSM": ["micronesia"],
    "STP": ["sao tome"], "TLS": ["timor", "east timor"], "PLE": ["palestine"],
    "BOL": ["bolivia"], "VEN": ["venezuela"], "KGZ": ["kyrgyzstan", "kyrgyz"],
    "NED": ["netherlands", "holland"], "GEQ": ["equatorial guinea"],
}
# Words commentators use while a match is being played (both seasons). Their
# density after a candidate start minus before it says "play began here";
# an end-of-match countdown has play before it and applause after.
PLAY_WORDS = {
    "barrier", "barriers", "biodiversity", "accelerator", "ecosystem", "ecosystems",
    "dispenser", "rope", "ropes", "climb", "climbing", "seconds", "scoring", "score", "shoot",
    "shooting", "launch", "launching", "intake", "suppression", "brace", "extinguisher",
    "wildfire", "balls", "ball", "shield", "chute", "zone", "units", "endgame", "pushing",
}
GAME_WORDS = {"rock", "paper", "scissors"}
STOP = {"republic", "of", "the", "and", "islands", "people's", "peoples", "people’s",
        "democratic", "state", "kingdom", "federal", "plurinational", "united"}


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def parse_number(tokens: Sequence[str], i: int) -> tuple[int | None, int]:
    """Number starting at tokens[i], digits or English words up to 999.

    "one hundred twenty three" -> 123, "forty eight" -> 48, "37" -> 37.
    Returns (value, tokens consumed), or (None, 0).
    """
    t = tokens[i] if i < len(tokens) else ""
    if t.isdigit():
        return int(t), 1
    cur, n, last = 0, 0, ""
    while i + n < len(tokens):
        w = tokens[i + n]
        if w in UNITS:
            if last in ("", "hundred", "and") or (last == "tens" and cur % 10 == 0
                                                   and UNITS[w] < 10):
                cur += UNITS[w]
                last = "unit"
            else:
                break
        elif w in TENS:
            if last in ("", "hundred", "and"):
                cur += TENS[w]
                last = "tens"
            else:
                break
        elif w == "hundred" and last == "unit" and 0 < cur < 10:
            cur *= 100
            last = "hundred"
        elif w == "and" and last == "hundred":
            last = "and"
        else:
            break
        n += 1
    if last in ("", "and"):
        return (cur, n - 1) if last == "and" else (None, 0)
    return cur, n


@dataclass
class Tok:
    t: float
    w: str
    stop: bool = False  # punctuation followed this word ("match, 22 points")


def tokens(words: Sequence[Sequence[Any]]) -> list[Tok]:
    out: list[Tok] = []
    for s, _e, w in words:
        # "one-hundred-twenty" and "3,2,1" arrive as one word sometimes
        raw = str(w)
        for part in re.split(r"[-,\s]+", raw):
            n = norm(part)
            if n:
                out.append(Tok(float(s), n))
        if out and raw.rstrip()[-1:] in ",.;:?!":
            out[-1].stop = True
    return out


def countdowns(toks: Sequence[Tok]) -> list[float]:
    """Times of 'three, two, one (go)' countdowns, at the moment they hit zero.

    Whisper often mishears the "one" ("three two wide"), so 3-2-<anything
    within 2.5 s> counts. These are both match starts and the end-game count
    the commentators do in the last seconds; align() treats each as both.
    """
    out: list[float] = []
    vals = {"3": 3, "three": 3, "2": 2, "two": 2, "1": 1, "one": 1}
    for i in range(len(toks) - 2):
        a, b, c = toks[i], toks[i + 1], toks[i + 2]
        if vals.get(a.w) != 3 or vals.get(b.w) != 2 or b.t - a.t > 3:
            continue
        if vals.get(c.w) == 1 and c.t - b.t < 3:
            zero = c.t + 1.0
            if i + 3 < len(toks) and toks[i + 3].w.startswith("go") and toks[i + 3].t - c.t < 3:
                zero = toks[i + 3].t
        elif c.t - b.t < 2.5:
            zero = b.t + 2.0
        else:
            zero = b.t + 2.0
        # Announcers' warm-up game: "three, two, one, rock paper scissors shoot".
        if any(t.w in GAME_WORDS for t in toks[i + 2:i + 6]):
            continue
        if not out or zero - out[-1] > 30:
            out.append(zero)
    # "... two, one, go" without a clean three in front of it.
    for i in range(len(toks) - 2):
        a, b, c = toks[i], toks[i + 1], toks[i + 2]
        if vals.get(a.w) == 2 and vals.get(b.w) == 1 and c.w.startswith("go") \
                and c.t - a.t < 4 and all(abs(c.t - z) > 30 for z in out):
            out.append(c.t)
    out.sort()
    return out


def numbers(toks: Sequence[Tok]) -> list[tuple[float, int]]:
    """Every number said, with its time. Used to spot the announced final score."""
    words = [t.w for t in toks]
    out: list[tuple[float, int]] = []
    i = 0
    while i < len(words):
        n, ln = parse_number(words, i)
        if n is not None and ln:
            out.append((toks[i].t, n))
            i += ln
        else:
            i += 1
    return out


def match_mentions(toks: Sequence[Tok]) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    words = [t.w for t in toks]
    for i, w in enumerate(words):
        if w not in ("match", "matches") or toks[i].stop:
            continue
        j = i + 1
        if j < len(words) and words[j] in ("number", "no"):
            j += 1
        n, ln = parse_number(words, j)
        if n is not None and ln and 0 < n < 1000:
            out.append((toks[i].t, n))
    return out


@dataclass
class CountryIndex:
    """Fuzzy country spotter over a token stream."""

    names: dict[str, list[list[str]]] = field(default_factory=dict)

    @classmethod
    def build(cls, code_to_name: dict[str, str]) -> CountryIndex:
        idx = cls()
        for code, name in code_to_name.items():
            forms = {name.lower()}
            forms.update(ALIASES.get(code, []))
            cleaned = [norm(x) for x in re.split(r"[\s-]+", name.lower())]
            core = [x for x in cleaned if x and x not in STOP]
            if core:
                forms.add(" ".join(core))
            idx.names[code] = [[norm(x) for x in f.split() if norm(x)] for f in forms]
        return idx

    def spot(self, toks: Sequence[Tok]) -> list[tuple[float, str]]:
        words = [t.w for t in toks]
        hits: list[tuple[float, str]] = []
        first: dict[str, set[str]] = {}
        for code, forms in self.names.items():
            for f in forms:
                if f:
                    first.setdefault(f[0][:3], set()).add(code)
        for i, w in enumerate(words):
            cands = first.get(w[:3])
            if not cands:
                continue
            for code in cands:
                for f in self.names[code]:
                    if not f:
                        continue
                    seg = " ".join(words[i:i + len(f)])
                    target = " ".join(f)
                    # short names must match exactly: "chad", "iran", "peru", "mali"
                    ok = seg == target if len(target) <= 5 else fuzz.ratio(seg, target) >= 85
                    if ok:
                        hits.append((toks[i].t, code))
                        break
        hits.sort()
        return hits


@dataclass
class MatchSpec:
    key: str
    number: int
    scheduled: float  # epoch seconds
    countries: list[str]
    scores: tuple[int, int] | None = None  # official (red, blue)


@dataclass
class Placement:
    key: str
    start: float | None  # seconds into the stream
    score: float
    hits: list[str]
    named_number: bool
    confident: bool
    expected: float
    anchor_kind: str = ""


def _window_codes(hits: Sequence[tuple[float, str]], lo: float, hi: float) -> set[str]:
    import bisect

    i = bisect.bisect_left(hits, (lo, ""))
    out: set[str] = set()
    while i < len(hits) and hits[i][0] <= hi:
        out.add(hits[i][1])
        i += 1
    return out


def _said_scores(m: MatchSpec, a: float, nums: Sequence[tuple[float, int]]) -> bool:
    """Both official scores spoken 2.5-7 min after the start: the result read-out."""
    if not m.scores or min(m.scores) < 10:
        return False  # small numbers are said all the time
    said = {n for t, n in nums if a + MATCH_LEN <= t <= a + 420}
    return m.scores[0] in said and m.scores[1] in said


def _play(play_t: Sequence[float], a: float) -> float:
    """(play words in the minute after a) - (minute before), clipped to [-3, 3].

    Short windows on purpose: announcers run warm-up games with their own
    "three, two, one" a minute or two before the real start (2025 field 1 did
    rock-paper-scissors), and only the real one is followed by play talk.
    """
    import bisect

    after = bisect.bisect_left(play_t, a + 60) - bisect.bisect_left(play_t, a + 2)
    before = bisect.bisect_left(play_t, a - 2) - bisect.bisect_left(play_t, a - 60)
    return max(-3.0, min(3.0, (after - before) / 3.0))


def endcount_doubt(play_t: Sequence[float], a: float) -> float:
    import bisect

    c = a + MATCH_LEN
    before = bisect.bisect_left(play_t, c) - bisect.bisect_left(play_t, a)
    after = bisect.bisect_left(play_t, c + MATCH_LEN) - bisect.bisect_left(play_t, c)
    return max(0.0, min(3.0, (after - before) / 4.0))


def _score(m: MatchSpec, a: float, kind: str, hits: Sequence[tuple[float, str]],
           mentions: Sequence[tuple[float, int]], expected: float, sigma: float,
           nums: Sequence[tuple[float, int]] = (), play: float = 0.0,
           play_t: Sequence[float] = ()) -> tuple[float, list[str], bool]:
    seen = _window_codes(hits, a - PRE, a + POST)
    mine = set(m.countries)
    got = sorted(seen & mine)
    others = len(seen - mine)
    named = any(n == m.number and a - 180 <= t <= a + 30 for t, n in mentions)
    dt = (a - expected) / sigma
    s = (1.0 * len(got) - 0.6 * min(others, 6) + (3.0 if named else 0.0) - 0.5 * dt * dt
         + 1.0 * play)
    if _said_scores(m, a, nums):
        s += 2.0
        named = True
    if kind == "mention":
        s -= 0.5  # weaker evidence of the actual start than a countdown
        # Play talk tells a start from an end countdown; it says nothing for
        # a call-out, and on the main stage it rewards endgame narration.
        s -= max(0.0, play)
    elif kind == "endcount":
        # This anchor assumes the countdown at a+150 ended a match. If there is
        # more play talk after that countdown than before it, it was a start.
        s -= 0.25 + endcount_doubt(play_t, a)
    elif kind == "onset":
        s -= ONSET_PENALTY
    return s, got, named


def _dp(specs: Sequence[MatchSpec], anchors: Sequence[tuple[float, str]],
        hits: Sequence[tuple[float, str]], mentions: Sequence[tuple[float, int]],
        expect: Any, sigma: float, nums: Sequence[tuple[float, int]] = (),
        play_t: Sequence[float] = ()) -> list[Placement]:
    """Monotone assignment of matches (schedule order) to anchors (time order)."""
    M, A = len(specs), len(anchors)
    NEG = -1e18
    # score[i][j]: match i on anchor j. Leaving a match unplaced costs SKIP.
    SKIP = -1.0
    sc = [[0.0] * A for _ in range(M)]
    meta: list[list[tuple[list[str], bool]]] = [[([], False)] * A for _ in range(M)]
    for i, m in enumerate(specs):
        e = expect(m)
        for j, (a, kind) in enumerate(anchors):
            if abs(a - e) > 6 * sigma:
                sc[i][j] = NEG
                continue
            s, got, named = _score(m, a, kind, hits, mentions, e, sigma, nums,
                                   _play(play_t, a), play_t)
            sc[i][j] = s
            meta[i][j] = (got, named)
    # best[i][j] = best total for matches[:i+1] with match i on anchor j (or skipped
    # with last used anchor j-ish). Use prefix-max over anchors for O(M*A).
    best = [[NEG] * (A + 1) for _ in range(M + 1)]  # column A = "nothing used yet"
    back: list[list[tuple[int, int] | None]] = [[None] * (A + 1) for _ in range(M + 1)]
    best[0][A] = 0.0
    for i in range(M):
        # pm[j]: best over prev-last-anchor < j, including "none used"
        pm_val, pm_arg = best[i][A], A
        for j in range(A):
            # skip match i keeping last anchor j
            if best[i][j] + SKIP > best[i + 1][j]:
                best[i + 1][j] = best[i][j] + SKIP
                back[i + 1][j] = (j, -1)
            if sc[i][j] > NEG / 2 and pm_val > NEG / 2:
                v = pm_val + sc[i][j]
                if v > best[i + 1][j]:
                    best[i + 1][j] = v
                    back[i + 1][j] = (pm_arg, j)
            if best[i][j] > pm_val:
                pm_val, pm_arg = best[i][j], j
        if best[i][A] + SKIP > best[i + 1][A]:
            best[i + 1][A] = best[i][A] + SKIP
            back[i + 1][A] = (A, -1)
    j = max(range(A + 1), key=lambda k: best[M][k])
    chosen: list[int] = [-1] * M
    for i in range(M, 0, -1):
        b = back[i][j]
        if b is None:
            break
        prev, used = b
        chosen[i - 1] = used
        j = prev
    out: list[Placement] = []
    for i, m in enumerate(specs):
        e = expect(m)
        k = chosen[i]
        if k < 0:
            out.append(Placement(m.key, None, SKIP, [], False, False, e))
            continue
        got, named = meta[i][k]
        n = len(m.countries)
        # A spoken number or score alone is not enough: 2025 t2-41 had no audio for
        # its play, only the read-out "scores are official for match number 41".
        confident = (named and len(got) >= 2) or len(got) >= max(3, (n + 1) // 2)
        out.append(Placement(m.key, anchors[k][0], sc[i][k], got, named, confident, e,
                             anchors[k][1]))
    return out


def align(words: Sequence[Sequence[Any]], stream_start: float, specs: Sequence[MatchSpec],
          code_to_name: dict[str, str], use_numbers: bool = True,
          trust_schedule: bool = True) -> list[Placement]:
    """use_numbers=False ignores spoken match numbers and scores; the evaluation
    uses that to keep them as independent ground truth.

    trust_schedule=False drops the time prior from the first pass (order, teams
    and call-outs only). For matches that slipped off their scheduled day: the
    2025 round robin was scheduled for Day 2 and played on Day 3."""
    specs = sorted(specs, key=lambda m: (m.scheduled, m.number))
    if not specs or not words:
        return [Placement(m.key, None, 0.0, [], False, False, 0.0) for m in specs]
    toks = tokens(words)
    hits = CountryIndex.build(code_to_name).spot(toks)
    mentions = match_mentions(toks) if use_numbers else []
    nums = numbers(toks) if use_numbers else []
    play_t = [t.t for i, t in enumerate(toks) if t.w in PLAY_WORDS
              and not (i and toks[i - 1].w in GAME_WORDS)]
    anchors: list[tuple[float, str]] = []
    for t in countdowns(toks):
        anchors.append((t, "countdown"))
        # The same count may be the end of a 2:30 match.
        anchors.append((t - MATCH_LEN, "endcount"))
    # Play onsets: where play talk jumps. Catches starts with no countdown at all
    # (2025 Finals 3 began after a crowd chant: "this final game starts...").
    if play_t:
        t, last_peak = play_t[0] - 60, -1e9
        prev = cur = -9.0
        while t < play_t[-1] + 60:
            nxt = _play(play_t, t + 5)
            if cur >= 2.0 and cur >= prev and cur > nxt and t - last_peak > 120:
                # The peak lags the start; back up to the first play word of the burst.
                first = [x for x in play_t if t - 40 <= x <= t + 5]
                at = (first[0] - 3) if first else t
                # Fallback only: an onset next to a real countdown just competes
                # with it and loses accuracy (measured on 54 hand-labelled starts: onsets near a countdown made 4 of them 30-380 s worse; 300 s back / 400 s ahead was the best window).
                if ONSET and not any(k != "mention" and at - ONSET_BACK <= a <= at + ONSET_FWD
                                     for a, k in anchors):
                    anchors.append((at, "onset"))
                last_peak = t
            prev, cur, t = cur, nxt, t + 5
    anchors.sort()
    merged: list[tuple[float, str]] = []
    for a in anchors:
        if merged and a[0] - merged[-1][0] < 20:
            if merged[-1][1] != "countdown" and a[1] == "countdown":
                merged[-1] = a
            continue
        merged.append(a)
    anchors = merged
    have = [a for a, _ in anchors]
    readout = {"score", "scores", "points", "official", "results", "result"}
    words_t = [(t.t, t.w) for t in toks]
    import bisect as _b

    for t, _n in mentions:
        i = _b.bisect_left(words_t, (t, ""))
        if any(w in readout for _, w in words_t[max(0, i - 6):i + 10]):
            continue  # "scores are official for match 41": after the match, not a start
        s = t + 20.0
        if not any(abs(s - a) < 60 for a in have):
            anchors.append((s, "mention"))
    anchors.sort()

    def flat(m: MatchSpec) -> float:
        return m.scheduled - stream_start

    # Pass 1: wide prior. The event can run 40+ minutes behind schedule.
    p1 = _dp(specs, anchors, hits, mentions, flat, sigma=900.0 if trust_schedule else 1e7,
             nums=nums, play_t=play_t)
    pts = [(m.scheduled, p.start - flat(m)) for m, p in zip(specs, p1)
           if p.confident and p.start is not None]
    if len(pts) < 2:
        return p1

    def drift(m: MatchSpec) -> float:
        # Local median of residuals from the nearest confident matches in time.
        near = sorted(pts, key=lambda x: abs(x[0] - m.scheduled))[:5]
        return flat(m) + statistics.median(r for _, r in near)

    # Pass 2: tight prior around the fitted drift.
    return _dp(specs, anchors, hits, mentions, drift, sigma=180.0, nums=nums, play_t=play_t)
