"""Orchestration: interest list -> stream -> transcript -> placement -> LLM -> POST.

State lives in MATCHWATCH_HOME/state.json and is written after every match,
so a killed run resumes where it stopped and a finished match is never redone
(the POST is idempotent by id anyway).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import align, asr, media, reason
from .config import Config
from .errors import MatchwatchError, YouTubeBlocked

PIPELINE_VERSION = "1"
FGA = "https://api.first.global/v1"
PRE_ROLL = 60.0  # intro / alliance roll call before the countdown
POST_ROLL = 75.0  # result read-out after the 2:30
MAX_ATTEMPTS = 6  # a match not found yet is retried on later runs, then given up

Log = Callable[[str], None]


# ---------- small I/O helpers ----------

def _get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 30) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "fgc-matchwatch (team SRB)",
                                               **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _cached(cfg: Config, name: str, url: str, max_age: float) -> Any:
    f = cfg.cache_dir / name
    if f.is_file() and time.time() - f.stat().st_mtime < max_age:
        return json.loads(f.read_text())
    data = _get_json(url, timeout=60)
    f.write_text(json.dumps(data))
    return data


def load_state(cfg: Config) -> dict[str, Any]:
    if cfg.state_file.is_file():
        data: dict[str, Any] = json.loads(cfg.state_file.read_text())
        return data
    return {"done": {}, "pending": {}, "stats": {}}


def save_state(cfg: Config, st: dict[str, Any]) -> None:
    tmp = cfg.state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(cfg.state_file)


def load_streams(cfg: Config) -> dict[str, Any]:
    if cfg.streams_file.is_file():
        data: dict[str, Any] = json.loads(cfg.streams_file.read_text())
        return data
    return {}


def save_streams(cfg: Config, s: dict[str, Any]) -> None:
    cfg.streams_file.write_text(json.dumps(s, indent=1))


def obs_id(match_key: str, code: str) -> str:
    return hashlib.sha256(f"{match_key}|{code}|{PIPELINE_VERSION}".encode()).hexdigest()[:32]


# ---------- schedule ----------

@dataclass
class Schedule:
    matches: list[dict[str, Any]]
    names: dict[str, str]

    def by_key(self, key: str) -> dict[str, Any] | None:
        for m in self.matches:
            if f"{m['tournamentKey']}-{m['id']}" == key:
                return m
        return None


def schedule(cfg: Config, max_age: float = 120) -> Schedule:
    d = _cached(cfg, f"fga-{cfg.year}.json",
                f"{FGA}?excludeMatchDetails=true&year={cfg.year}", max_age)
    names = {r["team"]["country"]: r["team"]["name"] for r in d.get("rankings") or []
             if r.get("team")}
    for m in d.get("matches") or []:
        for p in m.get("participants") or []:
            names.setdefault(p["country"], p["country"])
    return Schedule(d.get("matches") or [], names)


def match_details(cfg: Config, m: dict[str, Any]) -> dict[str, Any]:
    url = f"{FGA}/matches?year={cfg.year}&tournamentKey={m['tournamentKey']}&id={m['id']}"
    try:
        d = _get_json(url)
        one = d[0] if isinstance(d, list) and d else d
        return one if isinstance(one, dict) else {}
    except (OSError, ValueError):
        return {}


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def match_time(m: dict[str, Any]) -> float:
    """Actual start if the API has it (the per-match endpoint fills startTime live
    in some years; it was empty for every 2025 match), else scheduled."""
    for k in ("startTime", "scheduledTime"):
        v = m.get(k)
        if v:
            try:
                return epoch(v)
            except ValueError:
                continue
    raise MatchwatchError(f"match {m.get('name')} has no time")


# ---------- interest ----------

def interest(cfg: Config, sched: Schedule, log: Log) -> list[dict[str, Any]]:
    """Matches to process, highest priority first. From the scout server, or with
    MATCHWATCH_MATCHES=t2-1,t2-7 an explicit list (used for the 2025 test bed)."""
    import os

    explicit = os.environ.get("MATCHWATCH_MATCHES", "").strip()
    if explicit:
        out = []
        for i, k in enumerate(x.strip() for x in explicit.split(",") if x.strip()):
            m = sched.by_key(k)
            if m:
                out.append({"key": k, "priority": 1000 - i, "reasons": ["explicit"]})
        return out
    if not cfg.scout_key:
        raise MatchwatchError("SCOUT_KEY is not set")
    d = _get_json(f"{cfg.scout_url}/api/interest", headers={"X-Scout-Key": cfg.scout_key})
    ms = d.get("matches") or []
    return sorted(ms, key=lambda x: -int(x.get("priority") or 0))


def post_observation(cfg: Config, obs: dict[str, Any]) -> None:
    body = json.dumps(obs).encode()
    req = urllib.request.Request(
        f"{cfg.scout_url}/api/observation", data=body, method="POST",
        headers={"Content-Type": "application/json", "X-Scout-Key": cfg.scout_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status != 200:
                raise MatchwatchError(f"POST /api/observation -> {r.status}")
    except urllib.error.HTTPError as exc:
        raise MatchwatchError(f"POST /api/observation -> {exc.code}: {exc.read()[:200]!r}") \
            from exc


# ---------- streams ----------

def stream_for(cfg: Config, streams: dict[str, Any], m: dict[str, Any]) -> dict[str, Any] | None:
    """The field stream covering this match: same field, went live before the match,
    still running or long enough to contain it. Falls back to the main feed."""
    t = epoch(m["scheduledTime"])
    fld = m.get("field") or m.get("fieldNumber")

    def covers(s: dict[str, Any]) -> bool:
        st = s.get("start")
        if not st or st > t + 3600:
            return False
        if s.get("live_status") == "is_live":
            return True
        dur = s.get("duration") or 0
        # VODs drop paused stretches, so a stream can be shorter than the day it
        # covered. Allow a generous tail rather than missing the stream.
        return t <= st + dur * 1.6 + 3600

    cands = [s for s in streams.values() if s.get("field") == fld and covers(s)]
    if not cands:
        cands = [s for s in streams.values() if s.get("field") is None and covers(s)]
    if not cands:
        return None
    return max(cands, key=lambda s: s["start"])


def _unit(vid: str) -> str:
    return "fgc-matchwatch-rec-" + "".join(c if c.isalnum() else "_" for c in vid)


def recorder_active(vid: str) -> bool:
    import subprocess

    try:
        r = subprocess.run(["systemctl", "--user", "is-active", _unit(vid)],
                           capture_output=True, text=True, timeout=10, check=False)
        return r.stdout.strip() in ("active", "activating")
    except OSError:
        return False


def ensure_recorders(cfg: Config, streams: dict[str, Any], log: Log) -> None:
    """One recorder per live field stream, as a transient systemd user unit.

    Transient units because the timer's oneshot service kills its own children
    when it exits; a recorder must outlive the five-minute run that started it.
    """
    import shutil
    import subprocess
    import sys

    if not shutil.which("systemd-run"):
        log("systemd-run not found; start `fgc-matchwatch record <id>` by hand for live streams")
        return
    exe = str(Path(sys.executable).parent / "fgc-matchwatch")
    for vid, s in streams.items():
        if s.get("live_status") != "is_live" or recorder_active(vid):
            continue
        r = subprocess.run(
            ["systemd-run", "--user", "--collect", f"--unit={_unit(vid)}",
             f"--setenv=MATCHWATCH_HOME={cfg.home}", exe, "record", "--", vid],
            capture_output=True, text=True, timeout=30, check=False)
        log(f"recorder for {vid} ({s.get('title')}): "
            + ("started" if r.returncode == 0 else f"failed: {r.stderr.strip()[-200:]}"))


def ensure_transcript(cfg: Config, s: dict[str, Any], engine_box: list[asr.Engine],
                      log: Log) -> dict[str, Any] | None:
    vid = s["video"]
    out = cfg.transcripts_dir / f"{vid}.json"
    tr = asr.load_transcript(out)
    if tr and tr.get("complete"):
        return tr
    rec = cfg.audio_dir / f"{vid}.live.ogg"
    recording = s.get("live_status") == "is_live" or (rec.is_file() and recorder_active(vid))
    if recording:
        if not rec.is_file():
            log(f"{vid} is live but has no recording yet (recorder starting?)")
            return tr
        audio = rec
    elif rec.is_file():
        # The stream ended and the recorder finished: its file is the whole stream,
        # on the same timeline as the VOD, so no need to download it again.
        audio = rec
    else:
        audio = media.download_audio(vid, cfg.audio_dir, log)
    if not engine_box:
        engine_box.append(asr.pick_engine(cfg, log))
    tr = asr.transcribe_file(engine_box[0], audio, out, vid, log, complete=not recording)
    if tr.get("complete"):
        # Keep transcripts, not audio: the disk is the scarce thing.
        for f in cfg.audio_dir.glob(f"{vid}.*"):
            f.unlink(missing_ok=True)
        log(f"{vid}: transcript complete ({len(tr['words'])} words); audio deleted")
    return tr


# ---------- one match ----------

def spec_of(m: dict[str, Any]) -> align.MatchSpec:
    return align.MatchSpec(
        key=f"{m['tournamentKey']}-{m['id']}",
        number=int(m["id"]),
        scheduled=match_time(m),
        countries=[p["country"] for p in m.get("participants") or []],
        scores=(int(m["redScore"]), int(m["blueScore"]))
        if m.get("played") or m.get("redScore") else None,
    )


def place(sched: Schedule, s: dict[str, Any], tr: dict[str, Any], m: dict[str, Any]
          ) -> align.Placement:
    """Align every match of this field during this stream, return the one we want."""
    fld = m.get("field")
    st = s["start"]
    end = st + max(s.get("duration") or 0, (tr.get("duration") or 0)) * 1.6 + 3600
    same = [x for x in sched.matches
            if x.get("field") == fld and st - 1800 <= match_time(x) <= end
            and x.get("participants")]
    specs = [spec_of(x) for x in same]
    key = f"{m['tournamentKey']}-{m['id']}"
    for p in align.align(tr["words"], st, specs, sched.names):
        if p.key == key:
            return p
    return align.Placement(key, None, 0.0, [], False, False, 0.0)


def build_observations(cfg: Config, sched: Schedule, m: dict[str, Any], s: dict[str, Any],
                       tr: dict[str, Any], p: align.Placement, log: Log,
                       stats: dict[str, int]) -> list[dict[str, Any]]:
    assert p.start is not None
    a, b = p.start - PRE_ROLL, p.start + align.MATCH_LEN + POST_ROLL
    words = [w for w in tr["words"] if a <= w[0] <= b]
    full = match_details(cfg, m)
    names = sched.names
    # The per-match endpoint's participants carry teamKey but no country, so
    # the schedule's list is the source of truth for who played.
    parts = sorted(m.get("participants") or [], key=lambda x: int(x["station"]))
    red = [(x["country"], f"{names.get(x['country'], x['country'])}, station {x['station']}")
           for x in parts if 11 <= int(x["station"]) <= 19]
    blue = [(x["country"], f"{names.get(x['country'], x['country'])}, station {x['station']}")
            for x in parts if 21 <= int(x["station"]) <= 29]
    ctx = reason.MatchContext(
        year=cfg.year, key=p.key, name=m.get("name", p.key), red=red, blue=blue,
        red_score=m.get("redScore"), blue_score=m.get("blueScore"),
        details=full.get("details"), excerpt_words=words,
    )
    facts, used, st = reason.analyse(ctx, a, names, cfg.llm_cmd, cfg.llm_model, log)
    for k, v in st.items():
        stats[k] = stats.get(k, 0) + v
    transcript = reason.excerpt_text(words, a)[:4000]
    url = f"https://www.youtube.com/watch?v={s['video']}&t={int(max(0, a))}"
    out = []
    now = int(time.time() * 1000)
    for code, tf in facts.items():
        stats["claims_kept"] = stats.get("claims_kept", 0) + tf.kept
        stats["claims_dropped"] = stats.get("claims_dropped", 0) + tf.dropped
        out.append({
            "id": obs_id(p.key, code), "matchKey": p.key, "code": code,
            "source": {"url": url, "start": int(max(0, a)), "end": int(b)},
            "summary": tf.summary, "facts": tf.facts(), "transcript": transcript,
            "model": used, "ts": now,
            "alignment": {"start": round(p.start, 1), "hits": p.hits,
                          "namedNumberOrScore": p.named_number, "anchor": p.anchor_kind},
        })
    return out


# ---------- runs ----------

def refresh_streams(cfg: Config, log: Log) -> dict[str, Any]:
    streams = load_streams(cfg)
    streams = media.discover(cfg.live_page, cfg.year, streams, log)
    save_streams(cfg, streams)
    return streams


def run_once(cfg: Config, log: Log, dry_run: bool = False, limit: int | None = None,
             discover: bool = True) -> int:
    """Process every new played match of interest once. Returns failures count."""
    cfg.ensure_dirs()
    st = load_state(cfg)
    sched = schedule(cfg)
    want = interest(cfg, sched, log)
    todo = [w for w in want if w["key"] not in st["done"]]
    log(f"{len(want)} matches of interest, {len(todo)} not done yet")
    if dry_run:
        streams = load_streams(cfg)
        for w in todo[:limit]:
            m = sched.by_key(w["key"])
            s = stream_for(cfg, streams, m) if m else None
            log(f"  would process {w['key']} (priority {w.get('priority')}): "
                + (f"stream {s['video']} {s['title']!r}" if s else "no stream known yet"))
        return 0
    streams = refresh_streams(cfg, log) if discover else load_streams(cfg)
    ensure_recorders(cfg, streams, log)
    engine_box: list[asr.Engine] = []
    failures = 0
    stats = st.setdefault("stats", {})
    for w in todo[:limit]:
        key = w["key"]
        m = sched.by_key(key)
        pend = st["pending"].setdefault(key, {"attempts": 0})
        if m is None or not m.get("played", True):
            continue
        try:
            s = stream_for(cfg, streams, m)
            if not s:
                raise MatchwatchError("no stream covers this match yet")
            tr = ensure_transcript(cfg, s, engine_box, log)
            if not tr or not tr.get("words"):
                raise MatchwatchError("no transcript yet")
            p = place(sched, s, tr, m)
            if p.start is None or not p.confident:
                raise MatchwatchError(f"not placed confidently (hits {p.hits}, "
                                      f"start {p.start})")
            log(f"{key}: at {p.start:.0f}s in {s['video']} (hits {p.hits}"
                f"{', number/score said' if p.named_number else ''})")
            obs = build_observations(cfg, sched, m, s, tr, p, log, stats)
            for o in obs:
                post_observation(cfg, o)
            st["done"][key] = {"ts": int(time.time()), "video": s["video"],
                               "start": p.start, "observations": [o["id"] for o in obs],
                               "withFacts": sum(1 for o in obs if o["facts"]["evidence"])}
            st["pending"].pop(key, None)
            log(f"{key}: posted {len(obs)} observations "
                f"({st['done'][key]['withFacts']} with facts)")
        except YouTubeBlocked:
            save_state(cfg, st)
            raise
        except (MatchwatchError, OSError, ValueError, KeyError) as exc:
            failures += 1
            pend["attempts"] += 1
            pend["error"] = str(exc)[:300]
            log(f"{key}: {exc} (attempt {pend['attempts']})")
            if pend["attempts"] >= MAX_ATTEMPTS:
                st["done"][key] = {"ts": int(time.time()), "gaveUp": pend["error"]}
                st["pending"].pop(key, None)
        save_state(cfg, st)
    return failures


def status(cfg: Config) -> dict[str, Any]:
    st = load_state(cfg)
    streams = load_streams(cfg)
    done = st.get("done", {})
    return {
        "home": str(cfg.home),
        "year": cfg.year,
        "scout": cfg.scout_url,
        "done": sum(1 for v in done.values() if "gaveUp" not in v),
        "gaveUp": {k: v["gaveUp"] for k, v in done.items() if "gaveUp" in v},
        "pending": st.get("pending", {}),
        "streams": {k: f"{v.get('title')} [{v.get('live_status')}]" for k, v in streams.items()},
        "stats": st.get("stats", {}),
        "freeVramMiB": asr.free_vram_mib(),
        "transcripts": sorted(p.name for p in Path(cfg.transcripts_dir).glob("*.json"))
        if cfg.transcripts_dir.is_dir() else [],
    }
