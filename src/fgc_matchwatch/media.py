"""YouTube access through yt-dlp: stream discovery, metadata, audio download.

Public videos only. If YouTube asks us to prove we are not a bot, wants a
sign-in, or answers 403, that is raised as YouTubeBlocked and the run stops:
no browser cookies, no proxies, no user-agent games. That is a rule, not a
missing feature.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import MatchwatchError, YouTubeBlocked

CHANNEL_STREAMS = "https://www.youtube.com/channel/UCTSVzV2M_ZH-dY1yAz5fiZw/streams"
BLOCK_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "http error 403",
    "confirm your age",
    "login required",
    "this video is private",
    "members-only",
)
# 63 kbps opus is plenty for speech and a third of the size of 251.
AUDIO_FORMAT = "250/249/251/bestaudio[acodec=opus]/bestaudio"
LIVE_AUDIO_FORMAT = "233/234/bestaudio/best"
TITLE_RE = re.compile(r"Day\s*(\d+)\s*,?\s*Field\s*(\d+)", re.IGNORECASE)
DAY_RE = re.compile(r"(\d{4}) FIRST Global Challenge\s*-\s*Day\s*(\d+)\s*$", re.IGNORECASE)


@dataclass
class Stream:
    video: str
    title: str
    day: int | None
    field: int | None  # None = the main program feed
    start: float | None  # epoch seconds the stream went live (release_timestamp)
    duration: float | None
    live_status: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def ytdlp_bin() -> str:
    here = Path(sys.executable).parent / "yt-dlp"
    if here.exists():
        return str(here)
    found = shutil.which("yt-dlp")
    if not found:
        raise MatchwatchError("yt-dlp not found")
    return found


def _check_block(stderr: str) -> None:
    low = stderr.lower()
    for m in BLOCK_MARKERS:
        if m in low:
            raise YouTubeBlocked(f"YouTube refused access ({m!r}). Real stop, not retried: "
                                 + stderr.strip().splitlines()[-1][:300])


def _run(args: list[str], timeout: float = 600) -> str:
    # node is the JS runtime present on archserver; without one yt-dlp warns
    # that YouTube extraction is deprecated and formats may be missing.
    cmd = [ytdlp_bin(), "--js-runtimes", "node", "--no-warnings", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if p.returncode != 0:
        _check_block(p.stderr)
        raise MatchwatchError(f"yt-dlp failed: {p.stderr.strip()[-400:]}")
    _check_block(p.stderr if "ERROR" in p.stderr else "")
    return p.stdout


def info(video: str) -> Stream:
    out = _run(["--skip-download", "-J", "--", video], timeout=120)
    d = json.loads(out)
    return stream_from_info(d)


def stream_from_info(d: dict[str, Any]) -> Stream:
    title = d.get("title") or ""
    day = field = None
    m = TITLE_RE.search(title)
    if m:
        day, field = int(m.group(1)), int(m.group(2))
    else:
        m2 = DAY_RE.search(title)
        if m2:
            day = int(m2.group(2))
    start = d.get("release_timestamp") or d.get("timestamp")
    return Stream(
        video=d["id"], title=title, day=day, field=field,
        start=float(start) if start else None,
        duration=float(d["duration"]) if d.get("duration") else None,
        live_status=d.get("live_status") or "",
    )


def embeds_on_page(url: str) -> list[str]:
    """YouTube ids embedded on the official live page. FIRST Global lists the
    per-field streams there as unlisted videos (2025 and the 2026 promo)."""
    req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", "replace")
    ids = re.findall(r"(?:youtube(?:-nocookie)?\.com/(?:embed|live)/|youtu\.be/|[?&]v=)"
                     r"([A-Za-z0-9_-]{11})", html)
    return list(dict.fromkeys(ids))


def channel_streams(limit: int = 10) -> list[str]:
    out = _run(["--flat-playlist", "--playlist-end", str(limit), "--print", "%(id)s",
                CHANNEL_STREAMS], timeout=180)
    return [x.strip() for x in out.splitlines() if x.strip()]


def _in_year(s: Stream, year: str) -> bool:
    if year in s.title:
        return True
    if s.start:
        return time.gmtime(s.start).tm_year == int(year)
    return s.live_status in ("is_live", "is_upcoming")


def discover(live_page: str, year: str, known: dict[str, Any],
             log: Callable[[str], None], seen: dict[str, Any] | None = None
             ) -> dict[str, Any]:
    """Merge newly found streams of `year` into `known` (video id -> stream dict).

    `seen` caches every id already looked at, matched or not, so a run every five
    minutes does not re-query the same dozen videos through YouTube each time;
    only live/upcoming ones and anything older than 6 h are looked at again.
    """
    seen = {} if seen is None else seen
    ids: list[str] = []
    try:
        ids += embeds_on_page(live_page)
    except OSError as exc:
        log(f"live page unreachable: {exc}")
    try:
        ids += channel_streams()
    except MatchwatchError as exc:
        log(f"channel listing failed: {exc}")
    now = time.time()
    for vid in dict.fromkeys(ids):
        old = known.get(vid)
        if old and old.get("manual"):
            continue
        if old and old.get("live_status") not in ("is_live", "is_upcoming"):
            continue
        prev = seen.get(vid)
        if (not old and prev and now - prev.get("checked", 0) < 6 * 3600
                and prev.get("live_status") not in ("is_live", "is_upcoming")):
            continue
        try:
            s = info(vid)
        except YouTubeBlocked:
            raise
        except MatchwatchError as exc:
            log(f"skip {vid}: {exc}")
            continue
        seen[vid] = {"title": s.title, "live_status": s.live_status, "checked": now}
        if not _in_year(s, year) or (s.day is None and s.field is None):
            if not prev:
                log(f"ignored stream {vid}: {s.title!r} [{s.live_status}] (not a "
                    f"{year} 'Day N, Field M' stream; `streams add` it if it is one)")
            continue
        known[vid] = s.to_json()
        log(f"stream {vid}: {s.title} [{s.live_status}]")
    return known


def download_audio(video: str, dest_dir: Path, log: Callable[[str], None]) -> Path:
    """Whole audio track of a finished stream (~60 MB/hour)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in dest_dir.glob(f"{video}.*"):
        if f.suffix in (".webm", ".m4a", ".opus") and not f.name.endswith(".part"):
            return f
    t0 = time.monotonic()
    _run(["-q", "--no-progress", "-f", AUDIO_FORMAT, "-o", str(dest_dir / "%(id)s.%(ext)s"),
          "--", video], timeout=3600)
    for f in dest_dir.glob(f"{video}.*"):
        if not f.name.endswith(".part"):
            log(f"downloaded {f.name} ({f.stat().st_size >> 20} MiB) in "
                f"{time.monotonic() - t0:.0f}s")
            return f
    raise MatchwatchError(f"download of {video} produced no file")


def record_live_file(video: str, dest: Path, log: Callable[[str], None]) -> int:
    """Record a live stream's audio from its start straight to a file.

    yt-dlp assembles the DASH fragments itself and writes `dest` + ".part" as it
    goes, which ffmpeg can read while it grows. This replaced piping into
    ffmpeg: with --live-from-start yt-dlp picks DASH (format 140) whose
    fragmented MP4 ffmpeg cannot open from a pipe, so on 2026 day 1 four of
    five field recorders stalled or died. Catch-up measured at ~25x realtime.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        [ytdlp_bin(), "--js-runtimes", "node", "--no-warnings", "-q", "--live-from-start",
         "-f", "140/bestaudio[ext=m4a]/bestaudio", "-o", str(dest), "--", video],
        capture_output=True, text=True, check=False)
    _check_block(p.stderr or "")
    log(f"recorder for {video} ended (yt-dlp {p.returncode}): {(p.stderr or '').strip()[-200:]}")
    return p.returncode


def record_live(video: str, dest: Path, log: Callable[[str], None]) -> int:
    """Record a live stream's audio from its start into `dest` until it ends.

    --live-from-start pulls the DVR from the beginning, so a recorder started
    late (or restarted after a reboot) still gets the early matches. Output is
    an .ogg opus file growing on disk; asr.transcribe_file(complete=False)
    eats it as it grows.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    ytdlp = subprocess.Popen(
        [ytdlp_bin(), "--js-runtimes", "node", "--no-warnings", "-q", "--live-from-start",
         "-f", LIVE_AUDIO_FORMAT, "-o", "-", "--", video],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    ff = subprocess.Popen(
        # Live HLS audio from yt-dlp carries corrupt AAC packets around stream
        # starts and format switches (seen on four of five 2026 day-1 fields);
        # without these flags ffmpeg stops writing while staying alive.
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-err_detect", "ignore_err",
         "-fflags", "+discardcorrupt+genpts", "-i", "pipe:0", "-vn", "-ac", "1",
         "-c:a", "libopus", "-b:a", "48k", "-f", "ogg", str(dest)],
        stdin=ytdlp.stdout,
    )
    if ytdlp.stdout:
        ytdlp.stdout.close()
    rc = ff.wait()
    _, err = ytdlp.communicate()
    errs = err.decode("utf-8", "replace") if err else ""
    _check_block(errs)
    log(f"recorder for {video} ended (ffmpeg {rc}, yt-dlp {ytdlp.returncode})")
    return ytdlp.returncode or rc
