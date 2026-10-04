"""Command-line interface for fgc-matchwatch.

stdout carries only real output (JSON from `status`, `align`, `streams list`);
all progress, warnings and errors go to stderr (and to MATCHWATCH_HOME/matchwatch.log).
Exit codes: 0 success, 1 one or more operations failed, 2 usage error,
3 YouTube refused access (a real stop: bot check / sign-in / 403), 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NoReturn

from . import __version__
from .errors import MatchwatchError, YouTubeBlocked

if TYPE_CHECKING:
    from .config import Config


class _UsageError(Exception):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    p = _ArgumentParser(prog="fgc-matchwatch",
                        description="Find FGC matches in the livestream, transcribe the "
                                    "commentary and post per-team scouting observations.")
    p.add_argument("-v", "--verbose", action="store_true", help="print detailed progress")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress non-error output")
    p.add_argument("--env-file", help="KEY=VALUE file (default MATCHWATCH_HOME/matchwatch.env)")
    p.add_argument("--year", help="season (default MATCHWATCH_YEAR or 2026)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", parser_class=_ArgumentParser)

    r = sub.add_parser("run", help="process new played matches of interest once")
    r.add_argument("--dry-run", action="store_true", help="show what would be processed")
    r.add_argument("--limit", type=int, help="at most N matches this run")
    r.add_argument("--matches", help="comma list of match keys instead of /api/interest")
    r.add_argument("--no-discover", action="store_true", help="do not look for new streams")

    w = sub.add_parser("watch", help="run, then repeat every --interval seconds")
    w.add_argument("--interval", type=int, default=300)
    w.add_argument("--once", action="store_true", help="a single pass (what the timer runs)")

    sub.add_parser("status", help="print state as JSON")

    s = sub.add_parser("streams", help="list or add streams")
    ss = s.add_subparsers(dest="scmd", parser_class=_ArgumentParser)
    ss.add_parser("list")
    ss.add_parser("discover", help="scan first.global/live and the channel")
    a = ss.add_parser("add", help="add a stream by YouTube id (manual override)")
    a.add_argument("video")
    a.add_argument("--field", type=int, help="override the field parsed from the title")

    t = sub.add_parser("transcribe", help="download + transcribe one stream")
    t.add_argument("video")

    al = sub.add_parser("align", help="print where every match sits in a stream (JSON)")
    al.add_argument("video")

    rec = sub.add_parser("record", help="record a live stream's audio until it ends")
    rec.add_argument("video")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        print(f"fgc-matchwatch: error: {exc}", file=sys.stderr)
        return 2
    if not args.cmd:
        parser.print_help(sys.stderr)
        return 2
    if args.year:
        os.environ["MATCHWATCH_YEAR"] = args.year
    if getattr(args, "matches", None):
        os.environ["MATCHWATCH_MATCHES"] = args.matches

    from . import config

    cfg = config.load(args.env_file)
    logf = None

    def log(message: str) -> None:
        nonlocal logf
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        try:
            if logf is None:
                cfg.home.mkdir(parents=True, exist_ok=True)
                logf = open(cfg.log_file, "a", buffering=1)  # noqa: SIM115 - lives for the process
            logf.write(line + "\n")
        except OSError:
            pass
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    try:
        return _dispatch(args, cfg, log)
    except YouTubeBlocked as exc:
        log(f"STOP: {exc}")
        return 3
    except MatchwatchError as exc:
        log(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        log("interrupted")
        return 130


def _dispatch(args: argparse.Namespace, cfg: Config, lg: Callable[[str], None]) -> int:
    from . import asr, media, pipeline

    if args.cmd == "status":
        print(json.dumps(pipeline.status(cfg), indent=1))
        return 0
    if args.cmd == "run":
        return 1 if pipeline.run_once(cfg, lg, dry_run=args.dry_run, limit=args.limit,
                                      discover=not args.no_discover) else 0
    if args.cmd == "watch":
        while True:
            rc = 1 if pipeline.run_once(cfg, lg) else 0
            if args.once:
                return rc
            time.sleep(args.interval)
    if args.cmd == "streams":
        cfg.ensure_dirs()
        streams = pipeline.load_streams(cfg)
        if args.scmd == "add":
            s = media.info(args.video).to_json()
            if args.field is not None:
                s["field"] = args.field
            s["manual"] = True
            streams[s["video"]] = s
            pipeline.save_streams(cfg, streams)
            lg(f"added {s['video']}: {s['title']} (day {s['day']}, field {s['field']})")
            return 0
        if args.scmd == "discover":
            streams = pipeline.refresh_streams(cfg, lg)
        print(json.dumps(streams, indent=1))
        return 0
    if args.cmd == "transcribe":
        cfg.ensure_dirs()
        streams = pipeline.load_streams(cfg)
        s = streams.get(args.video) or media.info(args.video).to_json()
        tr = pipeline.ensure_transcript(cfg, s, [], lg)
        lg(f"{args.video}: {len(tr['words']) if tr else 0} words")
        return 0
    if args.cmd == "align":
        streams = pipeline.load_streams(cfg)
        known = streams.get(args.video)
        if not known:
            raise MatchwatchError(f"unknown stream {args.video}; `streams add` it first")
        tr = asr.load_transcript(cfg.transcripts_dir / f"{args.video}.json")
        if not tr:
            raise MatchwatchError(f"no transcript for {args.video}; `transcribe` it first")
        sched = pipeline.schedule(cfg)
        from . import align

        fld, st = known.get("field"), known["start"]
        end = st + (tr.get("duration") or 0) * 1.6 + 3600
        same = [x for x in sched.matches if x.get("field") == fld and x.get("participants")
                and st - 1800 <= pipeline.match_time(x) <= end]
        res = align.align(tr["words"], st, [pipeline.spec_of(x) for x in same], sched.names)
        print(json.dumps([p.__dict__ for p in res], indent=1))
        return 0
    if args.cmd == "record":
        cfg.ensure_dirs()
        return media.record_live(args.video, cfg.audio_dir / f"{args.video}.live.ogg", lg)
    return 2
