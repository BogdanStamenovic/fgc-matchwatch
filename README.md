# fgc-matchwatch

Finds FIRST Global Challenge matches in FIRST Global's YouTube livestreams,
transcribes the field commentary with Whisper on the local GPU, has an LLM
extract what each team in the match is good at, bad at, tried to do and
struggled with, and posts one observation per team per match into the
[fgc-scout](https://github.com/BogdanStamenovic/fgc-scout) knowledge base.

It is a separate project from fgc-scout on purpose: Whisper weights, CUDA
libraries and yt-dlp (about 2.6 GB of venv plus 1.6 GB of model) do not belong
in a light scouting server. This side is heavy and runs on archserver; the
scout server stays light and only receives JSON.

## How it works

```
/api/interest ──► schedule (api.first.global) ──► which stream? ──► audio ──► Whisper ──► align ──► LLM ──► verify ──► POST /api/observation
```

1. **Interest.** `GET /api/interest` on the scout server returns played
   matches worth scouting, highest priority first.
2. **Streams.** FIRST Global streams every field separately. In 2025 the
   official page (first.global/live) embedded five *unlisted* YouTube streams
   per day, titled `2025 FIRST Global Challenge - Day N, Field M`, next to the
   main program feed on the channel (`@F1RSTglobal`). Fields 1–4 have their own
   play-by-play announcer; field 5 is the main stage, so its stream carries the
   booth commentary. `streams discover` scrapes the live page and the channel
   and parses those titles; `streams add <id>` is the manual override.
   A match maps to the stream of its field that was live at its scheduled time;
   if there is none, it falls back to the main feed.
3. **Audio.** A finished stream (VOD) is downloaded as 63 kbps opus, about
   30 MB per hour. A live stream is recorded by `fgc-matchwatch record <id>`
   with `--live-from-start`, as a transient systemd user unit that the timer
   starts on its own. Audio is deleted once its transcript is complete.
4. **Transcription.** faster-whisper, word timestamps, VAD, in 30-minute
   chunks, so a crash or a reboot loses at most one chunk and a growing live
   recording can be transcribed as it grows. One transcript per stream: field
   streams are mostly silence between matches, so this costs 2–3 minutes per
   field-day and aligning every match on the field together is more robust
   than cutting each match blind.
5. **Alignment** (`align.py`, pure logic, see the docstring). The anchors are
   "three, two, one, go" countdowns, each also taken as a possible end-of-match
   count 150 s after a start; "match N" call-outs; and, only where no countdown
   is near, the point where play talk jumps. Each (match, anchor) pair is scored
   on how many of the match's six countries are named around it, minus countries
   not in the match, plus a bonus when the match number or both official scores
   are said, plus play talk after versus before, plus a prior around the
   scheduled time. Matches on a field run in schedule order, so a dynamic
   program assigns them to anchors in increasing time. Drift is then fitted
   from the confident placements and the assignment is rerun with a tighter
   prior. VODs drop paused stretches (field 1 on Day 1 2025 lost about 38
   minutes at lunch), and the local drift fit absorbs that.
6. **LLM.** One `claude -p --model sonnet` call per match (keyless, no tools,
   no MCP, custom system prompt) covers all six teams. Every claim must carry a
   verbatim quote. The prompt is year-aware: 2026 vocabulary by default
   (SUPPRESSION UNIT, BRACE zones, EXTINGUISHER, FIRE SHIELD, partner climb);
   2025 vocabulary is used only with `--year 2025`, for the test bed.
7. **Verification (the feedback loop).** Each quote must be found in the
   transcript (fuzzy ≥ 88, to forgive punctuation), and the team must be named
   within 30 s before or 12 s after it, so a claim about "they" cannot be pinned
   on the wrong robot. Claims that fail are dropped and counted. The free-text
   summary is kept only if at least one claim survived. Invalid JSON gets one
   retry with the parser error fed back, and a second failure skips the match.
8. **POST.** One observation per team, id = sha256(matchKey|code|pipeline
   version), so a re-POST replaces rather than duplicates.

State (`state.json`) is written after every match: finished matches are never
redone, and a match that cannot be placed yet is retried on the next runs (up
to 6 times, so about 30 minutes under the timer) before it is given up.

## Measured (2025 test bed, 4 Oct 2026)

| What | Number |
|---|---|
| Whisper large-v3-turbo, int8_float16, RTX 4060: field streams (mostly silence) | median 124–180x realtime, ~1.2 GiB VRAM |
| same, main stage / main feed (constant talk) | median 60–72x realtime |
| distil-large-v3 int8_float16 on the same 10-min clip | 71x (turbo: 54x); caught the same match-number call-outs |
| large-v3 int8_float16 | **did not fit**: OOM with 2.7 GiB free (cvoiced holds 5.5 GiB) |
| CPU fallback, `small` int8, 10 threads | 11.6x realtime on talk |
| Field streams aligned (Day 1 + Day 3, fields 1–5) | 188 matches; 182 placed confidently (96.8%); the rest have no usable commentary and are skipped |
| Fresh hand-checked sample 1 (15 matches never looked at while tuning) | 15/15 right match, 15/15 start within ~3 s of the spoken "go" |
| Fresh hand-checked sample 2 (15 more, drawn after the first round of fixes) | 14/15 right and within ~3 s; 1 wrong but "confident" (t2-41: its play-by-play is missing from the VOD, and it was placed on the score read-out). Fixed since: it is now skipped |
| Tuning set: 54 hand-labelled starts (`bench/onset_sweep.py`) | 53/54 within 30 s, 52/54 within 10 s; not independent, it is what the fixes were tuned on |
| Main feed vs. field-5 stream, same 35 Day 1 matches (the same booth audio, so this checks consistency, not truth) | 30/35 agree to within 1 s; of the other 5, 2 sit at a VOD gap, 1 match was restarted, 2 differ by 100–200 s |
| Ablated aligner (no number/score features) vs. spoken match number or score read-out | 0 verified wrong among confident placements (2 flagged, both checker false alarms) |
| LLM (Sonnet via `claude -p`), 12 matches / 72 observations | 12/12 valid JSON first try; 132 claims kept, 13 dropped by verification (9%); 57/72 observations have at least one fact |

Why turbo over distil: distil was 30% faster on the clip, but both are far
faster than needed (a 5 h field-day takes 2–3 minutes), and turbo is the
multilingual model, so it handles accented commentary and Spanish-heavy
moments better. That last part is reasoning, not a measurement.

## Install

```sh
ownbox install fgc-matchwatch
```

or by hand (the venv is ~2.6 GB, so on archserver it lives on /mnt/offload):

```sh
git clone https://github.com/BogdanStamenovic/fgc-matchwatch ~/data/fgc-matchwatch
cd ~/data/fgc-matchwatch
uv venv /mnt/offload/fgc-matchwatch/venv && ln -s /mnt/offload/fgc-matchwatch/venv .venv
uv pip install --python .venv/bin/python -e ".[gpu,dev]"
./install.sh        # systemd user units, timer left DISABLED
```

Needs `ffmpeg`, `node` (yt-dlp's JS runtime for YouTube), and `claude`
logged in. Config goes in `/mnt/offload/fgc-matchwatch/matchwatch.env` (never
in git):

```sh
SCOUT_URL=https://scout.example
SCOUT_KEY=...
# optional
MATCHWATCH_YEAR=2026
MATCHWATCH_WHISPER=large-v3-turbo
MATCHWATCH_MIN_VRAM_MIB=1800
MATCHWATCH_GPU_RELEASE_CMD=   # e.g. a curl that asks another model server to unload
```

## Usage

| Command | What it does |
|---|---|
| `fgc-matchwatch run [--dry-run] [--limit N] [--matches k1,k2]` | one pass over new played matches of interest |
| `fgc-matchwatch watch [--once] [--interval S]` | `run` in a loop; `--once` is what the timer runs |
| `fgc-matchwatch status` | state as JSON: done, pending with reasons, streams, stats, free VRAM |
| `fgc-matchwatch streams discover \| list \| add <id> [--field N]` | find or override the stream list |
| `fgc-matchwatch transcribe <id>` | download and transcribe one stream |
| `fgc-matchwatch align <id>` | where every match sits in a stream, as JSON |
| `fgc-matchwatch record <id>` | record a live stream's audio until it ends |

Global flags: `--year`, `--env-file`, `-v`, `-q`, `--version`. stdout carries
only JSON output; progress goes to stderr and `matchwatch.log`.

Exit codes: 0 ok · 1 something failed (see log) · 2 usage · **3 YouTube refused
access** · 130 interrupted.

### During the event

```sh
systemctl --user enable --now fgc-matchwatch.timer
```

Runs every 5 minutes, 00:00–10:55 UTC (09:00–19:55 KST), 7–10 Oct 2026. Each
run discovers streams, starts recorders for live ones, transcribes new audio,
and processes new played matches of interest. archserver must be awake during
those hours with ≥ 1.8 GiB VRAM free; otherwise Whisper falls back to the CPU
`small` model (~12x realtime, lower quality, still works).

### Test bed (2025)

```sh
fgc-matchwatch --year 2025 streams add Hy2VGJjoMoo     # 2025 Day 1, Field 1, etc.
SCOUT_URL=http://127.0.0.1:3079 SCOUT_KEY=dev \
  fgc-matchwatch --year 2025 run --matches t2-98,t2-361
```

with a local scout server started as `SCOUT_KEY=dev SCOUT_YEAR=2025 PORT=3079
node server/server.mjs`, whose `/api/interest` then returns Serbia's 2025 matches.

## Limitations

- **YouTube blocks are a hard stop.** A bot check, sign-in wall or 403 exits
  with code 3. There are no cookies, proxies or user-agent tricks, by rule.
  If YouTube starts asking archserver to sign in, the tool stops working.
- **Per-field streams have to be findable.** In 2025 they were unlisted videos
  linked only from first.global/live. For Day 2 2025 none were recoverable
  afterwards (the Wayback Machine has no snapshot of the page that day), so
  Day 2's fields 1–4 have no source and those matches are given up on.
  If 2026 embeds them differently, run `streams add <id>` by hand.
- **The live path is only partly tested.** Recording (`--live-from-start`,
  22 min of DVR fetched in 90 s), the transient systemd unit and incremental
  transcription were exercised on a NASA live stream. A real FIRST Global
  live day, and a stream ending and turning into a VOD, were not.
- **startTime is unused in practice.** The per-match API has a `startTime`
  field, but it was empty for every 2025 match. If it is filled live in 2026 it
  becomes the prior; the alignment does not depend on it.
- **Main-stage booth talk is weaker evidence.** Field 5 commentators
  interview guests during matches, and the main feed sometimes cuts to pit
  interviews. Alignment is less certain there (the 2 main-feed vs. field-5
  disagreements above), and observations from those matches often say little
  about the teams.
- **Matches with no commentary produce nothing.** Some 2025 matches (t2-43,
  t2-41) fall in stretches of VOD with no speech for the match itself.
- **Misheard names lose facts, never invent them.** "Lesotho" transcribed as
  "Lizotho" means its quotes are not attributed and its observation is empty.
  The alias table in `align.py` is the fix, one country at a time.
- **The summary is LLM prose.** It is published only when at least one
  verified claim exists, but its sentences are not individually verified.
- **2026 vocabulary is untested on 2026 audio**, because none exists yet. The
  prompt and Whisper's initial prompt carry the 2026 terms; whether the
  commentators actually say "brace" and "suppression unit" is unknown.
- **The final "go" time is a few seconds early or late.** It is the moment
  the countdown hits zero in Whisper's word timestamps, not a frame-accurate
  start.
