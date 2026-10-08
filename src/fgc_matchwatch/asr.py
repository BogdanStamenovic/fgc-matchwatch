"""Whisper transcription of stream audio, chunked and resumable.

A transcript is stored per audio source as JSON:
    {"source": id, "duration": s, "model": "...", "device": "...",
     "chunks": {"0": true, ...}, "words": [[start, end, "word"], ...]}
Times are seconds from the start of that audio file. Chunks are transcribed
independently so a crash or a reboot loses at most one chunk of work, and so a
growing live recording can be transcribed as it grows.
"""

from __future__ import annotations

import ctypes
import glob
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import MatchwatchError

SAMPLE_RATE = 16000
CHUNK_S = 1800
# Overlap so a word cut by a chunk boundary is heard whole in one of them.
OVERLAP_S = 5


def _preload_cuda_libs() -> None:
    """CTranslate2 dlopens libcublas/libcudnn by soname; the pip wheels put them
    under site-packages/nvidia/*/lib, which is not on the linker path, and
    LD_LIBRARY_PATH cannot be changed for the running process. Loading them
    RTLD_GLOBAL first makes the later dlopen find them."""
    for p in sorted(glob.glob(sys.prefix + "/lib/python*/site-packages/nvidia/*/lib/*.so*")):
        try:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def free_vram_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_holders() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def decode(path: Path, start: float = 0.0, duration: float | None = None) -> Any:
    import numpy as np

    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=False)
    if raw.returncode != 0:
        raise MatchwatchError(f"ffmpeg failed on {path}: {raw.stderr.decode()[-300:]}")
    return np.frombuffer(raw.stdout, np.int16).astype(np.float32) / 32768.0


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        pass
    # A live .ogg still being written often has no duration in its header;
    # a stream-copy scan reads packet times without decoding (seconds per hour).
    scan = subprocess.run(["ffmpeg", "-nostdin", "-v", "info", "-i", str(path), "-c", "copy", "-f", "null", "-"],
                          capture_output=True, text=True, check=False).stderr
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", scan)
    if times:
        h, m, s = times[-1]
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise MatchwatchError(f"cannot read duration of {path}")


# 2026 and 2025 game words plus the phrases alignment depends on. Whisper's
# initial prompt biases spelling of rare words; it does not force them.
VOCAB_PROMPT = (
    "FIRST Global Challenge. Ranking match number 37 on field 2. Red alliance, blue alliance. "
    "Three, two, one, go! Suppression unit, brace, extinguisher, fire shield, "
    "barriers, biodiversity units, accelerator, ecosystem."
)


@dataclass
class Engine:
    model_name: str
    device: str
    compute: str
    models_dir: Path

    _model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        if self.device == "cuda":
            _preload_cuda_libs()
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute,
            download_root=str(self.models_dir), cpu_threads=10,
        )

    def words(self, audio: Any) -> list[list[Any]]:
        self.load()
        segs, _info = self._model.transcribe(
            audio, language="en", word_timestamps=True, vad_filter=True, beam_size=1,
            initial_prompt=VOCAB_PROMPT, condition_on_previous_text=False,
        )
        out: list[list[Any]] = []
        for s in segs:
            for w in s.words or []:
                out.append([round(w.start, 2), round(w.end, 2), w.word.strip()])
        return out


def pick_engine(cfg: Any, log: Callable[[str], None]) -> Engine:
    """GPU when there is room for the model, CPU otherwise. Degrades, never fails."""
    want = cfg.whisper_device
    if want == "cpu":
        return Engine(cfg.cpu_model, "cpu", "int8", cfg.models_dir)
    free = free_vram_mib()
    if free is not None and free < cfg.min_free_vram_mib and cfg.gpu_release_cmd:
        log(f"only {free} MiB VRAM free; running MATCHWATCH_GPU_RELEASE_CMD")
        subprocess.run(cfg.gpu_release_cmd, shell=True, check=False, timeout=180)
        time.sleep(3)
        free = free_vram_mib()
    if free is not None and free >= cfg.min_free_vram_mib:
        return Engine(cfg.whisper_model, "cuda", cfg.whisper_compute, cfg.models_dir)
    if free is None:
        log("no usable GPU (nvidia-smi failed); using CPU Whisper")
    else:
        log(f"only {free} MiB VRAM free (need {cfg.min_free_vram_mib}); held by: "
            + "; ".join(gpu_holders()) + f". Falling back to CPU model {cfg.cpu_model}")
    return Engine(cfg.cpu_model, "cpu", "int8", cfg.models_dir)


def load_transcript(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(path)


def transcribe_file(
    engine: Engine,
    audio: Path,
    out: Path,
    source: str,
    log: Callable[[str], None],
    complete: bool = True,
) -> dict[str, Any]:
    """Transcribe `audio` into `out`, skipping chunks already done.

    complete=False marks the file as still growing (a live recording): the last,
    partial chunk is not marked done so it is redone once more audio exists.
    """
    data = load_transcript(out) or {"source": source, "chunks": {}, "words": []}
    duration = probe_duration(audio)
    data["duration"] = duration
    n = int(duration // CHUNK_S) + 1
    words = data["words"]
    for i in range(n):
        start = i * CHUNK_S
        end = min(duration, start + CHUNK_S)
        full = end - start >= CHUNK_S or complete
        if data["chunks"].get(str(i)) or end - start < 1:
            continue
        t0 = time.monotonic()
        pcm = decode(audio, start, end - start + OVERLAP_S)
        got = engine.words(pcm)
        took = time.monotonic() - t0
        # Drop words this chunk heard in the overlap that the next chunk owns,
        # and anything already stored for this span from a partial earlier run.
        words[:] = [w for w in words if not (start <= w[0] < end)]
        words.extend([[round(w[0] + start, 2), round(w[1] + start, 2), w[2]]
                      for w in got if w[0] < end - start])
        words.sort(key=lambda w: w[0])
        if full:
            data["chunks"][str(i)] = True
        data["model"] = engine.model_name
        data["device"] = engine.device
        data.setdefault("speed", []).append(round((end - start) / max(took, 1e-6), 1))
        _save(out, data)
        log(f"  {source} chunk {i + 1}/{n}: {end - start:.0f}s audio in {took:.1f}s "
            f"({(end - start) / max(took, 1e-6):.0f}x) on {engine.device}")
    data["complete"] = complete and all(data["chunks"].get(str(i)) for i in range(n))
    _save(out, data)
    return data
