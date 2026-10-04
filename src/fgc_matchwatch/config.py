"""Runtime configuration, all from the environment.

Everything heavy (audio, models, transcripts, state) lives under
MATCHWATCH_HOME, which defaults to /mnt/offload/fgc-matchwatch because the
root filesystem on archserver has ~7 GB free and a Whisper model plus a day of
stream audio would eat a third of it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOME = "/mnt/offload/fgc-matchwatch"


def _load_env_file(path: Path) -> None:
    """KEY=VALUE lines into os.environ, without overriding what is already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Config:
    home: Path
    scout_url: str
    scout_key: str
    year: str
    whisper_model: str
    whisper_compute: str
    whisper_device: str
    cpu_model: str
    min_free_vram_mib: int
    llm_cmd: list[str]
    llm_model: str
    live_page: str
    gpu_release_cmd: str
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def audio_dir(self) -> Path:
        return self.home / "audio"

    @property
    def transcripts_dir(self) -> Path:
        return self.home / "transcripts"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    # Per season: match keys like "t2-98" repeat every year, so one state file
    # would make 2026's match 98 look done because 2025's was.
    @property
    def state_file(self) -> Path:
        return self.home / f"state-{self.year}.json"

    @property
    def streams_file(self) -> Path:
        return self.home / f"streams-{self.year}.json"

    @property
    def log_file(self) -> Path:
        return self.home / "matchwatch.log"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    def ensure_dirs(self) -> None:
        for d in (self.home, self.audio_dir, self.transcripts_dir, self.models_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


def load(env_file: str | None = None) -> Config:
    home = Path(os.environ.get("MATCHWATCH_HOME", DEFAULT_HOME))
    _load_env_file(Path(env_file) if env_file else home / "matchwatch.env")
    home = Path(os.environ.get("MATCHWATCH_HOME", str(home)))
    return Config(
        home=home,
        scout_url=os.environ.get("SCOUT_URL", "http://127.0.0.1:3077").rstrip("/"),
        scout_key=os.environ.get("SCOUT_KEY", ""),
        year=os.environ.get("MATCHWATCH_YEAR", "2026"),
        # Measured on the RTX 4060 (see README): turbo at int8_float16 is
        # ~54x realtime on commentary and needs ~1.2 GiB, so it fits next to
        # cvoiced's resident scorer. large-v3 did not fit in the 2.7 GiB left.
        whisper_model=os.environ.get("MATCHWATCH_WHISPER", "large-v3-turbo"),
        whisper_compute=os.environ.get("MATCHWATCH_COMPUTE", "int8_float16"),
        whisper_device=os.environ.get("MATCHWATCH_DEVICE", "auto"),
        cpu_model=os.environ.get("MATCHWATCH_CPU_WHISPER", "small"),
        min_free_vram_mib=int(os.environ.get("MATCHWATCH_MIN_VRAM_MIB", "1800")),
        llm_cmd=os.environ.get("MATCHWATCH_LLM_CMD", "claude -p").split(),
        llm_model=os.environ.get("MATCHWATCH_LLM_MODEL", "sonnet"),
        live_page=os.environ.get("MATCHWATCH_LIVE_PAGE", "https://first.global/live/"),
        # Optional shell command run when VRAM is short, e.g. asking another
        # resident model server to unload. Empty means: fall back to CPU.
        gpu_release_cmd=os.environ.get("MATCHWATCH_GPU_RELEASE_CMD", ""),
    )
