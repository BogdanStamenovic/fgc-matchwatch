from __future__ import annotations

import json
from pathlib import Path

import pytest

from fgc_matchwatch import pipeline
from fgc_matchwatch.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "fgc-matchwatch" in capsys.readouterr().out


def test_usage_error_returns_2() -> None:
    assert main(["--bogus"]) == 2
    assert main([]) == 2


def test_status_on_empty_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("MATCHWATCH_HOME", str(tmp_path))
    assert main(["-q", "status"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["done"] == 0 and st["home"] == str(tmp_path)


def test_obs_id_stable_and_valid() -> None:
    a = pipeline.obs_id("t2-37", "KAZ")
    assert a == pipeline.obs_id("t2-37", "KAZ") and a != pipeline.obs_id("t2-37", "SRB")
    assert len(a) == 32 and a.isalnum()


def test_stream_for_prefers_field_stream() -> None:
    m = {"scheduledTime": "2025-10-30T11:29:00-05:00", "field": 1}
    t = pipeline.epoch(m["scheduledTime"])
    streams = {
        "main": {"video": "main", "field": None, "start": t - 4000, "duration": 30000,
                 "live_status": "was_live"},
        "f1": {"video": "f1", "field": 1, "start": t - 3000, "duration": 19000,
               "live_status": "was_live"},
        "f1old": {"video": "f1old", "field": 1, "start": t - 200000, "duration": 19000,
                  "live_status": "was_live"},
        "f2": {"video": "f2", "field": 2, "start": t - 3000, "duration": 19000,
               "live_status": "was_live"},
    }
    cfg = None
    assert pipeline.stream_for(cfg, streams, m)["video"] == "f1"  # type: ignore[arg-type,index]
    del streams["f1"]
    assert pipeline.stream_for(cfg, streams, m)["video"] == "main"  # type: ignore[arg-type,index]


def test_dry_run_with_explicit_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MATCHWATCH_HOME", str(tmp_path))
    sched = pipeline.Schedule([{"tournamentKey": "t2", "id": 1, "name": "Ranking Match 1",
                                "scheduledTime": "2025-10-30T11:15:00-05:00", "field": 1,
                                "participants": [], "played": True}], {})
    monkeypatch.setattr(pipeline, "schedule", lambda cfg, max_age=120: sched)
    assert main(["-q", "run", "--dry-run", "--matches", "t2-1,t2-999"]) == 0
