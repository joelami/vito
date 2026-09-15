"""
Periodic in-season refresh for the two real play-by-play-derived datasets
this project pulls from nflverse/cfbfastR (see sports/nfl/nflfastr_features.py
and sports/cfb/cfbfastr_features.py) -- wired into scheduler.py's daily
_run_full() so the trailing EPA/success-rate features stay current
week-to-week during the season, instead of frozen at whatever snapshot was
present the day the R2 volume was first populated.

Two genuinely different mechanisms, not a copy-paste oversight:

  CFB (refresh_cfb_success_data): the fetch script has zero extra
  dependencies beyond pandas (a plain CSV.gz over HTTP) -- safe to call
  directly, in-process, no isolation needed.

  NFL (refresh_nfl_epa_data): the fetch script needs `nfl_data_py`, which
  hard-pins pandas<2.0 -- installing it into the MAIN app's environment
  would silently downgrade pandas and break core/backtest.py for every
  sport (a real regression this project already hit once, see
  requirements.txt's own comment). Run via a small, lazily-created,
  ISOLATED venv instead (persisted on the same volume as Datasets/, so
  it's built once, not on every boot) -- the main process's own
  dependencies are never touched.

Both are wrapped so a failure here (network hiccup, nflverse's upstream
data not updated yet, a transient pip failure) never takes down the
scheduler's real job (settling/logging picks) -- logged loudly, not
silently swallowed, but never fatal.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent.parent / "Datasets"
NFL_VENV_DIR = DATASETS_DIR / ".venvs" / "nfl_fetch"


def refresh_cfb_success_data() -> None:
    try:
        from sports.cfb import config as cfb_config
        from scripts.fetch_cfbfastr_team_game_success import fetch_and_save

        current_season = cfb_config.season_for_date(datetime.now())
        fetch_and_save(current_season, current_season, force_seasons={current_season})
        print(f"[dataset_refresh] CFB success-rate data refreshed for season {current_season}")
    except Exception as e:
        print(f"[dataset_refresh] CFB success-rate refresh FAILED (non-fatal, keeping existing data): {e}",
              file=sys.stderr)


def _ensure_nfl_venv() -> Path:
    """Creates the isolated venv (once) if it doesn't exist yet, installs
    nfl_data_py into it. Returns the venv's python executable path.
    Idempotent -- safe to call every refresh, a no-op after the first
    successful run since NFL_VENV_DIR then already exists."""
    venv_python = NFL_VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python3")
    if venv_python.exists():
        return venv_python

    NFL_VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dataset_refresh] creating isolated venv at {NFL_VENV_DIR} for nfl_data_py "
          f"(one-time, persists on the Datasets volume from here on)...")
    subprocess.run([sys.executable, "-m", "venv", str(NFL_VENV_DIR)], check=True, timeout=120)
    subprocess.run([str(venv_python), "-m", "pip", "install", "--quiet", "nfl_data_py"],
                    check=True, timeout=600)
    return venv_python


def refresh_nfl_epa_data() -> None:
    try:
        from sports.nfl import config as nfl_config

        current_season = nfl_config.season_for_date(datetime.now())
        venv_python = _ensure_nfl_venv()
        fetch_script = Path(__file__).parent.parent / "scripts" / "fetch_nflfastr_team_game_epa.py"
        result = subprocess.run(
            [str(venv_python), str(fetch_script), str(current_season), str(current_season), "--force-last"],
            cwd=str(fetch_script.parent.parent),  # backend/ -- same cwd fetch scripts already assume for DATASETS_DIR
            capture_output=True, text=True, timeout=900,
        )
        for line in result.stdout.splitlines():
            print(f"[dataset_refresh] {line}")
        if result.returncode != 0:
            print(f"[dataset_refresh] NFL EPA refresh subprocess exited {result.returncode}: "
                  f"{result.stderr[-2000:]}", file=sys.stderr)
        else:
            print(f"[dataset_refresh] NFL EPA data refreshed for season {current_season}")
    except Exception as e:
        print(f"[dataset_refresh] NFL EPA refresh FAILED (non-fatal, keeping existing data): {e}",
              file=sys.stderr)


def refresh_all() -> None:
    """Called once per day from scheduler.py's _run_full(), before the
    day's pipelines are rebuilt, so the freshly-pulled data is actually
    reflected in that same day's ratings/predictions -- not a day behind.

    Also clears the in-process "current form" caches in
    sports/nfl/nflfastr_features.py and sports/nfl/qb_features.py -- real
    bug found and fixed 2026-09-15 (app owner: "the data looks stale to
    me"): those caches populate once, on first use, and never invalidate
    themselves, so a long-lived Railway process would silently keep
    reusing whatever was in memory from the FIRST pipeline build after
    boot for the rest of that deployment's uptime, no matter how many
    days of fresh data this function pulled onto disk. See
    nflfastr_features.reset_caches()'s own docstring for the full story.
    CFB's equivalent (sports/cfb/cfbfastr_features.py) does NOT need this
    -- checked directly: its own load_team_game_success() already
    re-reads and re-caches unconditionally on every call, so it was never
    actually affected by this bug."""
    refresh_cfb_success_data()
    refresh_nfl_epa_data()

    from sports.nfl import nflfastr_features, qb_features
    nflfastr_features.reset_caches()
    qb_features.reset_cache()
