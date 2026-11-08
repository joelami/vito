"""
Periodic in-season refresh for the real play-by-play/game-level datasets
this project pulls from nflverse/cfbfastR/MoneyPuck (see
sports/nfl/nflfastr_features.py, sports/cfb/cfbfastr_features.py, and
sports/nhl/moneypuck_xg_features.py) -- wired into scheduler.py's daily
_run_full() so the trailing EPA/success-rate/xG features stay current
week-to-week during the season, instead of frozen at whatever snapshot was
present the day the R2 volume was first populated.

Three genuinely different mechanisms, not a copy-paste oversight:

  CFB (refresh_cfb_success_data) and NHL (refresh_nhl_xg_data): both
  fetch scripts have zero extra dependencies beyond pandas (a plain
  HTTP GET) -- safe to call directly, in-process, no isolation needed.

  NFL (refresh_nfl_epa_data): the fetch script needs `nfl_data_py`, which
  hard-pins pandas<2.0 -- installing it into the MAIN app's environment
  would silently downgrade pandas and break core/backtest.py for every
  sport (a real regression this project already hit once, see
  requirements.txt's own comment). Run via a small, lazily-created,
  ISOLATED venv instead (persisted on the same volume as Datasets/, so
  it's built once, not on every boot) -- the main process's own
  dependencies are never touched.

All three are wrapped so a failure here (network hiccup, an upstream
provider's data not updated yet, a transient pip failure) never takes
down the scheduler's real job (settling/logging picks) -- logged loudly,
not silently swallowed, but never fatal.

All three return a status dict now (added 2026-10-12, see refresh_all()'s
own docstring for the real incident this closes) rather than None, so
scheduler.py can record what actually happened -- ok/fail and a detail
string -- into a place the running app can report back (/api/admin/
scheduler-status), instead of this only ever being knowable by grepping
Railway logs nobody watches proactively (the same real gap this
project's own scheduler heartbeat already exists to close for the
overall harness run -- see scheduler.py's scheduler_status docstring).

NHL added 2026-10-15 (app owner: "let's go back in with this season's
data... CFB, NFL & NHL"): the original MoneyPuck xG feature had NO
refresh mechanism at all -- its own module docstring honestly flagged
this gap but nothing closed it, so every live NHL matchup was scoring
off the PREVIOUS season's trailing form the entire 2026-27 season (see
sports/nhl/moneypuck_xg_features.py's own top docstring for the full
incident). Unlike NFL/CFB, there's no per-season endpoint to force-
refresh just the current year -- MoneyPuck's real team-level file is one
consolidated all-seasons download (confirmed directly, ~127MB, ~9s to
fetch) -- so refresh_nhl_xg_data() always does a full re-fetch, no
force_seasons concept needed.
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent.parent / "Datasets"
NFL_VENV_DIR = DATASETS_DIR / ".venvs" / "nfl_fetch"


def refresh_cfb_success_data() -> dict:
    try:
        from sports.cfb import config as cfb_config
        from scripts.fetch_cfbfastr_team_game_success import fetch_and_save

        current_season = cfb_config.season_for_date(datetime.now())
        fetch_and_save(current_season, current_season, force_seasons={current_season})
        print(f"[dataset_refresh] CFB success-rate data refreshed for season {current_season}")
        return {"ok": True, "detail": f"season {current_season} refreshed",
                "timestamp": datetime.now().isoformat()}
    except Exception as e:
        print(f"[dataset_refresh] CFB success-rate refresh FAILED (non-fatal, keeping existing data): {e}",
              file=sys.stderr)
        return {"ok": False, "detail": str(e), "timestamp": datetime.now().isoformat()}


def refresh_nhl_xg_data() -> dict:
    try:
        from scripts.fetch_moneypuck_team_xg import fetch_and_save
        from sports.nhl import moneypuck_xg_features

        fetch_and_save()
        moneypuck_xg_features.reset_caches()  # real fetch succeeded -- drop the stale cached table immediately, not next boot
        print("[dataset_refresh] NHL xG data refreshed")
        return {"ok": True, "detail": "refreshed", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        print(f"[dataset_refresh] NHL xG refresh FAILED (non-fatal, keeping existing data): {e}",
              file=sys.stderr)
        return {"ok": False, "detail": str(e), "timestamp": datetime.now().isoformat()}


def _venv_is_functional(venv_python: Path) -> bool:
    """
    Real bug this closes (2026-10-12, found while investigating the app
    owner's "why isn't the success-rate dataset refreshed every day"):
    _ensure_nfl_venv() used to treat "the venv DIRECTORY exists" as "the
    venv WORKS" -- but `python3 -m venv` creating the directory and `pip
    install nfl_data_py` succeeding are two separate subprocess calls;
    if the venv creation succeeds but the pip install fails for ANY
    reason (a transient network blip, a timeout, the container's outbound
    access being briefly unavailable during that one boot), the directory
    is left behind anyway -- a real, reproduced failure mode, confirmed
    directly: a bare venv with nothing installed still has a real
    `bin/python3`, so `venv_python.exists()` returns True, and the old
    code returned that broken interpreter as if it were ready to use.
    Every later call to the fetch script through it then dies at its own
    `import pandas` with ModuleNotFoundError -- caught and logged as a
    real failure (not silently swallowed), but because nothing ever
    re-checked the venv itself, that exact failure would repeat
    IDENTICALLY, forever, once a day, until a human manually deleted the
    venv directory. This function is the actual fix: verify real
    importability (not just path existence) every time, so a broken venv
    gets rebuilt instead of permanently wedging the refresh.
    """
    if not venv_python.exists():
        return False
    try:
        result = subprocess.run(
            [str(venv_python), "-c", "import nfl_data_py"],
            capture_output=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def _ensure_nfl_venv() -> Path:
    """Creates the isolated venv if it doesn't exist, OR if it exists but
    isn't actually functional (see _venv_is_functional()'s docstring for
    the real incident this guards against) -- rebuilt from scratch in
    that case (wiped, not patched, so a partial/corrupt install can't
    linger). Returns the venv's python executable path."""
    venv_python = NFL_VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python3")
    if _venv_is_functional(venv_python):
        return venv_python

    if NFL_VENV_DIR.exists():
        print(f"[dataset_refresh] existing venv at {NFL_VENV_DIR} is not functional (nfl_data_py "
              f"doesn't import) -- rebuilding from scratch rather than reusing a broken install.")
        shutil.rmtree(NFL_VENV_DIR)

    NFL_VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dataset_refresh] creating isolated venv at {NFL_VENV_DIR} for nfl_data_py "
          f"(persists on the Datasets volume from here on, re-verified on every call)...")
    subprocess.run([sys.executable, "-m", "venv", str(NFL_VENV_DIR)], check=True, timeout=120)
    subprocess.run([str(venv_python), "-m", "pip", "install", "--quiet", "nfl_data_py"],
                    check=True, timeout=600)
    return venv_python


def refresh_nfl_epa_data() -> dict:
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
            detail = result.stderr[-2000:]
            print(f"[dataset_refresh] NFL EPA refresh subprocess exited {result.returncode}: {detail}",
                  file=sys.stderr)
            return {"ok": False, "detail": detail, "timestamp": datetime.now().isoformat()}
        print(f"[dataset_refresh] NFL EPA data refreshed for season {current_season}")
        return {"ok": True, "detail": f"season {current_season} refreshed",
                "timestamp": datetime.now().isoformat()}
    except Exception as e:
        print(f"[dataset_refresh] NFL EPA refresh FAILED (non-fatal, keeping existing data): {e}",
              file=sys.stderr)
        return {"ok": False, "detail": str(e), "timestamp": datetime.now().isoformat()}


def refresh_all() -> dict:
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
    actually affected by this bug.

    Returns {"cfb": {...}, "nfl": {...}} status dicts (see
    refresh_cfb_success_data()/refresh_nfl_epa_data()'s own return
    shapes) -- real incident this closes (2026-10-12, app owner: "why
    isn't the success rate dataset get refreshed everyday like it
    should?"): before this, whether a refresh actually succeeded was
    only ever knowable by reading Railway logs (which nobody watches
    proactively -- the same real gap the overall scheduler heartbeat,
    scheduler.scheduler_status, already exists to close for the harness
    run itself). scheduler.py's _run_full() now stores this return value
    into that same status dict, exposed via /api/admin/scheduler-status,
    so "did today's refresh actually work, and if not why" is a real,
    checkable fact instead of a guess -- this is also what let a real,
    reproduced bug (see _venv_is_functional()'s docstring) get found and
    fixed instead of silently repeating forever.

    NHL added 2026-10-15 -- see module docstring. refresh_nhl_xg_data()
    resets its own cache internally on success (see that function), so
    nothing extra is needed here the way NFL's reset_caches() call below
    is."""
    cfb_result = refresh_cfb_success_data()
    nfl_result = refresh_nfl_epa_data()
    nhl_result = refresh_nhl_xg_data()

    from sports.nfl import nflfastr_features, qb_features
    nflfastr_features.reset_caches()
    qb_features.reset_cache()

    return {"cfb": cfb_result, "nfl": nfl_result, "nhl": nhl_result}
