"""
Downloads real CFB play-by-play (sportsdataverse/cfbfastR's raw PBP,
free, direct from GitHub releases -- no API key, confirmed 2026-09-14:
https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/
ncaa_mfb_pbp_cfbfastr, per-season CSV.gz, 2013+) and aggregates it to a
per-(game, team) SUCCESS RATE summary -- the CFB analog of sports/nfl/
nflfastr_features.py, with one real, honest difference explained below.

WHY THIS ISN'T EPA (and that's not a corner cut, it's an honest
constraint): cfbfastR's own EPA/WPA numbers are a MODEL OUTPUT (an
xgboost win-probability/expected-points model shipped with the R
package) -- the raw PBP release downloaded here does NOT include those
columns, confirmed directly (105 real columns, zero epa/wp/success
columns). Running that model requires a native xgboost + libomp runtime
this environment doesn't have (no Homebrew, no system package install
available) -- a genuine environment limitation, not something worth
faking around.

SUCCESS RATE is the real, externally-motivated substitute: unlike EPA,
it's not a model output at all -- it's a deterministic down-and-distance
RULE (the same one nflverse's own `success` column uses, and the same
one Football Outsiders/Bill Connelly's SP+ popularized for CFB
specifically): a play "succeeds" if it gains >=50% of yards-to-go on 1st
down, >=70% on 2nd, or >=100% (i.e. converts) on 3rd/4th. Computable
directly from this raw data's own down/distance/yards_gained columns, no
model needed -- a real, legitimate, if less granular, first play-level
efficiency signal for CFB, which has had ZERO play-level signal until
now (see core/factor_taxonomy.py's coverage_report()).

Output: Datasets/College Football/cfbfastr_team_game_success.csv -- one
row per (game, team), matching sports/nfl/nflfastr_features.py's shape
exactly so the same trailing-feature-building pattern applies.

Run with: python3 scripts/fetch_cfbfastr_team_game_success.py [start] [end]
(defaults to 2013-2025 -- 2013 is this data's own real earliest season).
No extra dependency needed beyond pandas -- unlike the NFL fetch script,
this reads a plain CSV.gz over HTTP, no nfl_data_py-style client library
(and so no pydantic/pandas version-pinning conflict risk either).
"""

import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent.parent / "Datasets" / "College Football"
OUT_PATH = DATASETS_DIR / "cfbfastr_team_game_success.csv"
RELEASE_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/ncaa_mfb_pbp_cfbfastr/ncaa_mfb_pbp_cfbfastr_{season}.csv.gz"

REAL_PLAY_TYPES = {"Rush", "Pass Reception", "Pass Incompletion", "Sack", "Passing Touchdown", "Rushing Touchdown"}


def success_rate_rule(row) -> bool:
    """Standard down-and-distance success definition (nflverse's own
    `success` column, Football Outsiders/Bill Connelly's SP+ for CFB
    specifically) -- NOT invented for this project."""
    down, distance, gained = row["down"], row["distance"], row["yards_gained"]
    if pd.isna(down) or pd.isna(distance) or pd.isna(gained) or distance <= 0:
        return None
    if down == 1:
        return gained >= 0.5 * distance
    if down == 2:
        return gained >= 0.7 * distance
    return gained >= distance  # 3rd/4th down: only a real conversion counts


def aggregate_season(season: int) -> pd.DataFrame:
    url = RELEASE_URL.format(season=season)
    print(f"[fetch_cfbfastr] downloading season {season}...")
    t0 = time.time()
    try:
        pbp = pd.read_csv(url, compression="gzip", low_memory=False,
                           usecols=["game_id", "offense_play", "defense_play", "home", "away",
                                    "play_type", "down", "distance", "yards_gained"])
    except Exception as e:
        raise RuntimeError(f"download/parse failed for {season}: {e}")
    print(f"[fetch_cfbfastr] {season}: {len(pbp):,} plays downloaded in {time.time()-t0:.1f}s")

    real_plays = pbp[pbp["play_type"].isin(REAL_PLAY_TYPES)].copy()
    real_plays["success"] = real_plays.apply(success_rate_rule, axis=1)
    real_plays = real_plays[real_plays["success"].notna()]

    rows = []
    for game_id, g in real_plays.groupby("game_id", observed=True):
        home_team = g["home"].iloc[0]
        away_team = g["away"].iloc[0]
        for team, is_home in ((home_team, True), (away_team, False)):
            off = g[g["offense_play"] == team]
            deff = g[g["defense_play"] == team]
            if off.empty and deff.empty:
                continue
            rows.append({
                "game_id": game_id, "season": season, "team": team, "is_home": is_home,
                "off_success_rate": off["success"].mean() if len(off) else None,
                "off_plays": len(off),
                "def_success_rate_allowed": deff["success"].mean() if len(deff) else None,
                "def_plays": len(deff),
            })
    return pd.DataFrame(rows)


def fetch_and_save(start: int, end: int, force_seasons: set = frozenset()) -> pd.DataFrame:
    """
    Callable core of this script, factored out of main() so
    core/dataset_refresh.py can call it in-process on a schedule (CFB's
    fetch has zero extra dependencies beyond pandas -- see this module's
    own docstring -- so, unlike NFL's equivalent, it's safe to run
    directly in the main app process, no isolated venv/subprocess needed).

    `force_seasons`: seasons to re-fetch and OVERWRITE even if already
    present -- real gap this closes: the plain resume-skip logic below is
    correct for a genuinely COMPLETE historical season (immutable, no
    reason to ever re-pull it) but wrong for the CURRENT, still-in-
    progress season, which gains new completed games every week. Without
    this, a periodic scheduler call would fetch the current season
    exactly once (whenever it first appears in the CSV) and then skip it
    forever, silently going stale for the rest of that entire season --
    the same "loud once, then silent forever" bug class this project has
    hit before (see core/matchup.py's _warned_missing_features comment).
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    already_done = set(existing["season"].unique()) if not existing.empty else set()
    # keep every already-done season EXCEPT ones this call was told to force-refresh
    kept = existing[~existing["season"].isin(force_seasons)] if not existing.empty else existing
    all_seasons = [kept] if not kept.empty else []
    if already_done:
        print(f"[fetch_cfbfastr] resuming -- {len(already_done)} season(s) already in {OUT_PATH}: {sorted(already_done)}"
              + (f", force-refreshing {sorted(force_seasons & already_done)}" if force_seasons & already_done else ""))

    for season in range(start, end + 1):
        if season in already_done and season not in force_seasons:
            continue
        try:
            season_df = aggregate_season(season)
            all_seasons.append(season_df)
            pd.concat(all_seasons, ignore_index=True).to_csv(OUT_PATH, index=False)
            print(f"[fetch_cfbfastr] {season}: {len(season_df)} team-game rows -- saved to {OUT_PATH}")
        except Exception as e:
            print(f"[fetch_cfbfastr] {season} FAILED: {e}", file=sys.stderr)

    final = pd.concat(all_seasons, ignore_index=True) if all_seasons else pd.DataFrame()
    print(f"[fetch_cfbfastr] DONE -- {len(final)} total team-game rows across "
          f"{final['season'].nunique() if not final.empty else 0} seasons, saved to {OUT_PATH}")
    return final


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2013
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    fetch_and_save(start, end)


if __name__ == "__main__":
    main()
