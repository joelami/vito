"""
Downloads real NFL play-by-play (same free, MIT-licensed nflverse source
as fetch_nflfastr_team_game_epa.py -- see that script's own docstring for
the full "why nfl_data_py, why isolated" story) and aggregates it to a
per-(game, team) STARTING QB identity + that QB's own EPA/dropback --
the first PLAYER-category signal this project has ever had (see
core/factor_taxonomy.py's coverage_report("NFL"), which found zero
before this).

A SEPARATE dataset from nflfastr_team_game_epa.csv, deliberately, not a
column bolted onto it: that file is already fetched/committed-in-shape
for every existing user of it (sports/nfl/nflfastr_features.py), and
adding a new column would mean re-fetching and re-validating everything
downstream of it for no reason -- a new, independent CSV is the cleaner,
lower-risk change.

STARTING QB is a PROXY, not a real box-score "who started" flag (nflverse's
raw PBP doesn't carry one) -- the standard, externally-used convention:
whichever player has the most real pass attempts (dropbacks) for a team
in a given game is treated as that game's starter. This is what public
football-analytics work does when a real starter flag isn't available,
and it's very rarely wrong in the NFL specifically (a genuine QB change
mid-game from an injury is a real but small minority of team-games) --
documented as an approximation, not silently assumed exact.

Real metrics computed per (game_id, team):
  - starting_qb_id / starting_qb_name: the proxy-starter's own nflverse
    player id / display name.
  - qb_epa_per_dropback: that SPECIFIC QB's own mean EPA on their real
    dropbacks (pass attempts + sacks) in this one game -- a genuinely
    player-level number, not a team-level one like nflfastr_team_game_epa's
    off_epa_per_play (which blends the QB with the whole offensive unit
    and, on a run play, doesn't involve the QB's throwing at all).

Output: Datasets/NFL/nflfastr_qb_starts.csv -- one row per (game_id, team).

Run with: python3 scripts/fetch_nflfastr_qb_starts.py [start_season] [end_season]
(defaults to 2006-2025, matching nflfastr_team_game_epa.py's own range).
Same isolated-venv requirement as that script -- see this repo's
core/dataset_refresh.py for the production wiring.
"""

import sys
import time
from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent.parent / "Datasets" / "NFL"
OUT_PATH = DATASETS_DIR / "nflfastr_qb_starts.csv"

REAL_DROPBACK_TYPES = {"pass", "qb_spike"}  # sacks show up as play_type=="pass" with epa set, not a separate type in this column set


def aggregate_season(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl

    print(f"[fetch_nflfastr_qb] downloading season {season}...")
    t0 = time.time()
    pbp = nfl.import_pbp_data(
        [season],
        columns=["game_id", "season", "week", "posteam", "home_team", "away_team",
                 "play_type", "passer_player_id", "passer_player_name", "epa", "game_date"],
        downcast=True,
    )
    print(f"[fetch_nflfastr_qb] {season}: {len(pbp):,} plays downloaded in {time.time()-t0:.1f}s")

    dropbacks = pbp[pbp["play_type"].isin(REAL_DROPBACK_TYPES) & pbp["passer_player_id"].notna()
                     & pbp["epa"].notna()].copy()

    rows = []
    for (game_id, team), g in dropbacks.groupby(["game_id", "posteam"], observed=True):
        counts = g["passer_player_id"].value_counts()
        starter_id = counts.index[0]
        starter_name = g[g["passer_player_id"] == starter_id]["passer_player_name"].iloc[0]
        starter_plays = g[g["passer_player_id"] == starter_id]
        rows.append({
            "game_id": game_id, "season": season, "team": team,
            "starting_qb_id": starter_id, "starting_qb_name": starter_name,
            "qb_epa_per_dropback": float(starter_plays["epa"].mean()),
            "qb_dropbacks": int(len(starter_plays)),
            "gameday": g["game_date"].iloc[0],
        })
    df = pd.DataFrame(rows)
    # Same real regression class this fix's sibling script (fetch_nflfastr_
    # team_game_epa.py) hit and fixed 2026-09-15 -- guarded here from the
    # start rather than after the fact.
    assert df.empty or df["gameday"].notna().all(), \
        f"season {season}: {df['gameday'].isna().sum()} row(s) with no gameday -- a real regression, not expected data"
    return df


def fetch_and_save(start: int, end: int, force_seasons: set = frozenset()) -> pd.DataFrame:
    """Same resume/force-refresh contract as fetch_nflfastr_team_game_epa.py's
    own fetch_and_save() -- see that function's docstring for the full
    "why force_seasons exists" reasoning (the current, still-in-progress
    season needs periodic re-pulling, not a one-time fetch)."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    already_done = set(existing["season"].unique()) if not existing.empty else set()
    kept = existing[~existing["season"].isin(force_seasons)] if not existing.empty else existing
    all_seasons = [kept] if not kept.empty else []
    if already_done:
        print(f"[fetch_nflfastr_qb] resuming -- {len(already_done)} season(s) already in {OUT_PATH}: {sorted(already_done)}"
              + (f", force-refreshing {sorted(force_seasons & already_done)}" if force_seasons & already_done else ""))

    for season in range(start, end + 1):
        if season in already_done and season not in force_seasons:
            continue
        try:
            season_df = aggregate_season(season)
            all_seasons.append(season_df)
            pd.concat(all_seasons, ignore_index=True).to_csv(OUT_PATH, index=False)
            print(f"[fetch_nflfastr_qb] {season}: {len(season_df)} team-game rows -- saved to {OUT_PATH}")
        except Exception as e:
            print(f"[fetch_nflfastr_qb] {season} FAILED: {e}", file=sys.stderr)

    final = pd.concat(all_seasons, ignore_index=True) if all_seasons else pd.DataFrame()
    print(f"[fetch_nflfastr_qb] DONE -- {len(final)} total team-game rows across "
          f"{final['season'].nunique() if not final.empty else 0} seasons, saved to {OUT_PATH}")
    return final


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2006
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    force = {end} if "--force-last" in sys.argv else frozenset()
    fetch_and_save(start, end, force_seasons=force)


if __name__ == "__main__":
    main()
