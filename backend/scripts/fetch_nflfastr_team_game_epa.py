"""
Downloads real NFL play-by-play data (nflverse/nflfastR, via the nfl_data_py
package -- free, MIT-licensed, updated nightly during the season, 1999+)
and aggregates it down to a per-(game, team) EPA/success-rate summary --
NOT the raw play-by-play, which is far too large to keep around (one
season alone is ~450MB in memory, 397 columns, ~50k rows). What this
project actually needs is a handful of team-game-level numbers to build
trailing features from, the same shape as every other trailing feature in
sports/nfl/features.py (team_last_n_pts_avg and friends).

Processes ONE SEASON AT A TIME deliberately -- download, aggregate,
discard the raw PBP, move to the next season -- so peak memory stays
bounded regardless of how many seasons get pulled, instead of holding
20 seasons x ~450MB simultaneously.

Real metrics computed per (game_id, team), offense and defense split:
  - off_epa_per_play / def_epa_per_play_allowed: mean `epa` on plays this
    team ran / faced. The single most-used team-strength metric in modern
    NFL analytics (a real, better-than-scoring-differential proxy for
    "how good is this offense/defense," per nflfastR's own stated purpose).
  - off_success_rate / def_success_rate_allowed: mean `success` (nflfastR's
    own down-and-distance-aware success definition -- not just "positive
    yardage"), a complementary, less-variance-prone signal than raw EPA.

Output: Datasets/NFL/nflfastr_team_game_epa.csv -- one row per (game_id,
team, is_home), small enough (roughly 32 teams x ~17-18 games x N seasons)
to load instantly and re-join against Vito's own games table by date/team
name (see sports/nfl/nflfastr_features.py for that join + the team-name
mapping nflverse's abbreviations need).

Run with: python3 scripts/fetch_nflfastr_team_game_epa.py [start_season] [end_season]
(defaults to 2006-2025, matching sports/nfl/loader.py's own historical range).

Deliberately NOT in requirements.txt: nfl_data_py hard-pins pandas<2.0
(confirmed directly -- pip's resolver refuses pandas>=2.2 the moment it's
installed alongside), which conflicts with the rest of this project's real
requirement on modern pandas. Nothing at runtime imports nfl_data_py or
needs it installed -- sports/nfl/nflfastr_features.py only ever reads the
CSV this script already produced, plain pandas, no special dependency.
Install it yourself, in a throwaway venv if you want to keep it off your
main environment entirely, only when actually re-running this fetch:
    pip install nfl_data_py
"""

import sys
import time
from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent.parent / "Datasets" / "NFL"
OUT_PATH = DATASETS_DIR / "nflfastr_team_game_epa.csv"

REAL_PLAY_TYPES = {"pass", "run"}  # exclude kneels/spikes/penalties/etc. -- not real offensive plays


def aggregate_season(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl

    print(f"[fetch_nflfastr] downloading season {season}...")
    t0 = time.time()
    pbp = nfl.import_pbp_data(
        [season],
        columns=["game_id", "season", "week", "posteam", "defteam", "home_team", "away_team",
                 "play_type", "epa", "success", "game_date"],
        downcast=True,
    )
    print(f"[fetch_nflfastr] {season}: {len(pbp):,} plays downloaded in {time.time()-t0:.1f}s")

    real_plays = pbp[pbp["play_type"].isin(REAL_PLAY_TYPES) & pbp["epa"].notna()].copy()

    rows = []
    for game_id, g in real_plays.groupby("game_id", observed=True):
        home_team = g["home_team"].iloc[0]
        away_team = g["away_team"].iloc[0]
        for team, is_home in ((home_team, True), (away_team, False)):
            off = g[g["posteam"] == team]
            deff = g[g["defteam"] == team]
            if off.empty and deff.empty:
                continue  # a team with zero recorded real plays this game -- don't fabricate a row
            rows.append({
                "game_id": game_id, "season": season, "team": team, "is_home": is_home,
                "off_epa_per_play": off["epa"].mean() if len(off) else None,
                "off_success_rate": off["success"].mean() if len(off) else None,
                "off_plays": len(off),
                "def_epa_per_play_allowed": deff["epa"].mean() if len(deff) else None,
                "def_success_rate_allowed": deff["success"].mean() if len(deff) else None,
                "def_plays": len(deff),
                # Real regression, found and fixed 2026-09-15: this key went
                # missing during this session's fetch_and_save() refactor,
                # which silently dropped "game_date" from the requested pbp
                # columns -- the 32 real rows fetched by that refactored
                # code (2026 season, via the new scheduler refresh path)
                # came back with gameday=NaN, which get_current_trailing_epa()
                # sorts by; caught by direct inspection before it could
                # quietly corrupt trailing EPA ordering for the live season.
                "gameday": g["game_date"].iloc[0],
            })
    df = pd.DataFrame(rows)
    # Guards the exact regression this key's own comment above describes --
    # fails loudly, at fetch time, rather than silently shipping a row
    # get_current_trailing_epa()'s sort-by-gameday would then mis-order.
    assert df.empty or df["gameday"].notna().all(), \
        f"season {season}: {df['gameday'].isna().sum()} row(s) with no gameday -- a real regression, not expected data"
    return df


def fetch_and_save(start: int, end: int, force_seasons: set = frozenset()) -> pd.DataFrame:
    """
    Callable core of this script, factored out of main() -- see
    sports/cfb's equivalent fetch script's identical fetch_and_save() for
    the full "why force_seasons exists" reasoning (the current,
    still-in-progress season needs re-pulling on a schedule, not
    skipping forever after its first appearance). Unlike CFB's version,
    this is NOT safe to call in the main app process directly (nfl_data_py
    hard-pins pandas<2.0 -- see this module's top docstring) -- callers
    must run it via an isolated venv/subprocess (see
    core/dataset_refresh.py) or a one-off manual invocation, never a
    top-level import into main.py/scheduler.py.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    already_done = set(existing["season"].unique()) if not existing.empty else set()
    kept = existing[~existing["season"].isin(force_seasons)] if not existing.empty else existing
    all_seasons = [kept] if not kept.empty else []
    if already_done:
        print(f"[fetch_nflfastr] resuming -- {len(already_done)} season(s) already in {OUT_PATH}: {sorted(already_done)}"
              + (f", force-refreshing {sorted(force_seasons & already_done)}" if force_seasons & already_done else ""))

    for season in range(start, end + 1):
        if season in already_done and season not in force_seasons:
            continue
        try:
            season_df = aggregate_season(season)
            all_seasons.append(season_df)
            # Save after EVERY season, not just at the end -- so a crash/
            # interrupt partway through many seasons of downloads loses at
            # most one season's work, not the whole run.
            pd.concat(all_seasons, ignore_index=True).to_csv(OUT_PATH, index=False)
            print(f"[fetch_nflfastr] {season}: {len(season_df)} team-game rows -- saved to {OUT_PATH}")
        except Exception as e:
            print(f"[fetch_nflfastr] {season} FAILED: {e}", file=sys.stderr)

    final = pd.concat(all_seasons, ignore_index=True) if all_seasons else pd.DataFrame()
    print(f"[fetch_nflfastr] DONE -- {len(final)} total team-game rows across "
          f"{final['season'].nunique() if not final.empty else 0} seasons, saved to {OUT_PATH}")
    return final


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2006
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    force = {end} if "--force-last" in sys.argv else frozenset()
    fetch_and_save(start, end, force_seasons=force)


if __name__ == "__main__":
    main()
