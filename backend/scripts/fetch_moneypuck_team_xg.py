"""
Downloads MoneyPuck's real, free, keyless, nightly-updated TEAM-level
per-game data (https://moneypuck.com/moneypuck/playerData/careers/
gameByGame/all_teams.csv -- confirmed directly, 2026-10-12: no API key,
no auth, served over plain HTTPS, "updated nightly" per MoneyPuck's own
data page) and aggregates it down to a small, real team-game xG working
file -- Datasets/NHL/moneypuck_team_game_xg.csv.

REPLACES the original skater-level approach (Datasets/NHL/
"2008_to_2024 copy 2.csv" + "2025 copy.csv"), which required averaging
each team's dressed skaters' individual on-ice xG as an approximate stand-
in for the real team total (see sports/nhl/moneypuck_xg_features.py's git
history / decision_log.jsonl's "nhl_moneypuck_trailing_xg_features" entry
for that approximation's own reasoning). This file's `position=="Team
Level"` rows carry the REAL team total directly (`xGoalsFor`/
`xGoalsAgainst`) -- confirmed by direct inspection of the raw file's own
columns -- so this is a strict correctness improvement, not just a
refresh mechanism: no more approximation needed at all.

Also, critically, this is the piece the original skater-level files never
had: a real, live, keyless, re-fetchable CURRENT-SEASON source, closing
the exact gap flagged (and left unresolved) in that module's own original
docstring ("nothing in core/dataset_refresh.py currently re-fetches this
MoneyPuck snapshot periodically... not currently wired"). Confirmed
directly: the file's own `season`/`gameDate` columns include real 2026-27
season rows (season==2026, max gameDate 20261011 as of this writing) --
this is a live, current data source, not a frozen historical one.

No per-season download endpoint exists (checked directly against
MoneyPuck's real download page) -- the ENTIRE multi-season file must be
re-downloaded on every refresh. Confirmed manageable: real measured size
~127MB / 233k raw rows (all seasons, all situations, all teams) as of
2026-10-12 -- small enough for a daily re-download, unlike the old
skater-level files' combined ~2.8GB. Filtered down to
position=="Team Level" AND situation=="all" (all-strengths-combined,
matching every other NHL trailing feature's own convention) before
saving -- the working file this produces is a few MB, not hundreds.

Run with: python3 scripts/fetch_moneypuck_team_xg.py (from backend/) --
always a full re-fetch+re-filter (no incremental/force_seasons concept,
unlike the NFL/CFB fetch scripts, since there's no per-season endpoint to
fetch incrementally from).
"""

import sys
import time
from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent.parent / "Datasets" / "NHL"
OUT_PATH = DATASETS_DIR / "moneypuck_team_game_xg.csv"
SOURCE_URL = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"

RAW_COLUMNS = ["team", "season", "gameId", "playerTeam", "opposingTeam", "home_or_away",
               "gameDate", "position", "situation", "xGoalsFor", "xGoalsAgainst", "playoffGame"]


def fetch_and_save() -> pd.DataFrame:
    print(f"[fetch_moneypuck_xg] downloading {SOURCE_URL} ...")
    t0 = time.time()
    raw = pd.read_csv(SOURCE_URL, usecols=RAW_COLUMNS, low_memory=False)
    print(f"[fetch_moneypuck_xg] {len(raw):,} raw rows downloaded in {time.time()-t0:.1f}s")

    team_level = raw[(raw["position"] == "Team Level") & (raw["situation"] == "all")].copy()
    team_level = team_level.drop(columns=["position", "situation"])
    team_level = team_level.rename(columns={"xGoalsFor": "xg_for", "xGoalsAgainst": "xg_against"})

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    team_level.to_csv(OUT_PATH, index=False)
    print(f"[fetch_moneypuck_xg] DONE -- {len(team_level):,} team-game rows "
          f"(seasons {sorted(team_level['season'].unique())}), saved to {OUT_PATH}")
    return team_level


def main():
    fetch_and_save()


if __name__ == "__main__":
    main()
