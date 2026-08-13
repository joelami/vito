"""
Loads and cleans the Retrosheet game logs in Datasets/MLB/gl{1990_99,2000_09,
2010_19,2020_25}/*.txt into a tidy, one-row-per-game DataFrame ready for
feature engineering. No modeling happens here - this module's only job is
turning the raw, header-less positional game logs into trustworthy columns,
mirroring sports/cfb/loader.py's shape exactly.

DIFFERENCE FROM sports/nfl/loader.py AND sports/cfb/loader.py: there is NO
betting-line data anywhere in THIS source (the Retrosheet game logs) - no
spread, no total, no moneyline, in any season. Every game log row here is
pure schedule/result data (teams, score, date, park, attendance). This
loader therefore has no ODDS_COLS, no devig step, no ATS/over-result columns
- there is nothing to compute them from in this file.

UPDATE: a real odds archive covering 2012-2021 (Datasets/MLB/archive/) was
found after this loader's first pass - see sports/mlb/odds_loader.py, which
joins it onto this loader's output by (date, franchise, franchise, score,
score). That archive does NOT cover the full 1990-2025 range this loader
does, so most of this module's history is still odds-free - callers that
want odds must call `odds_loader.attach_odds(load_games())` explicitly;
`load_games()` itself still returns pure Retrosheet data with no odds
columns, unchanged from the original design. See sports/mlb/verify.py for
the real backtest numbers this now supports.

Already-checked facts about this source (see config.py for the franchise-
mapping detail, not re-derived here):
  - Positional, comma-delimited, no header row. Field indices used here
    (0-18) are the schedule/result columns; everything after that in each
    row is pitch-by-pitch box-score and starting-lineup data, not needed for
    team-level ratings and dropped via `usecols`.
  - 83,228 total rows across 1990-2025. 13 of those are true final-score
    TIES (curfew/weather-suspended games from the 1990s-2010s that were
    never resumed - real, rare, and not something modern MLB rules allow to
    stand uncompleted anymore). A tie has no winner, which breaks the
    downstream `home_win` (0/1) and Elo actual-score conventions in a way
    that would silently misrepresent 13 games as home losses - dropped
    rather than mishandled, documented instead of silently discarded.
  - 68 rows have non-blank `completion_info` (a game suspended on one date,
    finished on a later date) and 36 have non-blank `protest_info` - in both
    cases the score recorded IS the true final score for that game per
    Retrosheet convention, so no special handling is needed beyond using the
    score as-is. 1 row has non-blank `forfeit_info` (1995-08-10 STL@LAN,
    forfeited to the visitor) - the recorded 2-1 score is the real final
    score, kept as-is.
  - Team codes are run through `config.canonical_team` so a
    relocated/renamed franchise's rating and rolling form carry through
    continuously (see config.py's FRANCHISE_CANONICAL for the exact mapping
    and how it was derived from this data).
"""

import glob
from pathlib import Path

import pandas as pd

from . import config


GAMELOG_COLS = [
    "date", "game_number", "day_of_week",
    "away_team", "away_league", "away_game_number",
    "home_team", "home_league", "home_game_number",
    "away_score", "home_score", "length_of_game_outs",
    "day_night", "completion_info", "forfeit_info", "protest_info",
    "park_id", "attendance", "time_of_game_minutes",
]


def _iter_files():
    for sub in config.DATA_GLOB_DIRS:
        for f in sorted(glob.glob(str(config.DATA_DIR / sub / "gl*.txt"))):
            yield f


def load_games() -> pd.DataFrame:
    """Returns a cleaned, chronologically-sorted DataFrame, one row per game."""
    frames = []
    for f in _iter_files():
        frames.append(pd.read_csv(
            f, header=None, usecols=range(len(GAMELOG_COLS)),
            names=GAMELOG_COLS, dtype=str, keep_default_na=False,
        ))
    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"]).copy()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    # True ties (13 rows, all pre-2017, all curfew/weather-suspended games
    # never resumed) - no winner, dropped rather than mislabeled as a home
    # loss. See module docstring.
    df = df[df["home_score"] != df["away_score"]].copy()

    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    # ---------- Team identity & season ----------
    df["home_franchise"] = df["home_team"].apply(config.canonical_team)
    df["away_franchise"] = df["away_team"].apply(config.canonical_team)

    df["season"] = df["date"].apply(config.season_for_date).astype(int)

    # ---------- Cheap, real, non-fabricated context flags ----------
    # Interleague games (AL team @ NL team or vice versa) are a real signal
    # present directly in the data (columns 4/7), unlike CFB's is_bowl/NFL's
    # is_divisional this costs nothing to compute honestly.
    df["is_interleague"] = (df["away_league"] != df["home_league"]).astype(int)
    df["is_night"] = (df["day_night"] == "N").astype(int)

    df["actual_margin"] = df["home_score"] - df["away_score"]     # home perspective, positive = home won by
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["actual_margin"] > 0).astype(int)

    # sort by date, then game_number (doubleheader game 1 before game 2) as
    # a deterministic tiebreaker for same-day games
    df = df.sort_values(["date", "game_number", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    df["game_id"] = df.index.astype(str).map(lambda i: f"mlb_{i}")

    keep_cols = [
        "game_id", "date", "season", "game_number", "park_id", "attendance",
        "is_interleague", "is_night",
        "home_team", "away_team", "home_franchise", "away_franchise",
        "home_league", "away_league",
        "home_score", "away_score", "actual_margin", "actual_total", "home_win",
    ]
    return df[[c for c in keep_cols if c in df.columns]]
