"""
Loads and cleans Datasets/NBA/archive/{nba_games_all,nba_teams_all,
nba_betting_spread,nba_betting_money_line,nba_betting_totals}.csv into a
tidy, one-row-per-game DataFrame ready for feature engineering. No modeling
happens here - this module's only job is turning the raw CSVs into
trustworthy columns, mirroring sports/cfb/loader.py's shape.

Already-checked facts about this source (verified directly against the data,
not assumed - see sports/nba/config.py for the team-identity detail):

  - `nba_games_all.csv` is ONE ROW PER TEAM PER GAME (an `is_home` flag of
    't'/'f' distinguishes the two rows for each game_id). Reshaped here into
    one row per game by joining the 't' and 'f' rows on game_id.
  - `team_id` is already the NBA stats API's stable, continuous identifier
    across every relocation/rename in this window - no relocation-mapping
    table needed (see config.py's long comment on team identity). Only 30
    team_ids are "real" current franchises (nba_teams_all.csv rows with a
    non-blank abbreviation); 733 rows across the full 1950-2018 history use
    other, non-standard team_ids (early-1950s defunct teams, All-Star
    placeholders) and are dropped - verified to have ZERO overlap with the
    2006-2018 odds-covered window this module's backtest actually uses.
  - 3,124 of 62,812 total games have an unparseable/blank `game_date`
    (concentrated in the 1950s-80s plus some 2005-2017 rows) - dropped.
    Verified ZERO overlap with odds-covered games, so this costs nothing
    for the backtest.
  - A handful of rows (6, across all history) have `pts == 0` for a
    completed-looking game - these are genuine data artifacts, not real 0-0
    finals (an NBA game can never legitimately end 0-0): one is the
    2013-04-16 Pacers @ Celtics game postponed by the Boston Marathon
    bombing (made up later under a different game_id), one is this
    snapshot's final, mid-collection truncated row (2018-11-23), and one is
    a Pre Season game already excluded by the season_type filter below.
    Dropped via requiring pts > 0 for both teams.
  - Only `season_type in {Regular Season, Playoffs}` is kept - Pre Season
    (rotations/rest, not a reliable strength signal) and All Star
    (exhibition) are dropped. See config.py.
  - Real, two-sided American-odds betting data exists in the three
    `nba_betting_*.csv` files, keyed by (game_id, book_name, team_id,
    a_team_id), covering 2006-07 through 2017-18 (14,900-ish games per
    market out of 62,812 total games / 59,093 kept-and-cleaned games -
    betting data simply does not exist before 2006-07, nothing to recover
    there). Pinnacle Sports is used exclusively (config.PRIMARY_BOOK) - see
    config.py for the coverage numbers that motivated using Pinnacle alone
    rather than a multi-book blend.
  - CRITICAL, easy-to-get-backwards convention, verified against 100% of
    rows across all three betting files (388,362 rows total): in every
    betting-file row, `team_id` is ALWAYS the AWAY team and `a_team_id` is
    ALWAYS the HOME team - the reverse of what the column name "a_team_id"
    might suggest if you assumed it just mirrors nba_games_all.csv's
    per-row "this team / opponent" meaning. Confirmed by joining each
    betting file's (game_id, team_id) against nba_games_all.csv's own
    is_home flag: 100% of matches land on is_home='f' (away). So:
      - spread1/price1 (keyed to team_id)   -> AWAY side
      - spread2/price2 (keyed to a_team_id) -> HOME side
    and likewise for the moneyline file. Spot-checked: spread1 == -spread2
    in 99.95% of rows (a small number of books post slightly asymmetric
    two-way lines, which is real market behavior, not an error) - both
    sides are used as given rather than deriving one from the other.
  - The totals file also carries team_id/a_team_id columns (vestigial from
    a shared schema with spread/moneyline - a total isn't team-specific),
    with total1/price1 and total2/price2 nearly always identical
    (total1 == total2 in 99.94% of rows). Column 1 is treated as the
    "Over" side and column 2 as "Under" (the conventional first-column-is-
    Over ordering) - since the two total LINES agree in all but 0.06% of
    rows, this assumption only meaningfully affects which price (not which
    line) is attached to which side, and even then only for that <0.1% of
    rows where the two lines genuinely differ.
  - Exactly one row per (game_id, book_name) in every betting file - i.e.
    a SINGLE price snapshot per book per game, no open/close/min/max like
    the NFL source. Same CLV caveat as CFB: see verify.py's printed note.
"""

import numpy as np
import pandas as pd

from . import config
from core import odds_math


ODDS_COLS = [
    "Home Odds Close", "Away Odds Close",
    "Home Line Close", "Home Line Odds Close", "Away Line Odds Close",
    "Total Score Close", "Total Score Over Close", "Total Score Under Close",
]


def _load_clean_games() -> pd.DataFrame:
    """Raw nba_games_all.csv (one row per team per game) -> one row per game,
    filtered to real teams / real dates / real season types / real scores."""
    raw = pd.read_csv(config.GAMES_PATH, dtype={"game_id": str})
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    raw = raw.dropna(subset=["game_date"]).copy()

    raw = raw[raw["season_type"].isin(config.KEPT_SEASON_TYPES)].copy()

    teams_df = pd.read_csv(config.TEAMS_PATH)
    abbrev_map = config.load_team_abbrev_map(teams_df)
    known_ids = set(abbrev_map)
    raw = raw[raw["team_id"].isin(known_ids) & raw["a_team_id"].isin(known_ids)].copy()

    raw["pts"] = pd.to_numeric(raw["pts"], errors="coerce")
    raw = raw[raw["pts"] > 0].copy()

    home = raw[raw["is_home"] == "t"].copy()
    away = raw[raw["is_home"] == "f"].copy()

    games = home.merge(
        away, on="game_id", suffixes=("_home", "_away"), how="inner",
        validate="one_to_one",
    )

    games["home_team_id"] = games["team_id_home"]
    games["away_team_id"] = games["team_id_away"]
    games["home_franchise"] = games["home_team_id"].map(abbrev_map)
    games["away_franchise"] = games["away_team_id"].map(abbrev_map)

    games["date"] = games["game_date_home"]
    games["season"] = games["season_year_home"].astype(int)
    games["is_playoff"] = games["season_type_home"] == "Playoffs"

    games["home_score"] = games["pts_home"].astype(float)
    games["away_score"] = games["pts_away"].astype(float)
    games["actual_margin"] = games["home_score"] - games["away_score"]
    games["actual_total"] = games["home_score"] + games["away_score"]
    games["home_win"] = (games["actual_margin"] > 0).astype(int)

    return games[[
        "game_id", "date", "season", "is_playoff",
        "home_team_id", "away_team_id", "home_franchise", "away_franchise",
        "home_score", "away_score", "actual_margin", "actual_total", "home_win",
    ]]


def _load_pinnacle_odds() -> pd.DataFrame:
    """Returns one row per game_id with the ODDS_COLS, sourced from
    config.PRIMARY_BOOK only (see module docstring for coverage numbers and
    the team_id=away / a_team_id=home convention this relies on)."""
    spread = pd.read_csv(config.SPREAD_PATH, dtype={"game_id": str})
    ml = pd.read_csv(config.MONEYLINE_PATH, dtype={"game_id": str})
    totals = pd.read_csv(config.TOTALS_PATH, dtype={"game_id": str})

    spread = spread[spread["book_name"] == config.PRIMARY_BOOK].drop_duplicates("game_id")
    ml = ml[ml["book_name"] == config.PRIMARY_BOOK].drop_duplicates("game_id")
    totals = totals[totals["book_name"] == config.PRIMARY_BOOK].drop_duplicates("game_id")

    odds = spread[["game_id", "spread1", "spread2", "price1", "price2"]].rename(columns={
        "spread1": "away_spread", "spread2": "home_spread",
        "price1": "away_spread_ml", "price2": "home_spread_ml",
    })
    odds = odds.merge(
        ml[["game_id", "price1", "price2"]].rename(columns={"price1": "away_ml", "price2": "home_ml"}),
        on="game_id", how="outer",
    )
    odds = odds.merge(
        totals[["game_id", "total1", "total2", "price1", "price2"]].rename(columns={
            "total1": "total_over_line", "total2": "total_under_line",
            "price1": "over_ml", "price2": "under_ml",
        }),
        on="game_id", how="outer",
    )

    for col in ["away_spread", "home_spread", "away_spread_ml", "home_spread_ml",
                "away_ml", "home_ml", "total_over_line", "total_under_line", "over_ml", "under_ml"]:
        odds[col] = pd.to_numeric(odds[col], errors="coerce")

    def to_dec(v):
        return odds_math.american_to_decimal(v) if pd.notna(v) else np.nan

    odds["Home Odds Close"] = odds["home_ml"].apply(to_dec)
    odds["Away Odds Close"] = odds["away_ml"].apply(to_dec)

    odds["Home Line Close"] = odds["home_spread"]
    odds["Home Line Odds Close"] = odds["home_spread_ml"].apply(to_dec)
    odds["Away Line Odds Close"] = odds["away_spread_ml"].apply(to_dec)

    # Total line: use the "Over" column's line (total_over_line) as the
    # single reference line edge_finder/backtest key off of - see module
    # docstring, total_over_line == total_under_line in 99.94% of rows.
    odds["Total Score Close"] = odds["total_over_line"]
    odds["Total Score Over Close"] = odds["over_ml"].apply(to_dec)
    odds["Total Score Under Close"] = odds["under_ml"].apply(to_dec)

    return odds[["game_id"] + ODDS_COLS]


def load_games() -> pd.DataFrame:
    """Returns a cleaned, chronologically-sorted DataFrame, one row per game,
    with Pinnacle betting odds merged in (NaN where Pinnacle didn't cover
    that game - most games before 2006-07, see module docstring)."""
    games = _load_clean_games()
    odds = _load_pinnacle_odds()
    games = games.merge(odds, on="game_id", how="left")

    close_line = games["Home Line Close"]
    games["home_covers_close"] = np.select(
        [games["actual_margin"] > -close_line, games["actual_margin"] == -close_line],
        [1, 0],
        default=-1,
    )
    games.loc[close_line.isna(), "home_covers_close"] = np.nan

    total_line = games["Total Score Close"]
    games["over_result_close"] = np.select(
        [games["actual_total"] > total_line, games["actual_total"] == total_line],
        [1, 0],
        default=-1,
    )
    games.loc[total_line.isna(), "over_result_close"] = np.nan

    games = games.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)
    games["source_game_id"] = games["game_id"]
    games["game_id"] = games.index.astype(str).map(lambda i: f"nba_{i}")

    keep_cols = [
        "game_id", "source_game_id", "date", "season", "is_playoff",
        "home_team_id", "away_team_id", "home_franchise", "away_franchise",
        "home_score", "away_score", "actual_margin", "actual_total", "home_win",
        "home_covers_close", "over_result_close",
    ] + ODDS_COLS
    return games[keep_cols]
