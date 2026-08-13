"""
Loads and cleans Datasets/NHL/archive/nhl_data_plus.csv into a tidy,
one-row-per-game DataFrame ready for feature engineering. No modeling
happens here - this module's only job is turning the raw, one-row-per-
TEAM-PER-GAME CSV into trustworthy, one-row-per-game columns, mirroring
sports/nba/loader.py's pivot shape (the closest structural match of any
existing sport here) and sports/cfb/loader.py's odds-handling shape.

See sports/nhl/config.py's module docstring FIRST for the full list of
corrections to the build brief (file roles were swapped, row shape was
wrong) and the detailed reasoning behind every cleaning decision below -
this docstring only covers the mechanics.
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

# Same assumed standard juice CFB uses for its single-sided spread/total -
# no two-sided price exists for either market in this data, see config.py.
ASSUMED_JUICE_DECIMAL = odds_math.american_to_decimal(-110.0)   # 1.9090909...
# Total market overround implied by -110/-110 both sides (4.7619%) - reused
# as the assumed overround when backing out the non-favorite moneyline side
# from `favorite_moneyline` (see config.py's MONEYLINE section for why this
# specific number and its limitations).
ASSUMED_ML_OVERROUND = 2.0 * odds_math.implied_prob(ASSUMED_JUICE_DECIMAL) - 1.0

PRESEASON_RESET_MAX_TOTAL_GAMES = 3   # a post-reset record.total_games <= this is treated as "season restarted"


def _drop_unplayed_placeholder_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    goals_for == goals_against == 0 never happens in a real completed NHL
    game (every completed game has a winner, and shootout winners are
    credited a goal in this data's final score - see config.py). Rows like
    this are schedule placeholders for games that were never actually
    played: the entirely-cancelled 2004-05 lockout season (~2,470 rows),
    ~194 COVID-postponed March/April 2020 games (frozen record, real dates,
    0-0 score), and a handful of isolated single-row data glitches elsewhere
    (a wrong record_summary value injected once mid-sequence, immediately
    reverting). Dropped rather than mishandled, same spirit as CFB's
    completed=0 filter and MLB's true-tie drop.
    """
    return df[~((df["goals_for"] == 0) & (df["goals_against"] == 0))].copy()


def _flag_preseason(df: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series (aligned to df's index): True for team-game rows
    that are preseason exhibitions, not part of that team's real standings.

    Mechanics: `record_wins + record_losses + record_otl` ("total_games") is
    a cumulative, POST-game counter that is supposed to only increase across
    a team's season (verified: it freezes, doesn't reset, at the regular
    season -> playoffs boundary - a team's final regular-season record stays
    constant through every playoff game). Preseason exhibitions are folded
    into the same file with no explicit flag, but they DO get their own
    (separate, low) win-loss tally that is NOT part of the real season - so
    the real regular season's first game shows total_games dropping back
    down to a small number (1, 2, 3...) after preseason had already
    accumulated a higher one. That drop-then-restart is the signal used
    here: for each (team, season), find the LAST row where total_games
    decreases from the immediately preceding row AND lands at <= 3 (a
    genuine restart-near-zero, not one of the isolated mid-season data
    glitches described in `_drop_unplayed_placeholder_rows` reverting to a
    misleadingly small number - checked directly: with placeholder rows
    already dropped, this exact rule produces exactly one qualifying reset
    for 573 of 590 real team-seasons that have a preseason to trim, zero
    resets for the 162 that don't need trimming (either no preseason rows
    present, or - the 2012-13 lockout season - the real season itself starts
    the "restart" low, correctly matched to January that year), and more
    than one candidate reset for only 17 team-seasons (an ambiguous
    residual, resolved here by taking the LATEST qualifying reset date -
    documented as a best-effort heuristic, not a perfect one).

    Every row strictly before that date, for that team-season, is flagged
    preseason. A game is treated as preseason for pivot purposes if EITHER
    team's row is flagged (see `load_games`).
    """
    d = df.sort_values(["team_id", "season", "date"], kind="stable")
    total_games = d["record_wins"] + d["record_losses"] + d["record_otl"]
    key = d["team_id"].astype(str) + "_" + d["season"].astype(str)   # robust, unambiguous groupby key
    prev_total = total_games.groupby(key).shift(1)
    is_reset = (total_games < prev_total) & (total_games <= PRESEASON_RESET_MAX_TOTAL_GAMES)

    # (team_id, season) -> latest reset date, as a dict lookup (index-safe,
    # unlike a merge which would otherwise need careful row-order bookkeeping)
    reset_dates = d.loc[is_reset].groupby(key[is_reset])["date"].max().to_dict()

    season_start = key.map(reset_dates)
    is_preseason = season_start.notna() & (d["date"] < season_start)
    return is_preseason.reindex(df.index)


def _flag_playoff(df: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series: True for rows at or after a team's regular
    season ended (its record_wins/losses/otl tally stops changing - see
    `_flag_preseason`'s docstring for the freeze behavior this relies on).
    Computed on the ALREADY-preseason-trimmed frame the caller passes in, so
    "first row where total_games stops increasing" unambiguously means the
    regular season -> playoffs boundary, not a preseason artifact.
    """
    d = df.sort_values(["team_id", "season", "date"], kind="stable")
    total_games = d["record_wins"] + d["record_losses"] + d["record_otl"]
    key = d["team_id"].astype(str) + "_" + d["season"].astype(str)   # robust, unambiguous groupby key
    is_season_max = total_games.groupby(key).transform("max") == total_games

    # earliest date each (team, season) reaches its own final regular-season tally -
    # everything strictly after that date, for that team-season, is playoffs
    last_reg_date = d.loc[is_season_max].groupby(key[is_season_max])["date"].min().to_dict()

    last_reg_date_series = key.map(last_reg_date)
    is_playoff = d["date"] > last_reg_date_series
    return is_playoff.reindex(df.index)


def _devig_favorite_moneyline(favorite_decimal: float) -> float:
    """
    Given the favorite's own decimal price, backs out an assumed decimal
    price for the other side using the SAME standard overround this project
    assumes for CFB's -110/-110 spread/total (see config.py's MONEYLINE
    section - a stated assumption, not derived from this dataset, since the
    real other-side price was never captured). Returns np.nan if the
    resulting implied probability for the other side would be outside
    (0, 1) - can happen for extreme/likely-erroneous favorite prices.
    """
    p_fav = odds_math.implied_prob(favorite_decimal)
    p_dog = (1.0 + ASSUMED_ML_OVERROUND) - p_fav
    if p_dog <= 0.0 or p_dog >= 1.0:
        return np.nan
    return 1.0 / p_dog


def load_games() -> pd.DataFrame:
    """Returns a cleaned, chronologically-sorted DataFrame, one row per game."""
    df = pd.read_csv(config.DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    for col in ["goals_for", "goals_against", "record_wins", "record_losses", "record_otl",
                "spread", "over_under", "favorite_moneyline",
                "shots", "power_play_goals", "power_play_opportunities"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["goals_for", "goals_against", "record_wins", "record_losses", "record_otl"]).copy()

    # ---------- Cleaning: placeholder/unplayed rows, non-NHL opponents, preseason ----------
    df = _drop_unplayed_placeholder_rows(df)
    df = df[df["team_id"].isin(config.REAL_TEAM_IDS)].copy()

    df["is_preseason"] = _flag_preseason(df)
    df = df[~df["is_preseason"]].copy()
    df["is_playoff"] = _flag_playoff(df)

    df["franchise_id"] = df["team_id"].apply(config.canonical_team_id)

    # ---------- Pivot: one row per team-game -> one row per game ----------
    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()
    games = home.merge(away, on="game_id", suffixes=("_home", "_away"), how="inner", validate="one_to_one")

    # a game is preseason (already filtered above, per-row) or playoff if EITHER
    # side's row says so - re-derive is_playoff at the game level defensively
    # (should already agree between both sides in the overwhelming majority of
    # games since it's the same real-world game, but OR is the conservative choice)
    games["is_playoff"] = games["is_playoff_home"] | games["is_playoff_away"]

    games["date"] = games["date_home"]
    games["season"] = games["season_home"].astype(int)
    games["home_team_id"] = games["franchise_id_home"]
    games["away_team_id"] = games["franchise_id_away"]
    games["home_team_name"] = games["team_name_home"]
    games["away_team_name"] = games["team_name_away"]
    # alias to the naming convention every other sport module uses
    # (home_franchise/away_franchise) — NHL's own canonical identity is the
    # numeric team_id (already resolved through the Arizona->Utah relocation
    # mapping), kept as-is everywhere internally; this is purely so generic
    # cross-sport code (pipeline.build_pipeline, core/matchup.py) can address
    # any sport the same way without needing to know NHL's team_id was the
    # "real" key all along.
    games["home_franchise"] = games["home_team_id"]
    games["away_franchise"] = games["away_team_id"]
    games["home_team"] = games["home_team_name"]
    games["away_team"] = games["away_team_name"]

    games["home_score"] = games["goals_for_home"]
    games["away_score"] = games["goals_for_away"]
    games["actual_margin"] = games["home_score"] - games["away_score"]
    games["actual_total"] = games["home_score"] + games["away_score"]

    # Raw box-score stats for THIS game (own shots/power-play numbers) - kept
    # here ONLY so features.py can build shift(1)-then-rolled TRAILING
    # features from a team's prior games. These are final box-score numbers,
    # known only once the game is over (verified: shots correlates 0.16 and
    # power_play_goals correlates 0.44 with this same game's goals_for) -
    # NEVER use these columns directly as a same-game feature, same leakage
    # concern as home_score/away_score.
    games["home_shots"] = games["shots_home"]
    games["away_shots"] = games["shots_away"]
    games["home_pp_goals"] = games["power_play_goals_home"]
    games["home_pp_opportunities"] = games["power_play_opportunities_home"]
    games["away_pp_goals"] = games["power_play_goals_away"]
    games["away_pp_opportunities"] = games["power_play_opportunities_away"]

    # true ties (only possible in the pre-shootout era, season 2004 in this
    # data - see config.py) have no winner; drop rather than mislabel as a
    # home loss, same handling MLB uses for its 13 true ties
    games = games[games["actual_margin"] != 0].copy()
    games["home_win"] = (games["actual_margin"] > 0).astype(int)

    # ---------- Odds ----------
    # Spread: verified to already be the home team's line (negative = home
    # favored) - see config.py point 3. Standard assumed -110/-110 juice,
    # same as CFB, since no two-sided spread price exists in this data.
    games["Home Line Close"] = games["spread_home"]
    games["Home Line Odds Close"] = np.where(games["Home Line Close"].notna(), ASSUMED_JUICE_DECIMAL, np.nan)
    games["Away Line Odds Close"] = np.where(games["Home Line Close"].notna(), ASSUMED_JUICE_DECIMAL, np.nan)

    games["Total Score Close"] = games["over_under_home"]
    games["Total Score Over Close"] = np.where(games["Total Score Close"].notna(), ASSUMED_JUICE_DECIMAL, np.nan)
    games["Total Score Under Close"] = np.where(games["Total Score Close"].notna(), ASSUMED_JUICE_DECIMAL, np.nan)

    # Moneyline: side-assignment via spread sign (LOW CONFIDENCE - see
    # config.py's MONEYLINE section for the measured 55.5-81.2% concordance
    # this heuristic actually achieves). Only assigned when spread is
    # non-zero and non-NaN; ties (spread == 0, 4 rows) are left unpriced
    # rather than guessed.
    fm = games["favorite_moneyline_home"]  # identical on both sides, like spread
    spread = games["Home Line Close"]
    home_is_favorite = spread < 0
    away_is_favorite = spread > 0
    has_ml = fm.notna() & (home_is_favorite | away_is_favorite)

    fav_decimal = fm.apply(lambda v: odds_math.american_to_decimal(v) if pd.notna(v) else np.nan)
    dog_decimal = fav_decimal.apply(lambda v: _devig_favorite_moneyline(v) if pd.notna(v) else np.nan)

    games["Home Odds Close"] = np.where(has_ml & home_is_favorite, fav_decimal,
                                 np.where(has_ml & away_is_favorite, dog_decimal, np.nan))
    games["Away Odds Close"] = np.where(has_ml & away_is_favorite, fav_decimal,
                                 np.where(has_ml & home_is_favorite, dog_decimal, np.nan))

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
    games["game_id"] = games.index.astype(str).map(lambda i: f"nhl_{i}")

    keep_cols = [
        "game_id", "source_game_id", "date", "season", "is_playoff",
        "home_team_id", "away_team_id", "home_team_name", "away_team_name",
        "home_franchise", "away_franchise", "home_team", "away_team",
        "home_score", "away_score", "actual_margin", "actual_total", "home_win",
        "home_covers_close", "over_result_close",
        "home_shots", "away_shots", "home_pp_goals", "home_pp_opportunities",
        "away_pp_goals", "away_pp_opportunities",
    ] + ODDS_COLS
    return games[keep_cols]
