"""
Loads and cleans Datasets/NFL/nfl.xlsx into a tidy, one-row-per-game
DataFrame ready for feature engineering. No modeling happens here — this
module's only job is turning the raw spreadsheet into trustworthy columns.
"""

import pandas as pd
import numpy as np

from . import config


ODDS_COLS = [
    "Home Odds Open", "Home Odds Min", "Home Odds Max", "Home Odds Close",
    "Away Odds Open", "Away Odds Min", "Away Odds Max", "Away Odds Close",
    "Home Line Open", "Home Line Min", "Home Line Max", "Home Line Close",
    "Away Line Open", "Away Line Min", "Away Line Max", "Away Line Close",
    "Home Line Odds Open", "Home Line Odds Min", "Home Line Odds Max", "Home Line Odds Close",
    "Away Line Odds Open", "Away Line Odds Min", "Away Line Odds Max", "Away Line Odds Close",
    "Total Score Open", "Total Score Min", "Total Score Max", "Total Score Close",
    "Total Score Over Open", "Total Score Over Min", "Total Score Over Max", "Total Score Over Close",
    "Total Score Under Open", "Total Score Under Min", "Total Score Under Max", "Total Score Under Close",
]


def _yn_to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().eq("Y")


def load_games() -> pd.DataFrame:
    """Returns a cleaned, chronologically-sorted DataFrame, one row per game."""
    df = pd.read_excel(config.DATA_PATH, sheet_name="Data")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Home Team", "Away Team"]).copy()

    for col in ODDS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["overtime"] = _yn_to_bool(df["Overtime?"])
    df["is_playoff"] = _yn_to_bool(df["Playoff Game?"])
    df["is_neutral_venue"] = _yn_to_bool(df["Neutral Venue?"])

    df["home_score"] = pd.to_numeric(df["Home Score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["Away Score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    df["home_franchise"] = df["Home Team"].apply(config.canonical_franchise)
    df["away_franchise"] = df["Away Team"].apply(config.canonical_franchise)
    df["is_divisional"] = df.apply(
        lambda r: config.is_divisional_game(r["home_franchise"], r["away_franchise"]), axis=1
    )
    df["season"] = df["Date"].apply(config.season_for_date)
    df["book_source"] = df["Date"].apply(config.book_source_for_date)

    df["actual_margin"] = df["home_score"] - df["away_score"]        # home perspective, positive = home won by
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["actual_margin"] > 0).astype(int)
    # ties count as neither team covering a moneyline; extremely rare post-2016 (OT rules)

    # ATS result vs the CLOSING home line (line stored as e.g. -3.5 = home favored by 3.5).
    # Home covers if actual_margin > -line; push if exactly equal.
    close_line = df["Home Line Close"]
    df["home_covers_close"] = np.select(
        [df["actual_margin"] > -close_line, df["actual_margin"] == -close_line],
        [1, 0],
        default=-1,  # -1 = home did not cover; 0 = push; 1 = covered. NaN line -> NaN handled below
    )
    df.loc[close_line.isna(), "home_covers_close"] = np.nan

    total_line = df["Total Score Close"]
    df["over_result_close"] = np.select(
        [df["actual_total"] > total_line, df["actual_total"] == total_line],
        [1, 0],
        default=-1,
    )
    df.loc[total_line.isna(), "over_result_close"] = np.nan

    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["game_id"] = df.index.astype(str).map(lambda i: f"nfl_{i}")

    keep_cols = [
        "game_id", "Date", "season", "is_playoff", "is_neutral_venue", "overtime", "is_divisional",
        "book_source",
        "Home Team", "Away Team", "home_franchise", "away_franchise",
        "home_score", "away_score", "actual_margin", "actual_total", "home_win",
        "home_covers_close", "over_result_close",
    ] + ODDS_COLS
    return df[[c for c in keep_cols if c in df.columns]].rename(columns={"Date": "date"})
