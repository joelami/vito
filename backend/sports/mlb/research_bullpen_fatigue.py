"""
SECOND hypothesis test in the same disciplined research loop as
sports/mlb/starting_pitcher.py (see decision_log.jsonl for the first:
starting_pitcher_rolling_er_per_start, adopt_cautiously).

HYPOTHESIS: bullpen fatigue - a team's bullpen workload in the days
immediately preceding a game is a real, currently-missing signal, distinct
from the starter-quality feature already adopted (which only captures the
starter's OWN recent performance, not the relief corps backing him up on
a given day). Well-documented in sabermetric/team bullpen-management
practice: teams routinely cap most relievers at ~2-3 consecutive days of
work specifically because command/velocity/performance degrades with
back-to-back-day usage - a "gassed" bullpen is a real, qualitatively
different signal from starter quality, and today's feature set has nothing
that touches it.

SCOPING, stated explicitly (same honest-proxy discipline as the starter
feature): this does NOT identify which specific reliever will pitch today
(unknowable pre-game - lineup/bullpen decisions aren't in this data) or
track innings/pitches thrown (would need out-by-out tracking, out of scope
here same as it was for the starter feature). What IS computed: a TEAM-
level daily count of RELIEF-PITCHER APPEARANCES (games pitched in relief,
not innings), summed over the trailing 3 calendar days strictly before this
game, walk-forward-safe - the shift(1)-before-rolling convention every
other feature in this project follows, generalized here to calendar-day
granularity (rolling over days) instead of per-start-count granularity
(rolling over starts, as the starter feature does). Named `bullpen_apps_l3d`
throughout - never "workload," "innings," or "fatigue score," since none of
those would be true of what's actually measured (an appearance COUNT).

Source: same Retrosheet event files as starting_pitcher.py
(Datasets/MLB/2010seve, 2020seve). Newly parsed here: `sub` lines with
fieldpos==1 identify a RELIEF pitcher entering the game (same field-
position convention `start` lines use for the starter). Verified directly
against 2020-07-28 SEA@ANA (the same game starting_pitcher.py's docstring
verifies): 7 `sub,...,fieldpos=1` lines (cortn001, grotz001 for Seattle;
ramin002, buchr001, mayem001, barnj002, middk001 for Anaheim), each with a
matching `data,er` entry alongside the two starters' - confirms these are
real relief-pitcher appearances, not some other substitution type.
"""

import csv
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings("ignore", category=FutureWarning)

from . import config

EVENT_SUBDIRS = ["2010seve", "2020seve"]
FATIGUE_WINDOW_DAYS = 3  # trailing calendar days - standard "back-to-back-plus" bullpen-management heuristic


def _iter_event_files():
    for sub in EVENT_SUBDIRS:
        for ext in ("EVA", "EVN"):
            for f in sorted(glob.glob(str(config.DATA_DIR / sub / f"*.{ext}"))):
                yield f


def _parse_file(path):
    """
    Returns a list of dicts, one per game: date, game_number, raw team
    codes, and the list of (team_side, pitcher_id, is_starter) appearances
    for that game - starters from `start` lines (fieldpos==1) plus
    relievers from `sub` lines (fieldpos==1).
    """
    games = []
    cur = None

    with open(path, encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            tag = row[0]

            if tag == "id":
                if cur is not None:
                    games.append(cur)
                cur = {"appearances": []}
            elif cur is None:
                continue
            elif tag == "info":
                key = row[1]
                val = row[2] if len(row) > 2 else ""
                if key in ("visteam", "hometeam", "date", "number"):
                    cur[key] = val
            elif tag == "start":
                player_id, home_flag, fieldpos = row[1], row[3], row[5]
                if fieldpos == "1":
                    side = "home" if home_flag == "1" else "away"
                    cur["appearances"].append((side, player_id, True))
            elif tag == "sub":
                player_id, home_flag, fieldpos = row[1], row[3], row[5]
                if fieldpos == "1":
                    side = "home" if home_flag == "1" else "away"
                    cur["appearances"].append((side, player_id, False))

    if cur is not None:
        games.append(cur)
    return games


def load_pitcher_appearances() -> pd.DataFrame:
    """
    Long format: one row per (team, date, pitcher, is_starter) appearance,
    across every parsed game. A pitcher who enters twice in one game (very
    rare - e.g. an emergency return) would produce two rows on the same
    date, which is fine for a raw appearance COUNT (each row = one relief
    stint that day).
    """
    rows = []
    for f in _iter_event_files():
        for g in _parse_file(f):
            date, number = g.get("date"), g.get("number", "0")
            home_raw, away_raw = g.get("hometeam"), g.get("visteam")
            if date is None or home_raw is None or away_raw is None:
                continue
            for side, pitcher_id, is_starter in g["appearances"]:
                team_raw = home_raw if side == "home" else away_raw
                rows.append({
                    "date": date, "game_number": number,
                    "team_raw": team_raw, "pitcher_id": pitcher_id, "is_starter": is_starter,
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["team_franchise"] = df["team_raw"].apply(config.canonical_team)
    return df


def build_bullpen_fatigue(appearances: pd.DataFrame, window_days: int = FATIGUE_WINDOW_DAYS) -> pd.DataFrame:
    """
    Team-level daily reliever-appearance count, rolled forward walk-forward
    -safe over the trailing `window_days` calendar days. Returns one row per
    (team_franchise, date) actually present in the source data, with
    `bullpen_apps_lNd` = sum of relief appearances by this team over the
    `window_days` days strictly BEFORE this date (today's own appearances,
    which haven't happened yet relative to a pre-game prediction, are never
    included).
    """
    relief = appearances[~appearances["is_starter"]]
    daily = relief.groupby(["team_franchise", "date"]).size().rename("relief_apps").reset_index()

    out_frames = []
    for team, grp in daily.groupby("team_franchise"):
        grp = grp.set_index("date").sort_index()
        full_range = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        s = grp["relief_apps"].reindex(full_range, fill_value=0)
        # shift(1) first (exclude today) then roll over the trailing window - the same
        # discipline every other rolling feature in this project follows, generalized
        # from "per prior start" granularity to "per prior calendar day" granularity.
        rolled = s.shift(1).rolling(window_days, min_periods=1).sum()
        frame = rolled.rename("bullpen_apps_lNd").reset_index().rename(columns={"index": "date"})
        frame["team_franchise"] = team
        out_frames.append(frame)

    return pd.concat(out_frames, ignore_index=True)


def attach_bullpen_fatigue(games: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins `home_bullpen_apps_l3d` / `away_bullpen_apps_l3d` onto
    `games`. Games outside event-file coverage (~2010-2025), or a team's
    first-ever appearance in the window (no trailing days to sum), get 0 -
    a defensible fallback since "no games logged yet" and "genuinely zero
    relief appearances in the last 3 days" are not distinguishable from a
    pre-game standpoint, and 0 is the correct floor value either way (never
    negative, and a team with no bullpen data isn't fatigued by definition).
    """
    appearances = load_pitcher_appearances()
    fatigue = build_bullpen_fatigue(appearances)

    home_f = fatigue.rename(columns={"team_franchise": "home_franchise", "bullpen_apps_lNd": "home_bullpen_apps_l3d"})
    away_f = fatigue.rename(columns={"team_franchise": "away_franchise", "bullpen_apps_lNd": "away_bullpen_apps_l3d"})

    merged = games.merge(home_f, on=["date", "home_franchise"], how="left")
    merged = merged.merge(away_f, on=["date", "away_franchise"], how="left")
    merged["home_bullpen_apps_l3d"] = merged["home_bullpen_apps_l3d"].fillna(0.0)
    merged["away_bullpen_apps_l3d"] = merged["away_bullpen_apps_l3d"].fillna(0.0)
    return merged


def main():
    apps = load_pitcher_appearances()
    n_games_files = apps[["date", "game_number", "team_raw"]].drop_duplicates().shape[0]
    print(f"[research_bullpen_fatigue] {len(apps):,} pitcher-appearance rows parsed "
          f"({apps['is_starter'].sum():,} starts, {(~apps['is_starter']).sum():,} relief appearances) "
          f"across {n_games_files:,} team-game rows.")

    from sports.mlb.loader import load_games
    from sports.mlb.odds_loader import attach_odds

    games = load_games()
    games = attach_odds(games)
    merged = attach_bullpen_fatigue(games)

    has_odds = merged["Home Odds Close"].notna() & merged["Away Odds Close"].notna()
    n_window = int(has_odds.sum())
    # "real" here = at least one side of the join found a matching team/date row in the
    # event-file data at all (vs. the 0-fallback for pure out-of-coverage games)
    matched = merged.merge(
        apps[["date", "team_franchise"]].drop_duplicates().assign(_matched=True),
        left_on=["date", "home_franchise"], right_on=["date", "team_franchise"], how="left"
    )["_matched"].fillna(False)
    n_window_matched = int((has_odds & matched.values).sum())
    print(f"\nBacktest-eligible games (real odds, 2012-2021): {n_window:,}")
    print(f"...home team matched to real event-file appearance data: {n_window_matched:,} "
          f"({n_window_matched / n_window * 100:.1f}%)")
    print(f"\nbullpen_apps_l3d distribution:\n{merged[['home_bullpen_apps_l3d','away_bullpen_apps_l3d']].describe()}")


if __name__ == "__main__":
    main()
