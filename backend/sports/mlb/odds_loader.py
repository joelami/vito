"""
Loads the real MLB odds archive and joins it onto sports/mlb/loader.py's
game-level output, producing the same "Home Odds Close" / "Home Line Close"
/ "Total Score Close" etc. column names core.edge_finder and core.backtest
expect (see sports/cfb/loader.py for the same naming convention - CFB uses
these exact column names too, for the same reason: unchanged core/ code).

UPDATE (superseding loader.py's original "no odds data exists anywhere in
this source" framing): a real odds archive was found after this module's
first pass, at Datasets/MLB/archive/. It does NOT cover the full 1990-2025
Retrosheet range - only 2012-2021 - so most of this module's history is
still odds-free, but a genuine, walk-forward-honest backtest is now possible
over the covered seasons. See sports/mlb/verify.py for the actual numbers.

Both files in Datasets/MLB/archive/ were read and compared directly before
picking one, not assumed from their names:

  - oddsDataMLB.csv: 45,530 rows, one row per TEAM-PERSPECTIVE per game (2
    mirrored rows per game - one from each team's side), 2012-03-28 through
    2021-10-03. Real American-odds moneyline, run line (MLB's spread
    equivalent - see below), and total, with GENUINE TWO-SIDED PRICES for
    all three markets (moneyLine/oppMoneyLine, runLine+runLineOdds /
    oppRunLine+oppRunLineOdds, total/overOdds/underOdds). This is a
    meaningfully better market than CFB's: CFB only had a real two-sided
    moneyline and had to assume standard -110 juice for spread/total, MLB's
    archive has real juice on all three. moneyLine/total/overOdds/underOdds
    have ZERO missing values across all 45,530 rows. runLine (the number)
    is missing 0.96% of rows; runLineOdds/oppRunLineOdds (the run-line's own
    two-sided price) is missing far more often - 21.4% of rows - so the
    run-line MARKET is only bettable on ~78.6% of rows. Missing
    runLineOdds is left as NaN, never assumed/fabricated - a game with a
    runLine number but no runLineOdds simply produces no spread opportunity
    for that game (core.edge_finder's existing None-price gate handles this
    correctly, no changes needed there).
  - runLine is essentially always exactly +/-1.5 (45,092 of 45,530 non-null
    rows, 99.0%) - unlike a football spread, MLB's run line is a
    near-universally FIXED proposition (favorite lays 1.5, underdog gets
    1.5); the market's actual pricing signal lives almost entirely in the
    two-sided juice (runLineOdds/oppRunLineOdds), not in a moving line
    number. Confirmed directly against this file, not assumed. Sign
    convention verified against the project's "Home Line" standard
    (negative = home favored): regressing home-perspective runLine's sign
    against the real two-sided moneyline favorite agrees 98.2% of the time
    - runLine is used as-is, no sign flip, same validation approach CFB used
    for its spread column.
  - oddsData.csv: checked and confirmed to be the SAME underlying odds
    records, not independent data - identical row count (45,530), identical
    date range, identical missingness pattern in runLine/runLineOdds,
    ~97-98% value agreement with oddsDataMLB.csv on the fields both files
    carry. Its only content not in oddsDataMLB.csv is an explicit
    doubleheader `gameNumber` (1/2) and home/away `at` (H/V) flag per row.
    NOT used here - this loader instead disambiguates the doubleheader case
    by matching odds rows to Retrosheet games on the actual final score
    (`runs`/`oppRuns` here vs. `home_score`/`away_score` in loader.py) - a
    more robust key since it's real, game-specific data rather than an
    assumption that both files list a team's games in the same row order.
  - Team codes here use a different abbreviation set than Retrosheet's
    (e.g. "CHC" vs Retrosheet's "CHN", "KC" vs "KCA", "LAA" vs "ANA", "WSH"
    vs "WAS") - ODDS_TEAM_TO_RETRO translates every code to its Retrosheet
    equivalent, then reuses `config.canonical_team` (the exact same
    function loader.py applies to Retrosheet codes) so no separate/parallel
    franchise mapping exists.
  - This is a SINGLE odds snapshot per game - no timestamp, no open-vs-close
    tag anywhere in either file - exactly the situation CFB's odds data was
    in. Every price is stored under the "Close" suffix purely so
    core/edge_finder.py's and core/backtest.py's column lookups work
    unchanged; there is no real open-vs-close distinction, so Closing Line
    Value (CLV) computed against these prices is trivially ~0 for every bet
    (both "price taken" and "closing price" resolve to the same recorded
    number). Flagged in verify.py's output exactly like CFB flags the same
    limitation - this backtest can speak to hit-rate/ROI against one real
    assumed-available price, not genuine CLV.
"""

import numpy as np
import pandas as pd

from . import config
from core import odds_math


ODDS_DATA_PATH = config.DATA_DIR / "archive" / "oddsDataMLB.csv"

ODDS_TEAM_TO_RETRO = {
    "CHC": "CHN", "CWS": "CHA", "KC": "KCA", "LAA": "ANA", "LAD": "LAN",
    "NYM": "NYN", "NYY": "NYA", "SD": "SDN", "SF": "SFN", "STL": "SLN",
    "TB": "TBA", "WSH": "WAS",
    # every other code observed in the file already matches its Retrosheet
    # code exactly (ARI, ATL, BAL, BOS, CIN, CLE, COL, DET, HOU, MIA, MIL,
    # MIN, OAK, PHI, PIT, SEA, TEX, TOR)
}


def _to_canonical(code: str) -> str:
    return config.canonical_team(ODDS_TEAM_TO_RETRO.get(code, code))


def _american_series_to_decimal(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: odds_math.american_to_decimal(float(v)) if pd.notna(v) else np.nan)


def load_odds() -> pd.DataFrame:
    """
    Returns the raw odds file, one row per team-perspective-per-game, with
    team/opponent translated to canonical franchise codes and every
    American price converted to decimal. Not yet game-level/joined - see
    `attach_odds`.
    """
    df = pd.read_csv(ODDS_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["team_franchise"] = df["team"].apply(_to_canonical)
    df["opponent_franchise"] = df["opponent"].apply(_to_canonical)

    for col in ["moneyLine", "oppMoneyLine", "runLineOdds", "oppRunLineOdds", "overOdds", "underOdds"]:
        df[f"{col}_dec"] = _american_series_to_decimal(df[col])

    return df.drop(columns=["season"])  # loader.py's own `season` is the source of truth after the join


def attach_odds(games: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins real odds onto `games` (sports/mlb/loader.py's output).
    Match key: (date, team_franchise == home_franchise, opponent_franchise
    == away_franchise, runs == home_score, oppRuns == away_score) - matching
    on the actual final score is what disambiguates doubleheaders (see
    module docstring); for the ~98.6% of games that aren't a doubleheader
    it's simply confirmation the right game was found.

    Games outside the odds file's 2012-2021 coverage, or with no matching
    row, get NaN in every odds column - exactly like CFB's sparse-coverage
    games, nothing here filters them out. The one real edge case found in
    this data (a single 2016-06-07 PIT/NYN doubleheader where BOTH games
    happened to end 3-1 - the score-match key can't tell them apart) is
    detected and left unmatched (NaN) rather than guessing, rather than
    risk silently assigning game 2's odds to game 1's result or vice versa.
    """
    odds = load_odds()

    merged = games.merge(
        odds,
        left_on=["date", "home_franchise", "away_franchise", "home_score", "away_score"],
        right_on=["date", "team_franchise", "opponent_franchise", "runs", "oppRuns"],
        how="left",
        suffixes=("", "_odds"),
    )

    dupe_counts = merged.groupby("game_id")["team"].transform("size")
    ambiguous = dupe_counts > 1
    n_ambiguous_games = merged.loc[ambiguous, "game_id"].nunique()
    if n_ambiguous_games:
        merged.loc[ambiguous, [c for c in odds.columns if c not in ("date",)]] = np.nan

    merged = merged.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)
    assert len(merged) == len(games), "attach_odds must return exactly one row per input game"

    merged["Home Odds Close"] = merged["moneyLine_dec"]
    merged["Away Odds Close"] = merged["oppMoneyLine_dec"]
    merged["Home Line Close"] = merged["runLine"]
    merged["Home Line Odds Close"] = merged["runLineOdds_dec"]
    merged["Away Line Odds Close"] = merged["oppRunLineOdds_dec"]
    merged["Total Score Close"] = merged["total"]
    merged["Total Score Over Close"] = merged["overOdds_dec"]
    merged["Total Score Under Close"] = merged["underOdds_dec"]

    # ATS / over-result columns, same convention as NFL/CFB loaders (used
    # only for reporting here - features.py doesn't currently build a
    # rolling ATS feature for MLB, see features.py's docstring).
    close_line = merged["Home Line Close"]
    merged["home_covers_close"] = np.select(
        [merged["actual_margin"] > -close_line, merged["actual_margin"] == -close_line],
        [1, 0], default=-1,
    )
    merged.loc[close_line.isna(), "home_covers_close"] = np.nan

    total_line = merged["Total Score Close"]
    merged["over_result_close"] = np.select(
        [merged["actual_total"] > total_line, merged["actual_total"] == total_line],
        [1, 0], default=-1,
    )
    merged.loc[total_line.isna(), "over_result_close"] = np.nan

    keep_extra = [
        "Home Odds Close", "Away Odds Close", "Home Line Close",
        "Home Line Odds Close", "Away Line Odds Close",
        "Total Score Close", "Total Score Over Close", "Total Score Under Close",
        "home_covers_close", "over_result_close",
    ]
    return pd.concat([games.reset_index(drop=True), merged[keep_extra]], axis=1)
