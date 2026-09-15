"""
Real per-team trailing EPA/success-rate features built from nflverse's
free, MIT-licensed play-by-play data (see scripts/fetch_nflfastr_team_game_
epa.py for how Datasets/NFL/nflfastr_team_game_epa.csv gets built -- one
row per (game, team), NOT raw play-by-play).

Why this exists: NFL's live ML_FEATURE_COLS (sports/nfl/features.py) has
zero features carrying real play-level efficiency signal -- team_pf_l10/
team_pa_l10 are just points scored/allowed, which conflate offense,
defense, special teams, and pure luck (a defense that forces 3 turnovers
returned for touchdowns "allows" fewer points without being a better
defense that day). EPA/play and success rate are the modern analytics-
community standard for separating real per-play efficiency from that
noise -- see core/factor_taxonomy.py's REGISTRY for where this sits
relative to the existing feature set.

TRAILING, walk-forward-safe by construction: exactly the same "last N
games, shift-then-roll, never includes the game being predicted" pattern
every other trailing feature in this project already uses (see e.g.
sports/nfl/features.py's own pf_l10/pa_l10) -- this module's whole job is
producing home_/away_ prefixed trailing columns in that same shape, so it
slots into ML_FEATURE_COLS the same way any other feature already does.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent.parent.parent / "Datasets" / "NFL" / "nflfastr_team_game_epa.csv"

# nflverse's team abbreviations -> this project's own stable franchise
# names (sports/nfl/loader.py's home_franchise/away_franchise, which
# already collapse relocations -- LA/STL both "Rams", LV/OAK both
# "Raiders" -- onto one identity; nflverse's own PBP data turned out to
# already be normalized to the CURRENT abbreviation for every season
# 2006-2025, confirmed directly: exactly 32 distinct codes across the
# whole pull, not 34+ with historical variants, so this is a flat 1:1 map,
# no season-conditional logic needed).
TEAM_CODE_TO_FRANCHISE = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders",
    "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans",
    "WAS": "Washington",
}

TRAILING_COLS = ["off_epa_per_play", "off_success_rate", "def_epa_per_play_allowed", "def_success_rate_allowed"]


_team_game_epa_cache = None


def get_current_trailing_epa(franchise: str, n_games: int = 10) -> dict:
    """
    Real gap this closes: sports/nfl/matchup.py's build_matchup_feature_row()
    builds an EXPLICIT, hardcoded feature dict for scoring a live/upcoming
    game (not a dynamic pass over ML_FEATURE_COLS) -- every other feature
    already has its own "current form" lookup wired in there (see
    pf_l10/ats_pct_l10 via pipeline["current_form"]); this is that same
    lookup for the EPA trailing columns, real and required, not optional --
    skipping it would silently break every live NFL prediction the moment
    ML_FEATURE_COLS includes a column this function's output doesn't have
    (scikit-learn's predict() requires the exact training columns).

    Returns league-average fallback (this data's own real mean) for a
    franchise with no rows at all -- same convention as every other
    trailing feature's cold-start handling in this project.

    Honest, documented limitation: this reads the LOCAL, already-fetched
    snapshot (Datasets/NFL/nflfastr_team_game_epa.csv) -- real for every
    game already in it, but only as fresh as the last time scripts/
    fetch_nflfastr_team_game_epa.py ran. nflverse updates nightly during
    the season; this needs the same periodic-refresh treatment
    core/live_results.py gives ESPN-synced ratings data to stay current
    week-to-week in-season -- not yet wired to the harness cycle, flagged
    honestly rather than silently pretending this is live.
    """
    global _team_game_epa_cache
    if _team_game_epa_cache is None:
        _team_game_epa_cache = load_team_game_epa()
    epa = _team_game_epa_cache
    league_avg = {col: float(epa[col].mean()) for col in TRAILING_COLS}

    team_rows = epa[epa["franchise"] == franchise].sort_values("gameday", kind="stable")
    if team_rows.empty:
        return league_avg
    recent = team_rows.tail(n_games)
    return {col: float(recent[col].mean()) for col in TRAILING_COLS}


def load_team_game_epa() -> pd.DataFrame:
    """One row per (game, team) with real franchise names attached --
    raises a clear error rather than silently no-opping if the data
    hasn't been fetched yet (see scripts/fetch_nflfastr_team_game_epa.py),
    since a feature silently reducing to all-NaN is a real bug class this
    project has hit before (see core/live_results.py's docstring)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found -- run `python3 scripts/fetch_nflfastr_team_game_epa.py` "
            f"from backend/ first (downloads real nflverse play-by-play, ~20 seasons)."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["gameday"])
    df["franchise"] = df["team"].map(TEAM_CODE_TO_FRANCHISE)
    unmapped = df[df["franchise"].isna()]["team"].unique()
    if len(unmapped):
        raise ValueError(f"nflverse team code(s) with no franchise mapping: {list(unmapped)} -- "
                          f"update TEAM_CODE_TO_FRANCHISE above before trusting this data.")
    return df


def build_trailing_epa_features(games: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """
    Returns `games` with 8 new columns added: home_/away_ prefixed
    trailing (prior `n_games`, shift-then-roll) off_epa_per_play,
    off_success_rate, def_epa_per_play_allowed, def_success_rate_allowed.
    League-average fallback (this data's own real mean, not a guessed
    0.0) for any team-game with fewer than `n_games` of real EPA history
    yet -- same cold-start convention as every other trailing feature.

    `games` must carry `home_franchise`/`away_franchise`/`date`/`game_id`
    (sports/nfl/loader.py's own shape) -- this only ever ADDS columns,
    never drops/reorders existing rows, so it's safe to call right before
    build_features() the same way MLB's attach_starter_quality() already
    does (see pipeline.py's "research-adopted features" dispatch comment).
    """
    epa = load_team_game_epa()
    league_avg = {col: float(epa[col].mean()) for col in TRAILING_COLS}

    # Long format: one row per (franchise, date, ...trailing cols...),
    # sorted chronologically per team so shift(1).rolling(n) never looks
    # at the game it's supposed to be predicting -- identical discipline
    # to sports/nfl/features.py's own pf_l10/pa_l10 construction.
    # groupby(...).transform() (per-column, Series in/Series out) rather
    # than .apply() -- sidesteps a real pandas-version incompatibility
    # (.apply()'s include_groups kwarg doesn't exist on the pandas version
    # this project runs) and is the more direct tool for a same-shape
    # per-group transform anyway.
    trailing = epa[["franchise", "gameday"] + TRAILING_COLS].sort_values(["franchise", "gameday"], kind="stable")
    trailing_cols_out = [f"{col}_trail" for col in TRAILING_COLS]
    for col, out_col in zip(TRAILING_COLS, trailing_cols_out):
        trailing[out_col] = trailing.groupby("franchise")[col].transform(
            lambda s: s.shift(1).rolling(n_games, min_periods=1).mean()
        )

    out = games.copy()
    for side, franchise_col in (("home", "home_franchise"), ("away", "away_franchise")):
        merged = out[[franchise_col, "date"]].merge(
            trailing, left_on=[franchise_col, "date"], right_on=["franchise", "gameday"], how="left",
        )
        for col, out_col in zip(TRAILING_COLS, trailing_cols_out):
            out[f"{side}_{col}_trail"] = merged[out_col].fillna(league_avg[col]).values
    return out
