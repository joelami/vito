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
        try:
            _team_game_epa_cache = load_team_game_epa()
        except FileNotFoundError as e:
            # Real incident this fixes (2026-09-16): the dataset file this
            # reads hadn't been uploaded to R2 yet, so a Volume without it
            # crashed every live NFL matchup score, not just training (see
            # pipeline.py's _safe_add_trailing_feature() for the training-
            # side fix and its own, fuller comment on the same incident).
            # Neutral 0.0 fallback rather than a real league average, since
            # there's no real data at all to average here.
            print(f"[nflfastr_features] get_current_trailing_epa: dataset not available, "
                  f"returning a neutral fallback: {e}")
            return {col: 0.0 for col in TRAILING_COLS}
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


# Offense col -> the opponent's own trailing col it gets adjusted against
# (an offense's EPA gets normalized by how good the DEFENSE it actually
# faced each game typically is; a defense's allowed-EPA gets normalized
# by how good the OFFENSE it faced typically is). Symmetric pairing, not
# arbitrary -- every "off_*" pairs with the matching "def_*_allowed" and
# vice versa.
_SOS_OPPONENT_COL = {
    "off_epa_per_play": "def_epa_per_play_allowed",
    "off_success_rate": "def_success_rate_allowed",
    "def_epa_per_play_allowed": "off_epa_per_play",
    "def_success_rate_allowed": "off_success_rate",
}


_adjusted_per_game_cache = None


def _build_adjusted_per_game_table(n_games: int) -> pd.DataFrame:
    """
    Shared core of the opponent adjustment, used by both
    build_trailing_epa_features_sos_adjusted() (training, below) and
    get_current_trailing_epa_sos_adjusted() (live scoring) -- computed
    once and cached so live scoring doesn't redo this self-join per call.
    Returns one row per (game_id, franchise) with `{col}_adj` columns:
    this team's raw per-game stat, opponent-adjusted (see
    build_trailing_epa_features_sos_adjusted()'s docstring for the
    method), NOT yet rolled into a trailing window -- that's the caller's
    job, since training rolls per-target-game and live scoring rolls
    relative to "now."
    """
    epa = load_team_game_epa()
    league_avg = {col: float(epa[col].mean()) for col in TRAILING_COLS}

    trailing = epa[["game_id", "franchise", "gameday"] + TRAILING_COLS].sort_values(
        ["franchise", "gameday"], kind="stable")
    trailing_cols_out = [f"{col}_trail" for col in TRAILING_COLS]
    for col, out_col in zip(TRAILING_COLS, trailing_cols_out):
        trailing[out_col] = trailing.groupby("franchise")[col].transform(
            lambda s: s.shift(1).rolling(n_games, min_periods=1).mean()
        )

    # Self-join on game_id to attach each row's OPPONENT's own trailing
    # columns (entering that same game) -- exactly 2 franchises per
    # game_id, so filtering out the self-match leaves exactly one
    # opponent row per original row.
    opp_cols = [f"{c}_trail" for c in TRAILING_COLS]
    self_joined = trailing.merge(
        trailing[["game_id", "franchise"] + opp_cols], on="game_id", suffixes=("", "_opp"))
    self_joined = self_joined[self_joined["franchise"] != self_joined["franchise_opp"]].copy()

    # Per-game opponent-adjusted raw value: this team's raw stat in this
    # one game, minus the opponent's typical (trailing, pre-game) rate on
    # the matching column, plus the league average to re-center.
    for col in TRAILING_COLS:
        opp_trail_col = f"{_SOS_OPPONENT_COL[col]}_trail_opp"
        opp_trail = self_joined[opp_trail_col].fillna(league_avg[_SOS_OPPONENT_COL[col]])
        self_joined[f"{col}_adj"] = self_joined[col] - opp_trail + league_avg[_SOS_OPPONENT_COL[col]]

    return self_joined.sort_values(["franchise", "gameday"], kind="stable")


def build_trailing_epa_features_sos_adjusted(games: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """
    Opponent-adjusted counterpart to build_trailing_epa_features() above --
    real gap that one has: it's schedule-BLIND, unlike Elo (which already
    opponent-adjusts by construction). A team's raw off_epa_per_play
    against three bad defenses in a row looks identical to the same three
    games against three good ones, even though the second is a much
    stronger real signal of offensive quality.

    Method (a standard "opponent-adjust by subtracting the opponent's own
    typical allowed rate, then re-center to the league average" approach,
    the same idea SRS-style ratings use, not invented for this project):
    for each of a team's past games, this looks up what the OPPONENT's
    own trailing (pre-that-game, walk-forward-safe on the opponent's side
    too) allowed-rate was, subtracts it from this team's raw performance
    in that one game, and adds back the league average so the adjustment
    is relative to a LEAGUE-AVERAGE opponent, not zero. THEN rolls those
    per-game adjusted values over this team's own trailing window --
    adjustment happens per-game, before rolling, because a team's last 10
    games each had a DIFFERENT opponent with a different strength at the
    time of THAT specific matchup, not one fixed opponent. (See
    _build_adjusted_per_game_table() above for the shared per-game-
    adjustment step this and the live-scoring counterpart both use.)

    A past game where the opponent itself had no trailing history yet
    (early in the opponent's own timeline) falls back to treating that
    one game as against a league-average opponent (adjustment ~0) rather
    than propagating NaN forward -- same cold-start convention as every
    other trailing feature here.

    Produces the same 8 home_/away_ prefixed columns as
    build_trailing_epa_features(), suffixed `_trail_sos` instead of
    `_trail` so both can coexist on the same feature row. ADOPTED via
    hypothesis test "nfl_nflverse_trailing_epa_sos_adjustment" (see
    decision_log.jsonl and research_nflfastr_epa_sos_adjustment.py):
    total_corr +0.0084 full-sample, confirmed STABLE across a season
    split-half (both halves improved independently, unlike the QB-
    continuity feature tested the same session, which looked good in
    aggregate but failed that exact check).
    """
    global _adjusted_per_game_cache
    epa = load_team_game_epa()
    league_avg = {col: float(epa[col].mean()) for col in TRAILING_COLS}

    if _adjusted_per_game_cache is None:
        _adjusted_per_game_cache = _build_adjusted_per_game_table(n_games)
    self_joined = _adjusted_per_game_cache

    sos_cols_out = [f"{col}_trail_sos" for col in TRAILING_COLS]
    for col, out_col in zip(TRAILING_COLS, sos_cols_out):
        self_joined[out_col] = self_joined.groupby("franchise")[f"{col}_adj"].transform(
            lambda s: s.shift(1).rolling(n_games, min_periods=1).mean()
        )

    out = games.copy()
    for side, franchise_col in (("home", "home_franchise"), ("away", "away_franchise")):
        merged = out[[franchise_col, "date"]].merge(
            self_joined[["franchise", "gameday"] + sos_cols_out],
            left_on=[franchise_col, "date"], right_on=["franchise", "gameday"], how="left",
        )
        for col, out_col in zip(TRAILING_COLS, sos_cols_out):
            out[f"{side}_{out_col}"] = merged[out_col].fillna(league_avg[col]).values
    return out


def get_current_trailing_epa_sos_adjusted(franchise: str, n_games: int = 10) -> dict:
    """
    Live-scoring counterpart to build_trailing_epa_features_sos_adjusted(),
    same real gap/convention as get_current_trailing_epa() -- see that
    function's docstring. Uses the same cached, already-opponent-adjusted
    per-game table (_build_adjusted_per_game_table()) that training uses,
    just takes this one franchise's most recent `n_games` ADJUSTED values
    directly (no re-rolling needed -- "current form" IS the mean of the
    last N real, already-adjusted games, the same relationship
    get_current_trailing_epa() has to build_trailing_epa_features()).
    """
    global _adjusted_per_game_cache
    try:
        if _adjusted_per_game_cache is None:
            _adjusted_per_game_cache = _build_adjusted_per_game_table(n_games)
        self_joined = _adjusted_per_game_cache
        epa = load_team_game_epa()
        league_avg_adj = {col: float(epa[col].mean()) for col in TRAILING_COLS}  # re-centered adjustment -> same mean as raw
    except FileNotFoundError as e:
        # Same real incident/fix as get_current_trailing_epa() above.
        print(f"[nflfastr_features] get_current_trailing_epa_sos_adjusted: dataset not available, "
              f"returning a neutral fallback: {e}")
        return {f"{col}_trail_sos": 0.0 for col in TRAILING_COLS}

    team_rows = self_joined[self_joined["franchise"] == franchise].sort_values("gameday", kind="stable")
    if team_rows.empty:
        return {f"{col}_trail_sos": league_avg_adj[col] for col in TRAILING_COLS}
    recent = team_rows.tail(n_games)
    return {f"{col}_trail_sos": float(recent[f"{col}_adj"].mean()) for col in TRAILING_COLS}


def reset_caches() -> None:
    """
    Real bug found and fixed 2026-09-15 (app owner: "the data looks stale
    to me... I'm seeing this issue across all leagues"): _team_game_epa_cache
    and _adjusted_per_game_cache both populate ONCE, on first use, and
    never invalidate themselves -- load_team_game_epa() itself always
    re-reads the CSV fresh (so TRAINING via build_trailing_epa_features()
    was never actually affected), but get_current_trailing_epa() and BOTH
    build_trailing_epa_features_sos_adjusted() (training) and
    get_current_trailing_epa_sos_adjusted() (live) all read through these
    frozen caches. In a long-lived Railway process (no restart between
    scheduler runs), this meant: core/dataset_refresh.py correctly pulls
    fresh nflverse data onto disk every day, but the SOS-adjusted trailing
    feature -- and live scoring's raw EPA lookup -- would keep silently
    reusing whatever was in memory from the FIRST pipeline build after
    boot, for the rest of that deployment's uptime, no matter how many
    days of real new games the CSV on disk gained. Call this once per
    scheduled full run, before rebuilding pipelines, so every day's build
    starts from a clean slate -- see scheduler.py's _run_full().
    """
    global _team_game_epa_cache, _adjusted_per_game_cache
    _team_game_epa_cache = None
    _adjusted_per_game_cache = None
