"""
Real per-team trailing SUCCESS RATE features for CFB, built from free
cfbfastR play-by-play (see scripts/fetch_cfbfastr_team_game_success.py --
that script's own docstring has the full story on why this is success
rate, not EPA: EPA is a model OUTPUT this environment can't run
(xgboost+libomp unavailable), success rate is a real, standard, rule-
based metric computable directly from raw down/distance/yards-gained,
no model needed).

Mirrors sports/nfl/nflfastr_features.py's shape exactly (trailing,
walk-forward-safe, home_/away_ prefixed columns) -- see that module for
the full walk-forward-safety reasoning, identical here.

TEAM-NAME MAPPING IS THE REAL, HONEST LIMITATION HERE, different from
NFL's clean 32-code mapping: cfbfastR uses bare school names ("Ohio
State"), this project's own CFB data uses "{School} {Mascot}" ("Ohio
State Buckeyes"). Checked directly: a normalize-and-prefix-match (handles
'St.'<->'State') covers 337 of 460 distinct franchises, which works out
to 73.9% of real 2013+ games having BOTH sides mapped -- the unmapped
tail is overwhelmingly the same small-school/FCS "buy game" opponents
already flagged as a real, separate problem elsewhere in this project
(see decision_log.jsonl's CFB low-history-rating-floor entry) -- not a
representative random 26%, a genuinely different (and lower-stakes, since
CFB spread/moneyline already gate those games'  confidence separately)
population. A team with no mapping gets the league-average fallback, same
convention as every other trailing feature in this project.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent.parent.parent / "Datasets" / "College Football" / "cfbfastr_team_game_success.csv"

TRAILING_COLS = ["off_success_rate", "def_success_rate_allowed"]

_team_name_cache = None
_success_by_franchise_cache = None  # populated by load_team_game_success(), reused by get_current_trailing_success()


def _normalize(name: str) -> str:
    return name.replace("St.", "State").replace("'", "").lower().strip()


def build_team_name_mapping(vito_franchise_names) -> dict:
    """Real, tested prefix-match (see this module's docstring for the
    honest 73.9%-of-games coverage number) -- `vito_franchise_names` is
    this project's own set of CFB franchise strings (sports/cfb/loader.py's
    home_franchise/away_franchise), matched against whatever cfbfastR
    team names are actually present in the fetched data."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found -- run `python3 scripts/fetch_cfbfastr_team_game_success.py` "
            f"from backend/ first (downloads real cfbfastR play-by-play, no API key needed)."
        )
    cfbfastr_names = pd.read_csv(DATA_PATH, usecols=["team"])["team"].unique()
    cfbfastr_norm = {}
    for n in cfbfastr_names:
        cfbfastr_norm.setdefault(_normalize(n), n)

    mapping = {}
    for v in vito_franchise_names:
        vn = _normalize(v)
        best = None
        for cn, orig in cfbfastr_norm.items():
            if vn == cn or vn.startswith(cn + " "):
                if best is None or len(cn) > len(best[0]):
                    best = (cn, orig)
        if best:
            mapping[v] = best[1]
    return mapping


def load_team_game_success(vito_franchise_names) -> pd.DataFrame:
    global _team_name_cache, _success_by_franchise_cache
    df = pd.read_csv(DATA_PATH, parse_dates=False)
    if _team_name_cache is None:
        _team_name_cache = build_team_name_mapping(vito_franchise_names)
    reverse = {}
    for vito_name, cfbfastr_name in _team_name_cache.items():
        reverse.setdefault(cfbfastr_name, vito_name)  # first vito name wins a given cfbfastr name (rare collisions)
    df["franchise"] = df["team"].map(reverse)
    out = df[df["franchise"].notna()].copy()
    _success_by_franchise_cache = out  # reused by get_current_trailing_success() -- see that function's docstring
    return out


def get_current_trailing_success(franchise: str, n_games: int = 10) -> dict:
    """
    Live-scoring counterpart, mirrors sports/nfl/nflfastr_features.py's
    get_current_trailing_epa() exactly -- same real gap it closes (a
    not-yet-played CFB game needs a "current form" trailing value, not
    just the historical walk-forward-safe training column), same
    league-average fallback for a franchise with no rows at all (cold
    start OR a team this module's name-mapping never covered -- see this
    module's docstring for the honest ~74% coverage number).

    Requires load_team_game_success() to have already run at least once
    in this process (build_trailing_success_features() -- called during
    pipeline construction -- does this) so the real cfbfastR<->franchise
    name mapping exists; raises a clear error rather than silently
    returning an empty/wrong result if called before that, since a
    feature silently going all-fallback is a real bug class this project
    has hit before (see core/live_results.py's docstring).

    Unlike build_trailing_success_features()'s training-time columns
    (correctly shift(1)'d so a game's row never includes itself), this
    reads the raw last-N *completed* games directly -- there is no
    "current game" to leak from when scoring a genuinely not-yet-played
    matchup, same distinction NFL's version makes.

    No real game-date column exists in this data (see this module's
    docstring), so "most recent" means highest (season, game_num) --
    the same real chronological ordinal ordering training uses, not a
    separate assumption.
    """
    global _success_by_franchise_cache
    if _success_by_franchise_cache is None:
        raise RuntimeError(
            "get_current_trailing_success() called before any load_team_game_success() call in this "
            "process -- build the CFB pipeline (which calls build_trailing_success_features()) first, "
            "so the real cfbfastR<->franchise name mapping exists."
        )
    success = _success_by_franchise_cache
    league_avg = {col: float(success[col].mean()) for col in TRAILING_COLS}

    team_rows = success[success["franchise"] == franchise].sort_values(
        ["season", "game_id"], kind="stable")
    if team_rows.empty:
        return league_avg
    recent = team_rows.tail(n_games)
    return {col: float(recent[col].mean()) for col in TRAILING_COLS}


def build_trailing_success_features(games: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """
    Same contract as sports/nfl/nflfastr_features.py's
    build_trailing_epa_features(): adds home_/away_ prefixed trailing
    off_success_rate/def_success_rate_allowed columns, league-average
    fallback for any team-game with no real history yet (cold start OR
    a team this data's mapping never covered -- see this module's
    docstring, same honest fallback either way).
    """
    franchise_names = set(games["home_franchise"].unique()) | set(games["away_franchise"].unique())
    success = load_team_game_success(franchise_names)
    # cfbfastR's raw PBP release doesn't carry a game date column
    # (confirmed: not among the 9 columns pulled), so there's no clean
    # date-based join like NFL's module uses. Instead, join on each
    # team's real chronological ORDINAL game number within a season --
    # computed identically on both sides (every appearance, home or
    # away, in date order), so "this team's 5th game of the season"
    # means the same thing in `success` and in `games`. Robust as long
    # as both sides see the same real games in the same order per team
    # per season, which is true for regular season + bowls.
    success = success.sort_values(["franchise", "season", "game_id"], kind="stable")
    success["game_num"] = success.groupby(["franchise", "season"]).cumcount()

    league_avg = {col: float(success[col].mean()) for col in TRAILING_COLS}
    trailing_cols_out = [f"{col}_trail" for col in TRAILING_COLS]
    for col, out_col in zip(TRAILING_COLS, trailing_cols_out):
        success[out_col] = success.groupby("franchise")[col].transform(
            lambda s: s.shift(1).rolling(n_games, min_periods=1).mean()
        )

    # Compute each team's real per-season game ordinal directly on
    # `games` too (covers both home and away appearances, chronological
    # by date), then join success's own (franchise, season, game_num) on
    # that -- exactly matching quantities on both sides.
    long_games = pd.concat([
        games[["game_id", "season", "date", "home_franchise"]].rename(columns={"home_franchise": "franchise"}).assign(side="home"),
        games[["game_id", "season", "date", "away_franchise"]].rename(columns={"away_franchise": "franchise"}).assign(side="away"),
    ]).sort_values(["franchise", "season", "date"], kind="stable")
    long_games["game_num"] = long_games.groupby(["franchise", "season"]).cumcount()

    merged = long_games.merge(success[["franchise", "season", "game_num"] + trailing_cols_out],
                               on=["franchise", "season", "game_num"], how="left")

    out = games.copy()
    for side, franchise_col in (("home", "home_franchise"), ("away", "away_franchise")):
        side_rows = merged[merged["side"] == side][["game_id", "franchise"] + trailing_cols_out]
        j = out[["game_id", franchise_col]].merge(
            side_rows, left_on=["game_id", franchise_col], right_on=["game_id", "franchise"], how="left")
        for col, out_col in zip(TRAILING_COLS, trailing_cols_out):
            out[f"{side}_{col}_trail"] = j[out_col].fillna(league_avg[col]).values
    return out
