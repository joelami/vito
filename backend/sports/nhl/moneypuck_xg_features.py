"""
Real team-level trailing expected-goals (xG) features built from MoneyPuck's
free, already-downloaded skater-game files (Datasets/NHL/2008_to_2024 copy 2.csv
+ 2025 copy.csv, situation=="all" rows). Direct successor to the already-
adopted shot-differential/special-teams features (see decision_log.jsonl,
"nhl_trailing_shot_diff_and_pp_rate"/"nhl_trailing_pk_pct", and this
project's own sports/nhl/features.py module docstring) -- modern hockey
analytics (MoneyPuck, Evolving-Hockey, Natural Stat Trick) treats raw shot-
attempt differential (the Corsi/Fenwick family) as a first-generation
possession proxy and xG (shots weighted by real historical conversion rate
given location/type/situation/danger) as its more precise descendant,
specifically because it separates shot VOLUME from shot QUALITY -- two
teams with identical shot differentials can have very different real
scoring-chance quality.

JOIN: MoneyPuck's `gameId` (real NHL API game ids) has no shared key with
this project's own `game_id` -- this is the exact problem
sports/nhl/research_player_matchup.py already solved and verified (its
STEP 1): join on (home_team_id, away_team_id, nearest date, 1-day
tolerance). Mirrored here directly rather than imported from that module --
that script is throwaway research, not production code, the same
relationship sports/nfl/nflfastr_features.py has to its own one-off fetch
script (self-contained production module, not a dependency on a research
script that could be edited/removed independently).

AGGREGATION -- the one real design decision this module makes on its own:
MoneyPuck's on-ice xG columns (OnIce_F_xGoals/OnIce_A_xGoals) are recorded
PER SKATER -- once per player, for every shot taken while that specific
player was on the ice. Multiple skaters share credit for the same shot (at
any moment roughly 5 skaters are on the ice together, more at 5-on-5 which
is the bulk of game time), so literally SUMMING every dressed skater's
on-ice xG for a team-game would ~5x overcount real team xG (verified
directly: inspecting one real game's raw rows shows materially different
OnIce_F_xGoals per skater on the same team-game, driven by real shift/
ice-time overlap differences, not a shared team total already sitting in
this file). Team-game xg_for/xg_against here is computed as the MEAN of
OnIce_F_xGoals/OnIce_A_xGoals across all of that team's dressed skaters
(situation=="all" -- all strengths combined, same all-strengths filter
research_player_matchup.py and shot_diff_l10's shots_for/shots_against
already use) for that game -- NOT a literal team total, but a
consistently-scaled per-team-game statistic (~team_total / (avg skaters
on ice), a roughly constant divisor across teams/games) that is monotonic
with the true team total and therefore carries identical signal for a
tree-based model (HistGradientBoostingRegressor, used throughout this
project's walk-forward pipeline, is invariant to monotonic per-feature
rescaling) -- documented here explicitly so this choice is never mistaken
for a naive-sum bug by a future reader.

TRAILING, walk-forward-safe by construction: exactly the same "last N
games, shift-then-roll, never includes the game being predicted" pattern
shot_diff_l10/pp_pct_l10/pk_pct_l10 already use in sports/nhl/features.py.

CACHING: load_team_game_xg() always re-reads the raw MoneyPuck CSVs fresh
(no stale module-level cache) -- the exact discipline
sports/nfl/nflfastr_features.py's load_team_game_epa() follows, learned
from a real caching bug found and fixed 2026-09-15 in that exact file (see
its reset_caches() docstring): a cache that only populates once and never
invalidates silently freezes a trailing feature at whatever snapshot
existed at process boot, even as the underlying data changes on disk.
get_current_trailing_xg() (the live-scoring path, called per matchup, not
once per pipeline build) DOES cache the expensive joined+aggregated
per-team-game table -- re-reading+re-joining the ~2.6GB raw CSV on every
live matchup call would be needlessly slow, and correctness only requires
that cache be invalidatable, not that it never exist -- via reset_caches(),
same interface as nflfastr_features.reset_caches(). Honest difference from
NFL/CFB: nothing in core/dataset_refresh.py currently re-fetches this
MoneyPuck snapshot periodically (it's a static already-downloaded file, not
pulled from a live API on a schedule the way nflverse/cfbfastR are) -- so
reset_caches() is provided for correctness and interface parity if that
ever changes, but is not currently wired into scheduler.py's daily refresh,
since there is nothing scheduled to refresh it against.
"""

from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "Datasets" / "NHL"
MP_FILE_OLD = DATASETS_DIR / "2008_to_2024 copy 2.csv"   # 2008-2024, skaters, detailed per-situation format
MP_FILE_NEW = DATASETS_DIR / "2025 copy.csv"             # 2025-current, same format

# MoneyPuck 3-letter team code -> this project's canonical numeric team_id.
# Identical mapping already verified directly against the full unique
# playerTeam list in the raw file in sports/nhl/research_player_matchup.py
# (34 codes cross-checked against sports/nhl/loader.py's cleaned team_id/
# team_name pairs) -- copied here rather than imported since that module is
# throwaway research, not production code (see module docstring). ARI and
# UTA both point at the same canonical id as config.py's own
# FRANCHISE_CANONICAL (Phoenix/Arizona -> Utah); ATL and WPG both point at
# the Atlanta Thrashers/Winnipeg Jets' single continuous team_id (28) the
# same way config.py's own docstring documents that franchise -- both
# targets are ALREADY the canonical id, so no separate
# config.canonical_team_id() call is needed downstream.
MONEYPUCK_CODE_TO_TEAM_ID = {
    "ANA": 25, "ARI": 129764, "ATL": 28, "BOS": 1, "BUF": 2, "CAR": 7, "CBJ": 29,
    "CGY": 3, "CHI": 4, "COL": 17, "DAL": 9, "DET": 5, "EDM": 6, "FLA": 26,
    "LAK": 8, "MIN": 30, "MTL": 10, "NJD": 11, "NSH": 27, "NYI": 12, "NYR": 13,
    "OTT": 14, "PHI": 15, "PIT": 16, "SEA": 124292, "SJS": 18, "STL": 19,
    "TBL": 20, "TOR": 21, "UTA": 129764, "VAN": 22, "VGK": 37, "WPG": 28, "WSH": 23,
}

TRAILING_COLS = ["xg_for", "xg_against"]
ROLL_WINDOW = 10  # same trailing window every other NHL feature uses -- see features.py's ROLL_WINDOW


def _load_raw_skater_xg() -> pd.DataFrame:
    """Raw MoneyPuck skater-game rows, filtered to situation=="all" (all
    strengths combined -- one row per skater per game, no double counting
    across 5v5/5v4/4v5/etc., same filter research_player_matchup.py uses),
    with team codes mapped to this project's numeric team_id."""
    cols = ["gameId", "season", "playerTeam", "opposingTeam", "home_or_away",
            "gameDate", "position", "situation", "OnIce_F_xGoals", "OnIce_A_xGoals"]
    old = pd.read_csv(MP_FILE_OLD, usecols=cols)
    new = pd.read_csv(MP_FILE_NEW, usecols=cols)
    raw = pd.concat([old, new], ignore_index=True)
    raw = raw[raw["situation"] == "all"].copy()
    raw["team_id"] = raw["playerTeam"].map(MONEYPUCK_CODE_TO_TEAM_ID)
    raw["opp_team_id"] = raw["opposingTeam"].map(MONEYPUCK_CODE_TO_TEAM_ID)
    raw = raw.dropna(subset=["team_id", "opp_team_id"]).copy()
    raw["date"] = pd.to_datetime(raw["gameDate"], format="%Y%m%d")
    return raw


def _build_gameid_map(raw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """MoneyPuck gameId -> our game_id, via (home_team_id, away_team_id,
    nearest date, 1-day tolerance) -- the exact join already solved and
    verified in research_player_matchup.py's build_gameid_map(), mirrored
    here (see module docstring for why this isn't a shared import)."""
    home_rows = raw[raw["home_or_away"] == "HOME"][["gameId", "team_id", "opp_team_id", "date"]].rename(
        columns={"team_id": "home_team_id", "opp_team_id": "away_team_id"})
    home_rows = home_rows.drop_duplicates(subset=["gameId"]).sort_values("date")

    ours = games[["game_id", "home_team_id", "away_team_id", "date"]].copy()
    ours["date_only"] = ours["date"].dt.tz_localize(None).dt.normalize()
    home_rows["date_only"] = home_rows["date"].dt.normalize()
    ours = ours.sort_values("date_only")
    home_rows = home_rows.sort_values("date_only")

    merged = pd.merge_asof(
        home_rows, ours, on="date_only", by=["home_team_id", "away_team_id"],
        direction="nearest", tolerance=pd.Timedelta("1D"),
    )
    return merged[["gameId", "game_id"]].dropna()


def load_team_game_xg(games: pd.DataFrame = None) -> pd.DataFrame:
    """
    One row per (game_id, team_id) with real `xg_for`/`xg_against` (our own
    canonical game_id/team_id space) -- always re-reads the raw MoneyPuck
    CSVs fresh (see module docstring's CACHING section). `games` defaults
    to sports.nhl.loader.load_games() if not supplied (tests pass a small
    synthetic frame instead) -- must carry `game_id`/`home_team_id`/
    `away_team_id`/`date` (loader.py's own shape) for the join.
    """
    if games is None:
        from sports.nhl.loader import load_games
        games = load_games()

    raw = _load_raw_skater_xg()
    per_game = raw.groupby(["gameId", "team_id"], as_index=False)[
        ["OnIce_F_xGoals", "OnIce_A_xGoals"]
    ].mean().rename(columns={"OnIce_F_xGoals": "xg_for", "OnIce_A_xGoals": "xg_against"})

    gid_map = _build_gameid_map(raw, games)
    per_game = per_game.merge(gid_map, on="gameId", how="inner")

    # merge_asof's nearest-within-1-day join can (rarely, ~1% per
    # research_player_matchup.py's own direct measurement) map two
    # different MoneyPuck gameIds onto the same our-side game_id --
    # collapsed via groupby-mean so every (game_id, team_id) pair is
    # guaranteed unique before this feeds a per-team rolling window below
    # (a duplicate here would silently double a game's weight in the
    # trailing average, not just risk a row-count mismatch).
    per_game = per_game.groupby(["game_id", "team_id"], as_index=False)[["xg_for", "xg_against"]].mean()
    per_game = per_game.merge(games[["game_id", "date"]].drop_duplicates("game_id"), on="game_id", how="left")
    return per_game.sort_values(["team_id", "date"], kind="stable").reset_index(drop=True)


def build_trailing_xg_features(games: pd.DataFrame, n_games: int = ROLL_WINDOW) -> pd.DataFrame:
    """
    Returns `games` with 4 new columns added: home_/away_ prefixed trailing
    (prior `n_games`, shift-then-roll) xg_for_l10/xg_against_l10 (named to
    match this sport's own shot_diff_l10/pp_pct_l10/pk_pct_l10 convention,
    not NFL's "_trail" suffix). League-average fallback (this data's own
    real mean) for any team-game with fewer than `n_games` of real xG
    history yet, or that never matched a real MoneyPuck game at all
    (coverage gap) -- same cold-start convention as every other trailing
    feature in this project.

    `games` must carry `home_team_id`/`away_team_id`/`date`/`game_id`
    (sports/nhl/loader.py's own shape, which build_features() preserves
    unchanged) -- only ever ADDS columns, never drops/reorders rows, so
    it's safe to call right after build_features() the same way
    sports/nfl/nflfastr_features.build_trailing_epa_features() does.
    """
    team_game = load_team_game_xg(games)
    league_avg = {col: float(team_game[col].mean()) for col in TRAILING_COLS}

    trailing = team_game.sort_values(["team_id", "date"], kind="stable").copy()
    trail_cols_out = [f"{c}_l10" for c in TRAILING_COLS]
    for col, out_col in zip(TRAILING_COLS, trail_cols_out):
        trailing[out_col] = trailing.groupby("team_id")[col].transform(
            lambda s: s.shift(1).rolling(n_games, min_periods=1).mean()
        )

    out = games.copy()
    for side, team_col in (("home", "home_team_id"), ("away", "away_team_id")):
        merged = out[[team_col, "game_id"]].merge(
            trailing[["game_id", "team_id"] + trail_cols_out],
            left_on=[team_col, "game_id"], right_on=["team_id", "game_id"], how="left",
        )
        for col, out_col in zip(TRAILING_COLS, trail_cols_out):
            out[f"{side}_{out_col}"] = merged[out_col].fillna(league_avg[col]).values
    return out


_team_game_xg_cache = None


def get_current_trailing_xg(team_id, n_games: int = ROLL_WINDOW) -> dict:
    """
    Live-scoring counterpart to build_trailing_xg_features() -- same real
    gap get_current_trailing_epa() closes for NFL (see that function's own
    docstring): core/matchup.py's generic build_matchup_feature_row()
    dispatches to a sport's own extra_matchup_features() hook for anything
    beyond its shared core columns, and this is that hook's live source for
    xg_for_l10/xg_against_l10 (see sports/nhl/features.py's
    extra_matchup_features()) -- skipping it would silently default both
    columns to 0.0 for every live NHL prediction the moment they join
    ML_FEATURE_COLS (core/matchup.py's own neutral-fallback warning), an
    extreme, wrong value (0.0 expected goals is nowhere near a real
    team-game total, ~2.9 goals).

    `team_id` is this project's own canonical numeric team_id (see
    config.canonical_team_id) -- the same identity space home_franchise/
    away_franchise already live in for NHL (see config.py's module
    docstring: NHL's "franchise" identity literally IS the numeric
    team_id).

    Returns league-average fallback (this data's own real mean) for a
    team_id with no matched rows at all -- same convention as every other
    trailing feature's cold-start handling in this project.
    """
    global _team_game_xg_cache
    if _team_game_xg_cache is None:
        _team_game_xg_cache = load_team_game_xg()
    xg = _team_game_xg_cache
    league_avg = {col: float(xg[col].mean()) for col in TRAILING_COLS}

    team_rows = xg[xg["team_id"] == team_id].sort_values("date", kind="stable")
    if team_rows.empty:
        return {f"{c}_l10": league_avg[c] for c in TRAILING_COLS}
    recent = team_rows.tail(n_games)
    return {f"{c}_l10": float(recent[c].mean()) for c in TRAILING_COLS}


def reset_caches() -> None:
    """See module docstring's CACHING section -- clears the cached
    joined+aggregated per-team-game table get_current_trailing_xg() uses.
    Not currently wired into any scheduled refresh (nothing re-fetches this
    MoneyPuck snapshot periodically today, see module docstring) -- exists
    for correctness and interface parity with
    sports/nfl/nflfastr_features.reset_caches() if that ever changes.
    Always safe to call (a no-op if the cache was never populated)."""
    global _team_game_xg_cache
    _team_game_xg_cache = None
