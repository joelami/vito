"""
Real team-level trailing expected-goals (xG) features built from
MoneyPuck's free, TEAM-level per-game data (Datasets/NHL/
moneypuck_team_game_xg.csv -- see scripts/fetch_moneypuck_team_xg.py for
how it's built). Direct successor to the already-adopted shot-
differential/special-teams features (see decision_log.jsonl,
"nhl_trailing_shot_diff_and_pp_rate"/"nhl_trailing_pk_pct", and this
project's own sports/nhl/features.py module docstring) -- modern hockey
analytics (MoneyPuck, Evolving-Hockey, Natural Stat Trick) treats raw shot-
attempt differential (the Corsi/Fenwick family) as a first-generation
possession proxy and xG (shots weighted by real historical conversion rate
given location/type/situation/danger) as its more precise descendant,
specifically because it separates shot VOLUME from shot QUALITY -- two
teams with identical shot differentials can have very different real
scoring-chance quality.

REWRITTEN 2026-10-15 (app owner: "let's go back in with all of this
season's data... and make the necessary changes"). The original version
of this module (see git history / decision_log.jsonl's "nhl_moneypuck_
trailing_xg_features" entry) read MoneyPuck's SKATER-level files
(Datasets/NHL/"2008_to_2024 copy 2.csv" + "2025 copy.csv") and
approximated team xG as the MEAN of each dressed skater's own on-ice xG
(documented at the time as "not a literal team total, but a consistently-
scaled... statistic that is monotonic with the true team total"). That
module's own docstring ALSO flagged, honestly, that nothing refreshed
those files for the current season at all -- confirmed directly while
investigating why NHL's live trailing form looked stale: the local
skater files topped out at MoneyPuck's own "season 2025" label (the
2025-26 season), so every live NHL matchup this entire 2026-27 season was
scoring off LAST season's trailing average, not this one's.

Both problems share one real fix: MoneyPuck also publishes a real,
free, keyless, nightly-updated TEAM-level file (confirmed directly,
2026-10-12: https://moneypuck.com/moneypuck/playerData/careers/gameByGame/
all_teams.csv, "updated nightly" per MoneyPuck's own data page) whose
`position=="Team Level"` rows carry the REAL team xGoalsFor/xGoalsAgainst
directly -- no skater-averaging approximation needed at all, and (unlike
the old static files) genuinely re-fetchable, so it can be kept current
in-season the same way NFL's EPA / CFB's success-rate datasets already
are (see core/dataset_refresh.py). Re-validated via sports/nhl/
research_xg_features.py after this rewrite -- see decision_log.jsonl for
the real, re-measured fit numbers (full sample + required split-half
check) against the CORRECTED, non-approximated data.

JOIN: MoneyPuck's `gameId` (real NHL API game ids) has no shared key with
this project's own `game_id` -- this is the exact problem
sports/nhl/research_player_matchup.py already solved and verified (its
STEP 1): join on (home_team_id, away_team_id, nearest date, 1-day
tolerance). Mirrored here directly rather than imported from that module --
that script is throwaway research, not production code, the same
relationship sports/nfl/nflfastr_features.py has to its own one-off fetch
script (self-contained production module, not a dependency on a research
script that could be edited/removed independently). Unchanged by this
rewrite -- the join problem and its solution are the same regardless of
whether the upstream rows are skater- or team-level.

TRAILING, walk-forward-safe by construction: exactly the same "last N
games, shift-then-roll, never includes the game being predicted" pattern
shot_diff_l10/pp_pct_l10/pk_pct_l10 already use in sports/nhl/features.py.

CACHING: load_team_game_xg() always re-reads the raw CSV fresh (no stale
module-level cache) -- the exact discipline sports/nfl/nflfastr_features.py's
load_team_game_epa() follows, learned from a real caching bug found and
fixed 2026-09-15 in that exact file (see its reset_caches() docstring): a
cache that only populates once and never invalidates silently freezes a
trailing feature at whatever snapshot existed at process boot, even as
the underlying data changes on disk. get_current_trailing_xg() (the
live-scoring path, called per matchup, not once per pipeline build) DOES
cache the expensive joined per-team-game table -- correctness only
requires that cache be invalidatable, not that it never exist -- via
reset_caches(), same interface as nflfastr_features.reset_caches(), now
ALSO actually wired into core/dataset_refresh.py's refresh_all() (the
real gap this rewrite closes -- see that module's own docstring).

Missing-dataset handling matches NFL/CFB's own convention exactly (see
core/dataset_refresh.py's module docstring and pipeline.py's
_safe_add_trailing_feature() for the real production incident, 2026-09-16,
that established this pattern): both load_team_game_xg() and
get_current_trailing_xg() raise/degrade on a genuinely missing file the
same honest way those two sports' equivalents do, rather than crashing an
entire sport's pipeline the way an earlier, unguarded version of this
exact wiring did before that incident.
"""

from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "Datasets" / "NHL"
DATA_PATH = DATASETS_DIR / "moneypuck_team_game_xg.csv"

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
# config.canonical_team_id() call is needed downstream. Unchanged by the
# 2026-10-15 rewrite -- this new file uses the identical 3-letter codes
# (`team`/`playerTeam` columns), confirmed directly.
MONEYPUCK_CODE_TO_TEAM_ID = {
    "ANA": 25, "ARI": 129764, "ATL": 28, "BOS": 1, "BUF": 2, "CAR": 7, "CBJ": 29,
    "CGY": 3, "CHI": 4, "COL": 17, "DAL": 9, "DET": 5, "EDM": 6, "FLA": 26,
    "LAK": 8, "MIN": 30, "MTL": 10, "NJD": 11, "NSH": 27, "NYI": 12, "NYR": 13,
    "OTT": 14, "PHI": 15, "PIT": 16, "SEA": 124292, "SJS": 18, "STL": 19,
    "TBL": 20, "TOR": 21, "UTA": 129764, "VAN": 22, "VGK": 37, "WPG": 28, "WSH": 23,
}

TRAILING_COLS = ["xg_for", "xg_against"]
ROLL_WINDOW = 10  # same trailing window every other NHL feature uses -- see features.py's ROLL_WINDOW


def _load_raw_team_xg() -> pd.DataFrame:
    """Raw MoneyPuck TEAM-level rows (already filtered to position=="Team
    Level" and situation=="all" by the fetch script -- see its own
    docstring), with team codes mapped to this project's numeric team_id.
    Raises FileNotFoundError with a clear message if the dataset hasn't
    been fetched yet -- same "raise loudly, don't silently go all-NaN"
    convention as sports/nfl/nflfastr_features.py's load_team_game_epa()."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found -- run `python3 scripts/fetch_moneypuck_team_xg.py` "
            f"from backend/ first (downloads real MoneyPuck team-game data, no API key needed)."
        )
    raw = pd.read_csv(DATA_PATH)
    raw["team_id"] = raw["playerTeam"].map(MONEYPUCK_CODE_TO_TEAM_ID)
    raw["opp_team_id"] = raw["opposingTeam"].map(MONEYPUCK_CODE_TO_TEAM_ID)
    raw = raw.dropna(subset=["team_id", "opp_team_id"]).copy()
    # Real bug found and fixed 2026-10-15 while verifying this rewrite:
    # .map() against a dict of int VALUES still upcasts the whole column to
    # float64 the moment ANY row fails to map (NaN forces float, even
    # before dropna() removes those rows) -- `games`' own home_team_id/
    # away_team_id are real int64, and merge_asof's `by=` columns refuse a
    # float64-vs-int64 mismatch outright (a real, loud MergeError, not a
    # silent wrong join) -- caught here by actually running the join, not
    # assumed safe from reading the code.
    raw["team_id"] = raw["team_id"].astype("int64")
    raw["opp_team_id"] = raw["opp_team_id"].astype("int64")
    raw["date"] = pd.to_datetime(raw["gameDate"], format="%Y%m%d")
    return raw


def _build_gameid_map(raw: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """MoneyPuck gameId -> our game_id, via (home_team_id, away_team_id,
    nearest date, 1-day tolerance) -- the exact join already solved and
    verified in research_player_matchup.py's build_gameid_map(), mirrored
    here (see module docstring for why this isn't a shared import).
    Unchanged by the 2026-10-15 rewrite -- same join, same solution,
    regardless of whether the upstream rows are skater- or team-level."""
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
    CSV fresh (see module docstring's CACHING section). `games` defaults
    to sports.nhl.loader.load_games() if not supplied (tests pass a small
    synthetic frame instead) -- must carry `game_id`/`home_team_id`/
    `away_team_id`/`date` (loader.py's own shape) for the join.

    No more per-skater groupby-mean needed (the 2026-10-15 rewrite's real
    simplification): the raw rows are already one real row per
    (gameId, team), so each team-game's xg_for/xg_against IS the raw
    value directly -- only the post-join merge_asof dedup below (needed
    because the DATE-based join itself, not the source data, can rarely
    collide two different MoneyPuck gameIds onto the same game_id) still
    needs its own groupby-mean.
    """
    if games is None:
        from sports.nhl.loader import load_games
        games = load_games()

    raw = _load_raw_team_xg()
    per_game = raw[["gameId", "team_id", "xg_for", "xg_against"]].drop_duplicates(subset=["gameId", "team_id"])

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
    team-game total, ~2.87 in this data).

    `team_id` is this project's own canonical numeric team_id (see
    config.canonical_team_id) -- the same identity space home_franchise/
    away_franchise already live in for NHL (see config.py's module
    docstring: NHL's "franchise" identity literally IS the numeric
    team_id).

    Real bug found and fixed 2026-10-15, discovered while verifying the
    MoneyPuck rewrite's claim to finally see this season's real games
    live: this used to call load_team_game_xg() -- which, with no
    `games` argument, defaults to sports.nhl.loader.load_games(), the
    STATIC BULK historical CSV that (same as NFL's/CFB's own bulk files)
    does not yet contain ANY 2026-27 season rows. load_team_game_xg()'s
    own gid_map join is an INNER join against that bulk `games` table --
    so even though the refreshed MoneyPuck data genuinely has this
    season's real games, every one of them was silently dropped before
    ever reaching this function, because there was no row in the bulk
    table to join them onto. Confirmed directly: Boston's own cached
    trailing rows topped out at December 2025 even after a fresh fetch
    that demonstrably included October 2026 rows in the raw file.

    Real fix, mirroring get_current_trailing_epa()'s OWN actual design
    (re-checked directly, not assumed): that function never joins to this
    project's own `games`/game_id at all for the LIVE path -- it reads
    the raw per-franchise rows straight from its own data source, sorted
    by date, and takes the most recent N. There is no need for a
    game_id match here either: "this team's own most recent real games"
    doesn't require matching them to OUR schedule at all, only
    build_trailing_xg_features() (the TRAINING path, which attaches
    values onto our own historical rows) actually needs that join. This
    now reads directly from _load_raw_team_xg(), the same way
    get_current_trailing_epa() reads directly from load_team_game_epa().

    Returns a neutral 0.0 fallback (NOT the league average) if the
    dataset genuinely isn't available at all -- same convention
    pipeline.py's _safe_add_trailing_feature() and NFL/CFB's equivalent
    live-scoring functions established 2026-09-16 after a real production
    incident (see core/dataset_refresh.py's module docstring) -- and the
    real league-average fallback (this data's own mean) for a team_id
    with no matched rows but the dataset itself IS present, same
    convention as every other trailing feature's cold-start handling.
    """
    global _team_game_xg_cache
    if _team_game_xg_cache is None:
        try:
            raw = _load_raw_team_xg()
            _team_game_xg_cache = raw[["team_id", "date", "xg_for", "xg_against"]].drop_duplicates(
                subset=["team_id", "date", "xg_for", "xg_against"])
        except FileNotFoundError as e:
            print(f"[moneypuck_xg_features] get_current_trailing_xg: dataset not available, "
                  f"returning a neutral fallback: {e}")
            return {f"{c}_l10": 0.0 for c in TRAILING_COLS}
    xg = _team_game_xg_cache
    league_avg = {col: float(xg[col].mean()) for col in TRAILING_COLS}

    team_rows = xg[xg["team_id"] == team_id].sort_values("date", kind="stable")
    if team_rows.empty:
        return {f"{c}_l10": league_avg[c] for c in TRAILING_COLS}
    recent = team_rows.tail(n_games)
    return {f"{c}_l10": float(recent[c].mean()) for c in TRAILING_COLS}


def reset_caches() -> None:
    """See module docstring's CACHING section -- clears the cached joined
    per-team-game table get_current_trailing_xg() uses. Now ACTUALLY wired
    into core/dataset_refresh.py's refresh_all() (the real gap the
    2026-10-15 rewrite closes -- see that module's own docstring and this
    module's top docstring), the same way sports/nfl/nflfastr_features.
    reset_caches() already is. Always safe to call (a no-op if the cache
    was never populated)."""
    global _team_game_xg_cache
    _team_game_xg_cache = None
