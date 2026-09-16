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
State Buckeyes"). A normalize-and-prefix-match, refined 2026-09-14 from
its first version's 73.9% (see decision_log.jsonl for the before/after):

  - Direct inspection of the real fetched data (469 distinct cfbfastR
    names) found it standardizes on POSTAL/STANDARD ABBREVIATIONS for
    state names inside a team name ("Central Mich.", "South Fla.",
    "Southern Miss.") rather than spelling them out -- ABBREV_EXPANSIONS
    below is that real, verified abbreviation table (every entry checked
    against an actual cfbfastR name, not guessed), expanded as a
    word-token pass so it applies regardless of position in the name.
  - Accented characters ("San José") get stripped to their ASCII base
    ("San Jose") via unicodedata -- cfbfastR's own data has no accents at
    all, confirmed directly, so this is a one-directional normalization,
    not a guess at how cfbfastR spells anything.
  - Parenthetical qualifiers ("Miami (OH)") are stripped for the general
    match -- necessary because most of them (small-school "(PA)"/"(MN)"
    disambiguators) don't correspond to anything in cfbfastR's FBS-only
    data anyway, so keeping them ONLY suppresses otherwise-real matches on
    THIS data specifically. The one place stripping them blind would
    create a genuine wrong-team collision (Miami (FL) vs Miami (OH), both
    real FBS programs with real cfbfastR entries) is handled explicitly by
    MANUAL_ALIASES below, checked BEFORE the general path ever runs.
  - MANUAL_ALIASES: hand-verified, one-off cases the mechanical rules
    above can't resolve because cfbfastR uses a genuinely different name
    or acronym, not just an abbreviation of the same words (e.g. "NIU" for
    Northern Illinois, "Southern California" for USC) -- each verified
    directly against the real fetched data, not guessed.

Coverage after this pass: see this module's own tests /
research_cfbfastr_success_features.py's printed coverage line for the
current real number. The remaining unmapped tail is overwhelmingly actual
small-school/FCS "buy game" opponents with no cfbfastR entry to map TO at
all (this data is FBS-focused) -- a real, different, lower-stakes
population than a naming mismatch (CFB spread/moneyline already gate
those games' confidence separately -- see decision_log.jsonl's CFB
low-history-rating-floor entry). A team with no mapping gets the
league-average fallback, same convention as every other trailing feature
in this project.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent.parent.parent / "Datasets" / "College Football" / "cfbfastr_team_game_success.csv"

TRAILING_COLS = ["off_success_rate", "def_success_rate_allowed"]

_team_name_cache = None
_success_by_franchise_cache = None  # populated by load_team_game_success(), reused by get_current_trailing_success()

# Real abbreviations cfbfastR's own team names use (verified directly
# against the 469 distinct names in the fetched data -- see this module's
# docstring). Keyed lowercase, without the trailing period (stripped
# before lookup).
ABBREV_EXPANSIONS = {
    "ala": "alabama", "ariz": "arizona", "ark": "arkansas", "caro": "carolina",
    "colo": "colorado", "conn": "connecticut", "fla": "florida", "ga": "georgia",
    "ill": "illinois", "ind": "indiana", "ky": "kentucky", "la": "louisiana",
    "mich": "michigan", "minn": "minnesota", "miss": "mississippi", "mo": "missouri",
    "neb": "nebraska", "okla": "oklahoma", "ore": "oregon", "so": "southern",
    "tenn": "tennessee", "tex": "texas", "va": "virginia", "val": "valley",
    "wash": "washington", "wis": "wisconsin",
}

# Hand-verified one-off aliases the mechanical rules above genuinely can't
# resolve -- keyed by the exact Vito franchise string (not normalized),
# each checked directly against a real cfbfastR name present in the
# fetched data (see this module's docstring for why each one exists).
MANUAL_ALIASES = {
    "USC Trojans": "Southern California",
    "Army Black Knights": "Army West Point",
    "Miami Hurricanes": "Miami (FL)",
    "Miami (OH) RedHawks": "Miami (OH)",
    "Northern Illinois Huskies": "NIU",
    # These three groups would otherwise land in the automatic-collision-
    # removal path above (multiple Vito franchise strings landing on one
    # cfbfastR name) -- hand-verified safe unlike the general case,
    # because EVERY variant shares the exact same real mascot (Vito-side
    # duplicate spelling of one real team, e.g. "San Jose State" vs "San
    # José St", not two different schools), confirmed directly against
    # the collision list this refinement surfaced (see decision_log.jsonl).
    "San Jose State Spartans": "San Jose St.",
    "San José St Spartans": "San Jose St.",
    "San José State Spartans": "San Jose St.",
    "Central State (OH) Marauders": "Central St. (OH)",
    "Central State Marauders": "Central St. (OH)",
    "Cumberland (TN) Bulldogs": "Cumberland (TN)",
    "Cumberland Bulldogs": "Cumberland (TN)",
}


def _normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"\([^)]*\)", " ", name)  # strip parenthetical qualifiers, see docstring
    name = name.replace("'", "")
    words = []
    for word in name.split():
        word = word.rstrip(".").lower()
        if word == "st":
            words.append("state")
        elif word in ABBREV_EXPANSIONS:
            words.append(ABBREV_EXPANSIONS[word])
        else:
            words.append(word)
    return " ".join(words).strip()


def build_team_name_mapping(vito_franchise_names) -> dict:
    """Real, tested normalize+prefix-match (see this module's docstring
    for the full reasoning and current coverage number) -- `vito_franchise_names`
    is this project's own set of CFB franchise strings (sports/cfb/loader.py's
    home_franchise/away_franchise), matched against whatever cfbfastR
    team names are actually present in the fetched data. MANUAL_ALIASES
    is checked first, per franchise, before falling through to the
    mechanical normalize+prefix path.

    SAFETY, not just coverage: a real bug found and fixed during this
    module's 2026-09-14 refinement -- greedy prefix-matching alone
    produced 35 cases where TWO OR MORE genuinely different Vito
    franchise strings matched the SAME cfbfastR name (e.g. "Arkansas
    Monticello Boll Weevils", a real, different, small D2 program, prefix-
    matching onto "Arkansas" the same as "Arkansas Razorbacks" does --
    plain textual prefix-matching can't tell "{2-word school} {1-word
    mascot}" apart from "{1-word school}{2-word continuation of the
    SAME school's own full name}" without real school-identity knowledge
    this function doesn't have). Confusing two real programs would be
    worse than leaving both unmapped: it would silently blend an
    unrelated team's success-rate history into a real program's trailing
    features. Every such collision -- from the mechanical path ONLY, not
    from MANUAL_ALIASES, which are hand-verified -- is therefore dropped
    entirely below: both/all sides fall back to the league average
    instead of either one risking a wrong match. This trades a small
    amount of coverage for correctness, which is the right trade here."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found -- run `python3 scripts/fetch_cfbfastr_team_game_success.py` "
            f"from backend/ first (downloads real cfbfastR play-by-play, no API key needed)."
        )
    cfbfastr_names = set(pd.read_csv(DATA_PATH, usecols=["team"])["team"].unique())
    cfbfastr_norm = {}
    for n in cfbfastr_names:
        cfbfastr_norm.setdefault(_normalize(n), n)

    mapping = {}
    fuzzy_matched = set()  # vito names matched via the mechanical path (not MANUAL_ALIASES) -- eligible for collision removal
    for v in vito_franchise_names:
        if v in MANUAL_ALIASES and MANUAL_ALIASES[v] in cfbfastr_names:
            mapping[v] = MANUAL_ALIASES[v]
            continue
        vn = _normalize(v)
        best = None
        for cn, orig in cfbfastr_norm.items():
            if vn == cn or vn.startswith(cn + " "):
                if best is None or len(cn) > len(best[0]):
                    best = (cn, orig)
        if best:
            mapping[v] = best[1]
            fuzzy_matched.add(v)

    from collections import defaultdict
    by_target = defaultdict(list)
    for v in fuzzy_matched:
        by_target[mapping[v]].append(v)
    dropped = 0
    for target, vs in by_target.items():
        if len(vs) > 1:
            for v in vs:
                del mapping[v]
                dropped += 1
    if dropped:
        print(f"[cfbfastr_features] dropped {dropped} ambiguous team-name match(es) "
              f"(see build_team_name_mapping's docstring) -- those franchises fall back "
              f"to the league average instead of risking a wrong match.")
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
        # Real incident this fixes (2026-09-16): this used to hard-raise
        # here, which was fine as long as build_trailing_success_features()
        # always ran successfully during pipeline construction first (the
        # normal case this docstring describes) -- but pipeline.py's
        # _safe_add_trailing_feature() can now ALSO leave this cache empty
        # on purpose, when the underlying dataset genuinely isn't
        # available yet (not uploaded to R2), so the rest of the sport's
        # pipeline can still build. A live matchup score landing here in
        # that exact situation must degrade the same way training already
        # did -- a neutral fallback, loudly logged -- not crash live
        # scoring on top of an already-known, already-handled gap. Can't
        # attempt to populate the cache HERE either: it needs this
        # project's own full set of franchise names to build the real
        # cfbfastR name mapping (only build_trailing_success_features()
        # has that, via the games dataframe it's called with), not just
        # the single `franchise` string this function receives -- calling
        # load_team_game_success() with anything less would silently
        # build and cache a broken, permanently-empty mapping.
        print(f"[cfbfastr_features] get_current_trailing_success: no successful pipeline build has "
              f"populated the cache yet (dataset likely unavailable) -- returning a neutral fallback "
              f"for {franchise!r}.")
        return {col: 0.0 for col in TRAILING_COLS}
    success = _success_by_franchise_cache
    league_avg = {col: float(success[col].mean()) for col in TRAILING_COLS}

    team_rows = success[success["franchise"] == franchise].sort_values(
        ["season", "game_id"], kind="stable")
    if team_rows.empty:
        return league_avg
    recent = team_rows.tail(n_games)
    return {col: float(recent[col].mean()) for col in TRAILING_COLS}



# ---------------------------------------------------------------------------
# Opponent-adjusted (SOS) trailing success rate -- direct CFB transfer of
# sports/nfl/nflfastr_features.py's build_trailing_epa_features_sos_adjusted()
# / get_current_trailing_epa_sos_adjusted() (adopted 2026-09-15, see
# decision_log.jsonl's "nfl_nflverse_trailing_epa_sos_adjustment" entry),
# flagged as the cleanest cross-sport transfer candidate in
# docs/model_improvement_candidates_2026-09-15.md's "Cross-sport structural
# ideas" section: the raw off_success_rate_trail/def_success_rate_allowed_trail
# columns above are schedule-BLIND -- three good offensive games against
# weak defenses look identical to three good games against elite ones -- and
# CFB's real cross-conference strength disparity (SEC vs. Sun Belt, say) is
# exactly the case this method exists for.
#
# Method (identical to NFL's, see that module's docstring for the full
# reasoning): for each of a team's past games, look up what the OPPONENT's
# own trailing (pre-that-game, walk-forward-safe on the opponent's side too)
# allowed/produced-rate was, subtract it from this team's raw performance in
# that one game, add back the league average so the adjustment is relative
# to a league-average opponent (not zero), THEN roll those per-game adjusted
# values over this team's own trailing window.
#
# The one real difference from NFL, both inherited honestly from this
# module's own team-name-mapping limitation (see module docstring): the
# opponent lookup can only find a trailing rate for an opponent this module
# successfully mapped to a Vito franchise (~72% of games). An opponent it
# could not map contributes NO adjustment for that one past game -- treated
# the same as "opponent has no trailing history yet" (falls back to the
# league average, i.e. a no-op adjustment for that specific game) rather
# than dropping the team's own game from its trailing window entirely. This
# is a LEFT join on game_id (paired games use an inner self-join exactly
# like NFL's, since exactly one real opponent row exists; unpaired games --
# opponent unmapped, or one of the handful of raw cfbfastR game_ids with
# only one team's row at all -- keep their own row with NaN opponent
# columns, filled with league_avg below), not the plain inner self-join
# NFL's version uses, specifically to implement that fallback instead of
# silently losing coverage of the team's OWN games.
# ---------------------------------------------------------------------------
_SOS_OPPONENT_COL = {
    "off_success_rate": "def_success_rate_allowed",
    "def_success_rate_allowed": "off_success_rate",
}

# Populated by build_trailing_success_features_sos_adjusted(), read by
# get_current_trailing_success_sos_adjusted() -- deliberately NOT gated by
# "if _adjusted_per_game_cache is None" the way NFL's equivalent is. NFL's
# populate-once pattern was a real bug (see nflfastr_features.reset_caches()'s
# docstring: it silently froze after the first pipeline build in a
# long-lived process). CFB's OWN existing convention (load_team_game_success()
# always re-reads/re-caches unconditionally, see dataset_refresh.py's
# refresh_all() docstring: "CFB's equivalent does NOT need this") is to
# always recompute on every real build call instead -- matched here so this
# new cache doesn't reintroduce the exact staleness class NFL just fixed.
_adjusted_per_game_cache = None


def _build_adjusted_per_game_table(vito_franchise_names, n_games: int) -> pd.DataFrame:
    """
    Shared core of the opponent adjustment -- mirrors sports/nfl/
    nflfastr_features.py's _build_adjusted_per_game_table() exactly in
    shape (see that function's docstring), adapted for CFB's two real
    differences: (1) self-join key is `game_id` alone, no `gameday` (no
    date column in the raw cfbfastR release -- same reason
    build_trailing_success_features() above uses a (season, game_num)
    ordinal join instead of NFL's date join); (2) the self-join is a LEFT
    join with an explicit paired/solo split (see module comment above),
    not NFL's plain inner join, to implement the "unmapped opponent falls
    back to league average" behavior instead of dropping the team's own
    game.

    Returns one row per (game_id, franchise) actually present in this
    module's franchise-mapped `success` data, with `{col}_adj` columns:
    this team's raw per-game stat, opponent-adjusted, NOT yet rolled into
    a trailing window (the caller's job, same division of labor as NFL's).
    """
    success = load_team_game_success(vito_franchise_names)
    league_avg = {col: float(success[col].mean()) for col in TRAILING_COLS}

    trailing = success[["game_id", "season", "franchise"] + TRAILING_COLS].sort_values(
        ["franchise", "season", "game_id"], kind="stable")
    trailing_cols_out = [f"{col}_trail" for col in TRAILING_COLS]
    for col, out_col in zip(TRAILING_COLS, trailing_cols_out):
        trailing[out_col] = trailing.groupby("franchise")[col].transform(
            lambda s: s.shift(1).rolling(n_games, min_periods=1).mean()
        )

    # Paired games (both sides mapped -- the normal case, exactly 2 rows
    # for this game_id in `trailing`): plain inner self-join, same as
    # NFL's, guaranteed exactly one opponent match. Solo games (opponent
    # unmapped, or one of the handful of raw game_ids with only one row
    # at all -- see load_team_game_success()'s own docstring) keep their
    # one row with opponent columns explicitly NaN, so the adjustment
    # step below falls back to league_avg for them instead of the row
    # disappearing from this team's own trailing-window pool entirely.
    opp_cols = [f"{c}_trail" for c in TRAILING_COLS]
    game_counts = trailing.groupby("game_id")["franchise"].transform("size")
    paired = trailing[game_counts == 2]
    solo = trailing[game_counts != 2].copy()

    self_joined_paired = paired.merge(
        paired[["game_id", "franchise"] + opp_cols], on="game_id", suffixes=("", "_opp"))
    self_joined_paired = self_joined_paired[
        self_joined_paired["franchise"] != self_joined_paired["franchise_opp"]
    ].copy()

    solo["franchise_opp"] = None
    for c in opp_cols:
        solo[f"{c}_opp"] = float("nan")

    self_joined = pd.concat([self_joined_paired, solo], ignore_index=True, sort=False)

    # Per-game opponent-adjusted raw value: this team's raw stat in this
    # one game, minus the opponent's typical (trailing, pre-game) rate on
    # the matching column (league-average fallback for both "opponent had
    # no trailing history yet" AND "opponent unmapped/unpaired" -- both
    # collapse to the same NaN-fill here), plus the league average to
    # re-center.
    for col in TRAILING_COLS:
        opp_col = _SOS_OPPONENT_COL[col]
        opp_trail_col = f"{opp_col}_trail_opp"
        opp_trail = self_joined[opp_trail_col].fillna(league_avg[opp_col])
        self_joined[f"{col}_adj"] = self_joined[col] - opp_trail + league_avg[opp_col]

    return self_joined.sort_values(["franchise", "season", "game_id"], kind="stable")


def build_trailing_success_features_sos_adjusted(games: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """
    Opponent-adjusted counterpart to build_trailing_success_features()
    above -- see the module comment block right above this function for
    the full method and CFB-specific coverage caveat. Produces the same 4
    home_/away_ prefixed columns as build_trailing_success_features(),
    suffixed `_trail_sos` instead of `_trail` so both can coexist on the
    same feature row (this is meant to be added ON TOP of the raw trailing
    columns, not replace them -- see research_success_rate_sos_adjustment.py).

    Uses the exact same (franchise, season, game_num) ordinal join
    build_trailing_success_features() uses to attach onto `games` (no real
    date column in the raw cfbfastR release -- see that function's
    docstring) -- computed independently here on the adjusted-per-game
    table so the ordinal sequence matches exactly (same sort key, same set
    of (franchise, season, game_id) rows, since `_build_adjusted_per_game_table()`
    is derived from the identical `success` frame).
    """
    global _adjusted_per_game_cache
    franchise_names = set(games["home_franchise"].unique()) | set(games["away_franchise"].unique())
    success = load_team_game_success(franchise_names)  # ensures _success_by_franchise_cache is populated for live lookup
    league_avg = {col: float(success[col].mean()) for col in TRAILING_COLS}

    # Always recompute (not "if None") -- see _adjusted_per_game_cache's
    # own module-level comment for why this deliberately does NOT mirror
    # NFL's populate-once/reset_caches() pattern.
    _adjusted_per_game_cache = _build_adjusted_per_game_table(franchise_names, n_games)
    self_joined = _adjusted_per_game_cache.copy()

    sos_cols_out = [f"{col}_trail_sos" for col in TRAILING_COLS]
    for col, out_col in zip(TRAILING_COLS, sos_cols_out):
        self_joined[out_col] = self_joined.groupby("franchise")[f"{col}_adj"].transform(
            lambda s: s.shift(1).rolling(n_games, min_periods=1).mean()
        )

    self_joined = self_joined.sort_values(["franchise", "season", "game_id"], kind="stable")
    self_joined["game_num"] = self_joined.groupby(["franchise", "season"]).cumcount()

    long_games = pd.concat([
        games[["game_id", "season", "date", "home_franchise"]].rename(columns={"home_franchise": "franchise"}).assign(side="home"),
        games[["game_id", "season", "date", "away_franchise"]].rename(columns={"away_franchise": "franchise"}).assign(side="away"),
    ]).sort_values(["franchise", "season", "date"], kind="stable")
    long_games["game_num"] = long_games.groupby(["franchise", "season"]).cumcount()

    merged = long_games.merge(self_joined[["franchise", "season", "game_num"] + sos_cols_out],
                               on=["franchise", "season", "game_num"], how="left")

    out = games.copy()
    for side, franchise_col in (("home", "home_franchise"), ("away", "away_franchise")):
        side_rows = merged[merged["side"] == side][["game_id", "franchise"] + sos_cols_out]
        j = out[["game_id", franchise_col]].merge(
            side_rows, left_on=["game_id", franchise_col], right_on=["game_id", "franchise"], how="left")
        for col, out_col in zip(TRAILING_COLS, sos_cols_out):
            out[f"{side}_{out_col}"] = j[out_col].fillna(league_avg[col]).values
    return out


def get_current_trailing_success_sos_adjusted(franchise: str, n_games: int = 10) -> dict:
    """
    Live-scoring counterpart to build_trailing_success_features_sos_adjusted(),
    mirrors sports/nfl/nflfastr_features.py's
    get_current_trailing_epa_sos_adjusted() exactly -- "current form" IS
    the mean of the last N real, already-adjusted games, no re-rolling
    needed.

    Requires build_trailing_success_features_sos_adjusted() (called during
    CFB pipeline construction) to have already run at least once in this
    process, same requirement get_current_trailing_success() has for the
    name-mapping cache -- raises a clear error rather than silently
    returning an empty/wrong result if called first, same reasoning as
    that function's own docstring.
    """
    if _adjusted_per_game_cache is None:
        raise RuntimeError(
            "get_current_trailing_success_sos_adjusted() called before any "
            "build_trailing_success_features_sos_adjusted() call in this process -- "
            "build the CFB pipeline (which calls it) first, so the opponent-adjusted "
            "per-game table exists."
        )
    self_joined = _adjusted_per_game_cache
    league_avg_adj = {col: float(_success_by_franchise_cache[col].mean()) for col in TRAILING_COLS}

    team_rows = self_joined[self_joined["franchise"] == franchise].sort_values(
        ["season", "game_id"], kind="stable")
    if team_rows.empty:
        return {f"{col}_trail_sos": league_avg_adj[col] for col in TRAILING_COLS}
    recent = team_rows.tail(n_games)
    return {f"{col}_trail_sos": float(recent[f"{col}_adj"].mean()) for col in TRAILING_COLS}


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
