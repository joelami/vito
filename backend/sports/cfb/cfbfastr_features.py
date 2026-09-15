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
