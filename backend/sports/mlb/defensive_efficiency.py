"""
Team defensive efficiency (DER), joined onto the MLB games frame.

WHAT THIS IS: Defensive Efficiency Ratio = 1 - (balls in play that became
hits) / (total balls in play) -- a real, decades-old sabermetric team-
defense stat (Bill James's "Defensive Efficiency Record"). This is the
direct answer to the specific gap the already-adopted starter_kbb_pct_
rolling feature names in its own reasoning (sports/mlb/features.py's
module docstring): K-BB% "only counts outcomes the pitcher himself
overwhelmingly controls... stripping out defense/ballpark/sequencing-luck
noise a runs-allowed proxy can't separate from true skill" -- but nothing
in ML_FEATURE_COLS measures the DEFENSE's own contribution once a ball is
actually put in play. DER is that missing complementary signal, and is
attributed to the FIELDING TEAM (a team defense stat by definition), not
to any individual pitcher -- distinct in kind from every other MLB
pitching feature adopted so far.

SOURCE: the same Datasets/MLB/2010seve/2020seve Retrosheet play-by-play
event files already parsed for starting_pitcher.py, starter_kbb_quality.py,
and research_bullpen_arm_quality.py -- same data source, different
aggregation, not a new acquisition. This module directly reuses starter_
kbb_quality's already-verified file iteration and base-code EXTRACTION
(`_iter_event_files`, `_extract_base_code`) rather than re-deriving that
parsing a second, riskier time, and attributes each play to the PITCHING
team (the fielding side) via the same start/sub fieldpos==1 tracking
every other MLB pitching feature here already uses -- vh_flag='0' means
the visiting team is batting, so the HOME team is pitching/fielding, and
vice versa (verified against the same 2020-07-28 SEA@ANA game every MLB
pitching-feature module in this project spot-checks -- see starter_kbb_
quality.py's docstring for that check).

Base-code CLASSIFICATION (which extracted codes are a real ball in play,
a hit, or neither) is this module's own new logic, NOT a reuse of
starter_kbb_quality's NON_PA_BASE_CODES/WALK_BASE_CODES exact-match sets
-- see the "Base-code classification" comment block below for exactly
why: those sets are exact-string matches that work for K-BB% (K/W/IW
never carry a suffix in this data) but silently miss suffixed variants of
the SAME semantic events (e.g. "SB2", "CS2(24)") that this module found
by checking every one of the 1,212 distinct base codes actually appearing
in the dataset, not by assuming the K-BB% feature's classification
transfers unchanged.

SCOPING NOTE, stated honestly rather than hidden: the classic Bill James
DER formula (numerator H-HR, denominator AB-K-HR+SF) treats a reached-on-
error (E) or fielder's-choice (FC) plate appearance as a CONVERTED OUT for
DER purposes, purely because neither counts as a "hit" -- even though the
batter reached base safely. That is a known, accepted property of this
specific metric (it measures "did the ball become a hit," not "did the
batter reach base"), not a bug introduced here -- this module reproduces
that same standard treatment rather than inventing a different one.

COVERAGE: same ~2010-2025 event-file window as every other Retrosheet-
derived MLB feature in this project. Any game outside that window, or a
team's true first tracked games within it, get the league-average
fallback below.
"""

import csv
import re

import numpy as np
import pandas as pd

from .starter_kbb_quality import EVENT_SUBDIRS, _extract_base_code, _iter_event_files

# Trailing team-GAMES considered. 10 matches ROLL_WINDOW in features.py --
# the same window every other team-level trailing stat here uses
# (pf_l10/pa_l10/win_pct_l10/ats_pct_l10), which keeps this feature's
# naming (`der_l10`) and recency-vs-stability tradeoff directly comparable
# to those rather than introducing a second, unexplained window size. A
# team fields roughly 27-30 balls in play per game, so 10 games already
# gives ~270-300 BIP entering the window -- a sample size in the same
# ballpark as a normal batting-average denominator, not a thin one.
ROLL_N_GAMES = 10

# ---------------------------------------------------------------------
# Base-code classification. NOTE, discovered while building this module
# and verified directly against every one of the 1,212 distinct base
# codes actually appearing in Datasets/MLB/2010seve+2020seve (not assumed
# from starter_kbb_quality.py's exact-string NON_PA_BASE_CODES/
# WALK_BASE_CODES sets): those sets work for K-BB% because K and W/IW
# never carry a trailing digit/parenthetical suffix in this data, but
# several of the SAME semantic baserunning-only events DO carry one here
# -- e.g. "SB2"/"SB3" (stolen base of 2nd/3rd, not just bare "SB"),
# "CS2(24)"/"CSH(242)" (caught stealing, not just bare "CS"/"CSH"),
# "POCS1(1)" etc. An exact-match check (as starter_kbb_quality.py uses)
# would silently miss all of these and misclassify them as a real,
# fielded ball in play -- a real accuracy problem for a stat this
# granular that K-BB%'s exact-match approach could tolerate (it only
# slightly dilutes the PA denominator) but DER cannot (it would count
# thousands of pure baserunning plays as fielding chances). This module
# therefore uses its own PREFIX-based classification instead of
# re-importing NON_PA_BASE_CODES/WALK_BASE_CODES directly -- same
# semantic set of excluded events, verified correct against the real,
# full distinct-code list before being trusted (see research_defensive_
# efficiency.py's printed diagnostic for the classification-bucket dump
# this was checked against).
# ---------------------------------------------------------------------

# Baserunning-only / record-keeping-only events -- prefix-matched (not
# exact) since these carry base-number/parenthetical suffixes in this
# data (see note above). Not a fielding chance at all, and NOT a real
# plate appearance either.
NON_PA_PREFIXES = ("NP", "SB", "SBH", "CS", "CSH", "PO", "POCS", "POCSH", "WP", "PB", "BK", "DI", "OA")

_SINGLE_RE = re.compile(r"^S\d")   # "S7", "S8", "S49", ... (fielder-location suffix)
_DOUBLE_RE = re.compile(r"^D\d")   # "D7", "D57", ...
_TRIPLE_RE = re.compile(r"^T\d")   # "T8", "T89", ...


def _classify_base_code(base: str) -> str:
    """
    Returns one of: "non_pa" (not a real plate appearance, not a fielding
    chance -- skip entirely), "non_bip" (a real PA, but not a fielding
    chance -- strikeout/walk/HBP/home run), "hit" (a real ball in play
    that became a hit -- single/double/triple/ground-rule double), or
    "bip_out" (a real ball in play that did NOT become a hit -- every
    numeric fielding-sequence out code, fielder's choice, reached-on-
    error, catcher's interference -- see module docstring's scoping note
    on why E/FC count as "not a hit" here, the same standard-DER
    treatment the classic Bill James formula gives them).
    """
    if not base:
        return "non_pa"
    if base.startswith(NON_PA_PREFIXES):
        return "non_pa"
    if base.startswith("K"):          # strikeout, incl. suffixed variants like "K23"
        return "non_bip"
    if base == "W" or base.startswith("IW"):
        return "non_bip"
    if base.startswith("HP"):         # hit by pitch
        return "non_bip"
    if base.startswith("HR"):         # home run -- can't be fielded, excluded from BIP
        return "non_bip"
    if base.startswith("DGR"):        # ground-rule double
        return "hit"
    if base == "S" or _SINGLE_RE.match(base):
        return "hit"
    if base == "D" or _DOUBLE_RE.match(base):
        return "hit"
    if base == "T" or _TRIPLE_RE.match(base):
        return "hit"
    return "bip_out"   # numeric fielding-sequence outs, FC, E, C, and anything else real-but-unrecognized

# League-average DER over a trailing-10-game window, computed directly
# from every clean team-game extracted from the 2010seve/2020seve event
# files (37,343 team-games; mean of the real, non-fallback rolling
# home_der_lN/away_der_lN values = 0.7072/0.7082 -- see research_
# defensive_efficiency.py's diagnostic run). Used as the fallback for (a)
# a team's true first tracked games in this window, and (b) any game
# outside the ~2010-2025 event-file coverage entirely. Matches the real-
# world sabermetric range for this stat (~0.69-0.71 in published sources).
LEAGUE_AVG_DER = 0.707


def _parse_file_der_log(path):
    """
    Same play-by-play walk as starter_kbb_quality._parse_file_pa_log --
    tracks which team is PITCHING (fielding) at each play via the same
    start/sub fieldpos==1 mechanism -- but tallies balls-in-play/hits per
    TEAM rather than K/BB per PITCHER, since DER is a team defense stat.
    Returns a list of (game_meta, team_stats) per game, where team_stats
    is {"home": {"bip": int, "hits": int}, "away": {"bip": int, "hits": int}}
    (the fielding team's own tallies for that one game).
    """
    games = []
    cur = None
    stats = None

    with open(path, encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            tag = row[0]

            if tag == "id":
                if cur is not None:
                    games.append((cur, stats))
                cur = {}
                stats = {"home": {"bip": 0, "hits": 0}, "away": {"bip": 0, "hits": 0}}
            elif cur is None:
                continue
            elif tag == "info":
                key = row[1]
                val = row[2] if len(row) > 2 else ""
                if key in ("visteam", "hometeam", "date", "number"):
                    cur[key] = val
            elif tag == "play" and len(row) >= 7:
                vh_flag, event = row[2], row[6]
                # vh_flag='0' -> visiting team batting -> HOME team is
                # pitching/fielding this play; vh_flag='1' -> the reverse.
                fielding_side = "home" if vh_flag == "0" else "away"
                base = _extract_base_code(event)
                kind = _classify_base_code(base)
                if kind in ("non_pa", "non_bip"):
                    continue
                stats[fielding_side]["bip"] += 1
                if kind == "hit":
                    stats[fielding_side]["hits"] += 1

    if cur is not None:
        games.append((cur, stats))
    return games


def load_team_game_der() -> pd.DataFrame:
    """
    One row per game found in the event files, with each side's own
    fielding tallies for that game: date, game_number, canonical
    franchises, home_bip/home_hits, away_bip/away_hits. Games missing a
    team/date entirely are dropped (mirrors starting_pitcher.load_event_
    games's discipline); a game with zero real balls in play for a side
    (extremely rare -- e.g. a very short relief-only irregular game) is
    kept with bip=0, handled downstream via the rolling sum (a game
    contributing 0/0 to the rolling numerator/denominator is a true no-op,
    not a fabricated value).
    """
    from . import config

    rows = []
    for f in _iter_event_files():
        for game_meta, stats in _parse_file_der_log(f):
            date, number = game_meta.get("date"), game_meta.get("number", "0")
            home_raw, away_raw = game_meta.get("hometeam"), game_meta.get("visteam")
            if date is None or home_raw is None or away_raw is None:
                continue
            rows.append({
                "date": date, "game_number": number,
                "home_team_raw": home_raw, "away_team_raw": away_raw,
                "home_bip": stats["home"]["bip"], "home_hits": stats["home"]["hits"],
                "away_bip": stats["away"]["bip"], "away_hits": stats["away"]["hits"],
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["home_franchise"] = df["home_team_raw"].apply(config.canonical_team)
    df["away_franchise"] = df["away_team_raw"].apply(config.canonical_team)
    return df


def _build_rolling_der(der_games: pd.DataFrame, n: int = ROLL_N_GAMES) -> pd.DataFrame:
    """
    Long-format per-team-game log, sorted chronologically per team, with a
    shift(1)-then-rolling(n, min_periods=1) SUM of balls-in-play/hits over
    the team's last `n` games (summed counts, then a ratio -- not an
    average of per-game rates, same convention starter_kbb_quality.py uses
    for K-BB%) -- current game's own result never included in its own
    feature, the same walk-forward discipline every rolling feature in
    this project follows.
    """
    home = der_games[["date", "game_number", "home_franchise", "away_franchise", "home_bip", "home_hits"]].rename(
        columns={"home_franchise": "team", "away_franchise": "opponent", "home_bip": "bip", "home_hits": "hits"}
    )
    home["side"] = "home"
    away = der_games[["date", "game_number", "home_franchise", "away_franchise", "away_bip", "away_hits"]].rename(
        columns={"away_franchise": "team", "home_franchise": "opponent", "away_bip": "bip", "away_hits": "hits"}
    )
    away["side"] = "away"

    log = pd.concat([home, away], ignore_index=True)
    log = log.sort_values(["team", "date", "game_number"], kind="stable").reset_index(drop=True)

    grp = log.groupby("team", group_keys=False)
    roll_bip = grp["bip"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    roll_hits = grp["hits"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        der = 1.0 - (roll_hits / roll_bip)
    log["der_lN"] = der.where(roll_bip.fillna(0) > 0, LEAGUE_AVG_DER)

    cols = ["date", "game_number", "team", "opponent", "der_lN"]
    home_roll = log[log["side"] == "home"][cols].rename(
        columns={"team": "home_franchise", "opponent": "away_franchise", "der_lN": "home_der_lN"}
    )
    away_roll = log[log["side"] == "away"][cols].rename(
        columns={"team": "away_franchise", "opponent": "home_franchise", "der_lN": "away_der_lN"}
    )

    out = der_games.merge(home_roll, on=["date", "game_number", "home_franchise", "away_franchise"], how="left")
    out = out.merge(away_roll, on=["date", "game_number", "home_franchise", "away_franchise"], how="left")
    return out


def attach_defensive_efficiency(games: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins three new columns onto `games` (sports/mlb/loader.py's
    output, or anything carrying date/game_number/home_franchise/
    away_franchise): `home_der_lN` / `away_der_lN` (each team's rolling
    defensive efficiency over their last ROLL_N_GAMES games, walk-forward
    -safe) and `der_diff_lN` (home minus away -- DER is a "higher is
    better" stat, same polarity/sign convention as sp_kbb_pct_diff_lN:
    positive means home's defense favors home). Any game outside the
    ~2010-2025 event-file coverage, or not cleanly matched, gets the
    LEAGUE_AVG_DER fallback -- same discipline as every other Retrosheet-
    derived MLB feature.
    """
    der_games = load_team_game_der()
    rolled = _build_rolling_der(der_games)

    key = ["date", "game_number", "home_franchise", "away_franchise"]
    dupe_counts = rolled.groupby(key).size()
    if (dupe_counts > 1).any():
        rolled = rolled.drop_duplicates(subset=key, keep="first")

    merged = games.merge(rolled[key + ["home_der_lN", "away_der_lN"]], on=key, how="left")
    assert len(merged) == len(games), "attach_defensive_efficiency must return exactly one row per input game"

    merged["home_der_lN"] = merged["home_der_lN"].fillna(LEAGUE_AVG_DER)
    merged["away_der_lN"] = merged["away_der_lN"].fillna(LEAGUE_AVG_DER)
    merged["der_diff_lN"] = merged["home_der_lN"] - merged["away_der_lN"]
    return merged


_current_snapshot_cache = None


def _compute_current_der_by_team() -> pd.Series:
    """
    Each team's trailing DER as of RIGHT NOW (its most recent
    ROLL_N_GAMES games, including the latest one -- unlike the walk-
    forward columns in `attach_defensive_efficiency` above, which
    deliberately exclude the current row) -- same convention and
    reasoning as features.current_form_snapshot() / park_factor.
    current_park_factor_snapshot(): no leakage concern, every game in the
    parsed event-file log is legitimately in the past relative to a
    brand-new one being scored. `.transform()` (Series in, Series out per
    group), not `.apply()` -- sidesteps the same pandas-version
    incompatibility sports/nfl/nflfastr_features.py's own comment
    documents (`.apply()`'s `include_groups` kwarg doesn't exist on the
    pandas version this project runs).
    """
    der_games = load_team_game_der()
    home = der_games[["date", "home_franchise", "home_bip", "home_hits"]].rename(
        columns={"home_franchise": "team", "home_bip": "bip", "home_hits": "hits"}
    )
    away = der_games[["date", "away_franchise", "away_bip", "away_hits"]].rename(
        columns={"away_franchise": "team", "away_bip": "bip", "away_hits": "hits"}
    )
    log = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")

    grp = log.groupby("team")
    log["_recent_bip"] = grp["bip"].transform(lambda s: s.rolling(ROLL_N_GAMES, min_periods=1).sum())
    log["_recent_hits"] = grp["hits"].transform(lambda s: s.rolling(ROLL_N_GAMES, min_periods=1).sum())

    latest = log.groupby("team").tail(1).set_index("team")
    with np.errstate(invalid="ignore", divide="ignore"):
        der_now = 1.0 - (latest["_recent_hits"] / latest["_recent_bip"])
    return der_now.where(latest["_recent_bip"] > 0, LEAGUE_AVG_DER)


def get_current_der(team_franchise: str) -> float:
    """
    Live-scoring counterpart to `attach_defensive_efficiency` above --
    same real gap/convention as park_factor.get_current_park_factor() and
    sports/mlb/features.py's other extra_matchup_features() hooks close:
    core/matchup.py's build_matchup_feature_row() has no loader row to
    read `home_der_lN`/`away_der_lN` from for a brand-new, not-yet-played
    game. Cached lazily (module-level, reset via `reset_cache()`).
    """
    global _current_snapshot_cache
    if _current_snapshot_cache is None:
        _current_snapshot_cache = _compute_current_der_by_team()

    if team_franchise in _current_snapshot_cache.index:
        return float(_current_snapshot_cache.loc[team_franchise])
    return LEAGUE_AVG_DER


def reset_cache() -> None:
    """See nflfastr_features.reset_caches()'s docstring for the real
    staleness bug class this exists to prevent in a long-lived process."""
    global _current_snapshot_cache
    _current_snapshot_cache = None
