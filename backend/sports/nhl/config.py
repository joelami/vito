"""
NHL-specific constants. Mirrors sports/cfb/config.py and sports/mlb/config.py's
structure so this sport shares the exact same downstream pipeline shape as
every other sport in this project. Every constant that's "tuned" rather than
derived lives here so it's obvious what's a modeling assumption vs. a fact
pulled from data.

=============================================================================
CORRECTIONS TO THE BUILD BRIEF - verified directly against the data, not
assumed. Read this before trusting anything else in this module.
=============================================================================

1. THE BRIEF HAD THE TWO CSV FILES BACKWARDS. `Datasets/NHL/archive/
   nhl_data_plus.csv` (32 columns) is the LEANER file - raw per-team-game box
   scores, odds, and record strings, no engineered columns at all.
   `nhl_data_extensive.csv` (131 columns) is the one with all the roll_3/
   roll_10/roll_30/season_*/opp_*/diff engineered columns, built FROM
   nhl_data_plus.csv's raw rows by `Datasets/NHL/archive/
   build_dataset_extra.py` (confirmed by diffing that script's
   `load_and_flatten_data()` output column list against nhl_data_plus.csv's
   32 columns - they match exactly; nhl_data_plus.csv IS that function's raw
   output). This module follows the brief's explicit instruction to use
   nhl_data_plus.csv anyway - it turns out to be the right file for the
   "recompute your own walk-forward-safe features, don't trust pre-computed
   ones" fallback the brief asked for, since there ARE no pre-computed
   rolling columns in it to accidentally trust. (For the record:
   build_dataset_extra.py's rolling logic - shift(1) BEFORE
   .rolling().mean(), and cumulative season stats also shifted - IS
   walk-forward-safe by inspection, so nhl_data_extensive.csv's roll_*
   columns are not obviously leaky either. They're simply not used here,
   per the brief's instruction to use nhl_data_plus.csv.)

2. THE BRIEF'S ROW-SHAPE CLAIM WAS WRONG. This is NOT one row per game - it
   is one row per TEAM PER GAME, exactly like the NBA source
   (sports/nba/loader.py's exact same shape): 61,376 rows / 30,688 unique
   game_ids = 2 rows per game_id, distinguished by `is_home` (1/0). Every
   column described as "opponent" info in the brief (`opp_*`) does not exist
   in this file at all - that was actually describing nhl_data_extensive.csv.
   loader.py pivots this into one row per game the same way sports/nba/
   loader.py does (merge the is_home==1 and is_home==0 rows on game_id).

3. `spread` IS the home team's line (negative = home favored), verified
   directly: it is IDENTICAL across both team-rows of the same game_id in
   6,651/6,651 games checked (never sign-flipped by team perspective), and
   home teams win 60.0% of games when spread < 0 vs. 40.2% when spread > 0
   (n=4,321 / 2,328) - a real, correctly-signed signal, not noise. It is
   overwhelmingly the NHL puck line's standard +/-1.5 (13,282 of 13,520
   priced rows), so it's used as-is, no sign flip - same convention as CFB's
   Home Line.

4. `favorite_moneyline` is a GENUINELY HARDER problem than CFB's single-sided
   spread/total, and harder than the brief anticipated. See the "MONEYLINE"
   section below - this is not just "one side given, back out the other,"
   the side identity itself (which team - home or away - the price belongs
   to) is not reliably recoverable from this file.

5. THREE REAL DATA-QUALITY ISSUES were found and are cleaned in loader.py,
   none of which the brief mentioned:
     a. ~1,619 games (mostly the entirely-cancelled 2004-05 lockout season,
        plus ~194 COVID-postponed March/April 2020 games, plus a handful of
        isolated data-glitch rows) are PLACEHOLDER schedule rows with
        goals_for == goals_against == 0 and a frozen record_summary - games
        that were scheduled but never actually played. Dropped, exactly like
        CFB drops completed=0 rows.
     b. A real, pre-shootout-era tie: 78 games in season 2004 (the tail of
        the 2003-04 season, before the 2005-06 shootout rule existed) have a
        genuine 0/0 win flag on both sides with a non-zero, non-equal-looking
        score... no - these end with equal goals_for on both sides (a true
        tie, e.g. 3-3), which is a legitimate NHL result in that pre-shootout
        era. Dropped rather than mishandled, same spirit as MLB's 13 true
        ties - a tie has no winner, which breaks the home_win 0/1 convention.
     c. Preseason exhibition games (no odds ~90% of the time, and NOT part of
        a team's real standings) are mixed into the file with no explicit
        flag. Identified and dropped via each team's own `record_wins +
        record_losses + record_otl` resetting to a low value after
        accumulating higher during Sept/Oct - see loader.py's
        `_flag_preseason` docstring for the full mechanics and the 17
        (of 590) team-seasons where this heuristic is ambiguous.
     d. All-Star games, Olympic/World-Cup exhibition games, and NHL
        preseason "Global Series" games against European club teams are
        present under non-NHL team_ids (All-Star draft "teams" like "Team
        McDavid", national teams like "Sweden"/"Finland", European clubs
        like "Jokerit"/"HC Davos"). 33 real current-era NHL franchise
        team_ids exist in this data (30 original + Vegas 37 + Seattle 124292
        + Utah 129764); 67 games (134 team-game rows) involve at least one
        non-NHL opponent and are dropped.

=============================================================================
MONEYLINE - the hardest, most honestly-caveated part of this module
=============================================================================
`favorite_moneyline` is a single American-odds number, IDENTICAL across both
team-rows of a game (like `spread`), sourced (per `download_dataset.py` line
123-128) from ESPN's pickcenter API: `homeTeamOdds.moneyLine` if ESPN flagged
home as `favorite`, else `awayTeamOdds.moneyLine`. That is a real market
price for a real, ESPN-identified favorite - but the flattening step that
produced nhl_data_plus.csv did NOT retain which side (home or away) ESPN's
`favorite` flag pointed at. It has to be reconstructed.

The obvious reconstruction - "the spread favorite (spread<0 => home, >0 =>
away) is the moneyline favorite" - is checked directly against
favorite_moneyline's own sign in loader.py's `_MONEYLINE_SIDE_CONCORDANCE`
computation and is FAR less reliable than expected for a same-market price:
  - Overall sign concordance between spread-implied favorite and
    favorite_moneyline's sign: 55.5% (n=6,641) - barely better than a coin
    flip.
  - Concordance RISES with how decisive the favorite is: 57.4% for
    |favorite_moneyline| >= 110, 69.4% for >= 150, 81.2% for >= 200 (n=521).
This pattern (low concordance for close games, much higher for lopsided
ones) is consistent with a real, if noisy, signal - not pure garbage - but
it means using the spread-sign heuristic to assign home/away will get the
side WRONG close to half the time for competitive games, which is exactly
where a market edge would matter most.

Decision made here: still compute a moneyline market (spread-sign
assignment + a devig, see below), because outright deleting it would throw
away real, if noisy, price information the brief explicitly asked for a
defensible attempt at - but treat every downstream moneyline number as
LOWER CONFIDENCE than spread/total, and verify.py prints the concordance
numbers above prominently so this is never silently trusted. This is a
materially deeper problem than CFB's "one clean side, standard-vig the
other" case - there, the side was never in doubt, only the counterparty
price was assumed.

Devig mechanics, once a side is assigned: `favorite_moneyline`'s raw implied
probability is computed, and the OTHER side's price is backed out by
assuming the SAME total market overround as this project already assumes for
CFB's -110/-110 spread/total (2 * implied_prob(-110) - 1 = 4.762%) - i.e.
p_dog = (1 + 0.04762) - p_fav, decimal_dog = 1 / p_dog. This is a stated
assumption (no real NHL-specific vig estimate exists without the missing
side), not derived from this dataset, and is on top of the side-assignment
uncertainty above.
"""

from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent.parent / "Datasets" / "NHL" / "archive" / "nhl_data_plus.csv"


# ---------- Power rating engine ----------
# Tuned specifically for NHL, NOT copy-pasted from any other sport. An
# 82-game season sits between NFL's 17 and MLB's 162 (arguing for a small-ish
# k_factor, similar reasoning to MLB/NBA), and NHL's real scoring margins are
# tiny and tightly bunched - after cleaning (see module docstring), mean home
# margin is +0.27 goals, margin std-dev is ~2.1 goals, actual home win rate
# is 54.5% (n=26,904 real, decided games) - much smaller margins even than
# MLB's runs, arguing for a gentle MOV multiplier close to MLB's.
#
# Selection process: identical to CFB/MLB/NBA - a grid search over
# (k_factor, home_field_adv, season_regression, mov_mult_base,
# mov_mult_divisor), scored by mean |predicted - actual| home-win rate across
# calibration deciles of elo_home_win_prob against real home_win (see
# verify.py section 2). The winning config scored 0.0102 mean decile error -
# see verify.py's printed calibration table for the actual numbers.
ELO_K_FACTOR = 6.0             # small, similar reasoning to MLB's 162-game season - 82 games/season is enough signal that per-game rating swings should be modest
ELO_START_RATING = 1500.0
HOME_FIELD_ADV_ELO = 25.0      # NHL home-ice advantage is real but modest (54.5% actual home win rate here, well below NFL's or CFB's) - smaller than NFL's 48, much smaller than CFB's 90
SEASON_REGRESSION = 0.20       # smaller than NFL's 0.33/CFB's 0.30 - NHL rosters (like MLB's) carry over materially year to year relative to a season's worth of signal
MOV_MULT_BASE = 1.5
MOV_MULT_DIVISOR = 1.5         # gentle curve for NHL's tiny goal margins - closer to MLB's 1.0/1.2 than NFL's 2.2/2.2 or CFB's 2.5/3.5

# Points-per-elo-point conversion (in GOALS, not points/runs), used to turn a
# rating differential into a predicted scoring margin. Empirically fit (not
# guessed): regressing actual_margin on rating_diff_pre at the tuned config
# above gives a slope of ~0.00807, i.e. ~124 elo points per 1 goal of margin
# (correlation 0.198 - real but modest, consistent with hockey's
# well-documented high game-to-game variance/parity relative to other major
# leagues). As with every other sport here, this is just the starting scale -
# `ensemble.compute_residual_stds` fits the actual out-of-sample residual
# std-dev at runtime.
ELO_POINTS_PER_MARGIN = 124.0


# ---------- Season boundaries ----------
# NHL seasons cross the Jan 1 boundary (regular season Oct -> April,
# playoffs into June, occasionally later - see COVID-era note below).
# Verified directly against this dataset's own `season` column: applying
# `dt.year + 1 if dt.month > 8 else dt.year` to every row reproduces the
# CSV's own `season` column exactly (0 mismatches across all 61,376 rows) -
# this is also, by inspection of build_dataset_extra.py's
# `load_and_flatten_data()`, the literal formula used to compute that column
# in the first place, so this isn't a coincidental match. Note this labels a
# season by the calendar year it ENDS in (e.g. the 2019-20 season is
# season=2020), matching the NHL's own "20XX-YY season" convention read as
# "ends in YY". The loader trusts the CSV's own `season` column directly but
# this function is kept for interface parity with the other sports and for
# scoring any future/manually-entered game with no season column yet.
#
# Two real oddities this formula still handles correctly, verified directly:
#   - The 2004-05 lockout season (entirely cancelled) appears in this data
#     ONLY as placeholder schedule rows (goals_for=goals_against=0, frozen
#     0-0-0 record) - real games, not fabricated ones - dropped in loader.py
#     per the module docstring, not a season-boundary problem.
#   - The COVID-shifted 2019-20 postseason (played Aug 2020) and 2020-21
#     Stanley Cup Final (played July 2021) both still resolve to their
#     correct season under this formula (Aug: month=8, not >8, stays in the
#     season that started the previous autumn; July: month=7, same) -
#     confirmed these rows carry real, non-frozen, incrementing playoff
#     records (see loader.py), i.e. real games, correctly labeled.
def season_for_date(dt) -> int:
    return dt.year + 1 if dt.month > 8 else dt.year


# ---------- Team identity / franchise continuity ----------
# `team_id` in this data is ALREADY a stable identifier across ordinary
# renames within one franchise - verified directly by inspecting every
# distinct (team_id, team_name) pair in the raw file:
#   - team_id 24: "Phoenix Coyotes" (through 2013-14) -> "Arizona Coyotes"
#     (2014-15 onward) - same id, no mapping needed.
#   - team_id 25: "Anaheim Mighty Ducks" (through 2005-06) -> "Anaheim
#     Ducks" (2006-07 onward) - same id, no mapping needed.
#   - team_id 28: "Atlanta Thrashers" (through 2010-11) -> "Winnipeg Jets"
#     (2011-12 onward) - the Atlanta -> Winnipeg relocation the brief called
#     out - already unified under one id, no mapping needed.
#   - team_id 8505/8594 in the raw team_id column is a display-name-only
#     split (see below): "Utah Hockey Club" (2024-25) -> "Utah Mammoth"
#     (2025-26) both under the SAME numeric team_id (129764 once printed as
#     an int) - no mapping needed, listed here only because it looks like a
#     rename at a glance.
#
# The ONE relocation in this window that does NOT carry a continuous
# team_id: the Arizona Coyotes' 2024 move to Utah. team_id 24 ("Arizona
# Coyotes") last appears 2024-04-18 (end of the 2023-24 regular season);
# team_id 129764 ("Utah Hockey Club") first appears 2024-09-22 (2024-25
# preseason) - a clean, non-overlapping hand-off, exactly the same pattern
# MLB's OAK -> ATH mapping was built from. This is the one entry
# FRANCHISE_CANONICAL needs.
#
# 33 distinct "real" NHL team_ids exist in this data across 2004-2026: the
# 30 standard franchises (ids 1-30, which already absorb the three renames
# above), Vegas Golden Knights (37, 2017 expansion), Seattle Kraken (124292,
# 2021 expansion), and Utah (129764, 2024 relocation of team_id 24). See
# loader.py for how the remaining ~70 non-NHL team_ids (All-Star draft
# teams, national teams, European exhibition clubs) are filtered out.
REAL_TEAM_IDS = frozenset(range(1, 31)) | {37, 124292, 129764}

FRANCHISE_CANONICAL = {
    24: 129764,   # Phoenix/Arizona Coyotes -> Utah Hockey Club/Mammoth, 2024 relocation
}


def canonical_team_id(team_id: int) -> int:
    return FRANCHISE_CANONICAL.get(team_id, team_id)


# ---------- ESPN display-name resolution (live harness only) ----------
# This dataset's own team_id has no relationship at all to ESPN's naming
# convention — ESPN's scoreboard API returns display-name strings
# ("Vegas Golden Knights"), never these numeric ids. This is the bridge the
# live forward-test harness (core/matchup.py) needs, mapping straight to
# the ALREADY-canonicalized id (post `FRANCHISE_CANONICAL` — e.g. every
# Utah name below resolves directly to 129764, never the retired id 24) so
# callers don't need to double-canonicalize.
ESPN_NAME_TO_TEAM_ID = {
    "Boston Bruins": 1, "Buffalo Sabres": 2, "Calgary Flames": 3, "Chicago Blackhawks": 4,
    "Detroit Red Wings": 5, "Edmonton Oilers": 6, "Carolina Hurricanes": 7,
    "Los Angeles Kings": 8, "Dallas Stars": 9, "Montreal Canadiens": 10,
    "New Jersey Devils": 11, "New York Islanders": 12, "New York Rangers": 13,
    "Ottawa Senators": 14, "Philadelphia Flyers": 15, "Pittsburgh Penguins": 16,
    "Colorado Avalanche": 17, "San Jose Sharks": 18, "St. Louis Blues": 19,
    "Tampa Bay Lightning": 20, "Toronto Maple Leafs": 21, "Vancouver Canucks": 22,
    "Washington Capitals": 23, "Anaheim Ducks": 25, "Florida Panthers": 26,
    "Nashville Predators": 27, "Winnipeg Jets": 28, "Columbus Blue Jackets": 29,
    "Minnesota Wild": 30, "Vegas Golden Knights": 37, "Seattle Kraken": 124292,
    "Utah Mammoth": 129764, "Utah Hockey Club": 129764, "Arizona Coyotes": 129764,
}


def canonical_franchise(espn_display_name: str):
    """ESPN display name -> this module's already-canonicalized numeric
    team_id. Falls back to the raw name unchanged on a miss (caller
    defaults to ELO_START_RATING via a dict-lookup miss) rather than
    crashing."""
    return ESPN_NAME_TO_TEAM_ID.get(espn_display_name, espn_display_name)


# Reverse of ESPN_NAME_TO_TEAM_ID — for display purposes only (e.g. the
# Ratings tab), since `home_franchise` here is always the numeric team_id
# with no display-name column to fall back to. Utah's three historical
# names all map to 129764; dict construction keeps whichever is defined
# LAST above ("Arizona Coyotes" — the outdated one), overridden explicitly
# below to prefer the current name.
TEAM_ID_TO_DISPLAY_NAME = {tid: name for name, tid in ESPN_NAME_TO_TEAM_ID.items()}
TEAM_ID_TO_DISPLAY_NAME[129764] = "Utah Mammoth"


def franchise_display_name(team_id) -> str:
    """This module's numeric team_id -> a real display name, for the UI.
    Falls back to the id itself (stringified) if unrecognized rather than
    crashing."""
    return TEAM_ID_TO_DISPLAY_NAME.get(team_id, str(team_id))
