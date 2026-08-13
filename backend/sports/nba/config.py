"""
NBA-specific constants. Mirrors sports/nfl/config.py, sports/cfb/config.py, and
sports/mlb/config.py's structure so all four sports share the exact same
downstream pipeline shape. Every constant that's "tuned" rather than derived
lives here so it's obvious what's a modeling assumption vs. a fact pulled
from data.

Data source: Datasets/NBA/archive/{nba_games_all,nba_teams_all,
nba_betting_spread,nba_betting_money_line,nba_betting_totals}.csv.

IMPORTANT CONTEXT NOT PRESENT IN CFB/MLB: real, two-sided American-odds
betting data exists here (10 books, 2006-07 through 2017-18 seasons), unlike
CFB (assumed -110 juice on spread/total) or MLB (zero odds anywhere). See
loader.py's module docstring for the full verification of book coverage,
odds direction convention, and the CLV caveat.

---------------------------------------------------------------------------
Tuning note: NBA constants are NOT copy-pasted from NFL/CFB/MLB. An 82-game
regular season is ~5x NFL's 17 games (much more signal accumulates per team
per season, arguing for a smaller k_factor so ratings don't overreact to
single-game noise a full season would average out - the same reasoning MLB's
162-game season used), but NBA margins/std-devs turn out to sit MUCH closer
to NFL's scale than MLB's tiny run-margins do. Checked directly against this
dataset (2006-07 through 2017-18 season_years, Regular Season + Playoffs,
known team_ids, valid non-zero scores - the same window the betting data
covers): home teams win 59.6% of games, average home margin is +3.0 points,
margin std-dev is ~13.3 points, average total is ~201.2 points with a std-dev
of ~20.7 - the margin scale is close to NFL's (~13-14) despite the much
higher-scoring game.

Selection process: identical to CFB/MLB - a grid search over (k_factor,
home_field_adv, season_regression, mov_mult_base, mov_mult_divisor), scored
by mean |predicted - actual| home-win rate across walk-forward calibration
deciles (see verify.py section 2). Run in three widening/refining passes
(coarse grid -> the winning region kept drifting toward higher
home_field_adv, so the range was widened twice and re-searched until the
optimum stopped moving toward the boundary): 225 combos, then 270 combos
extending home_field_adv up to 120, then a final 40-combo pass around
k=[11-14]/hfa=[100-120] confirmed the optimum sits at hfa=105, not still
climbing. Winning config's mean |predicted - actual| across calibration
deciles: 0.0045 - noticeably tighter than CFB's own tuned result (0.022),
consistent with NBA's 82-game season and Elo's log-loss-style formula having
far more per-team signal to fit against. See verify.py's printed calibration
table for the full decile-by-decile numbers; the values below are what the
search selected, not NFL's/CFB's/MLB's numbers reused on faith.
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent.parent / "Datasets" / "NBA" / "archive"
GAMES_PATH = DATA_DIR / "nba_games_all.csv"
TEAMS_PATH = DATA_DIR / "nba_teams_all.csv"
SPREAD_PATH = DATA_DIR / "nba_betting_spread.csv"
MONEYLINE_PATH = DATA_DIR / "nba_betting_money_line.csv"
TOTALS_PATH = DATA_DIR / "nba_betting_totals.csv"

# Pinnacle Sports is the sharp/professional book in this data (same reasoning
# NFL's own loader used Pinnacle-years as the "harder to beat" era). Verified
# directly against this dataset (see loader.py docstring / verify.py section 1)
# rather than assumed: Pinnacle covers 14,894/14,914 (99.9%) of spread-bearing
# games, 14,822/14,906 (99.4%) of moneyline-bearing games, and 14,899/14,918
# (99.9%) of total-bearing games, essentially matching "any book" coverage in
# every season from 2006-07 through 2017-18 - so there is no meaningful
# coverage gap to fill with a second book, and using a single sharp book
# avoids any book-shopping/best-price ambiguity a multi-book blend would add.
PRIMARY_BOOK = "Pinnacle Sports"


# ---------- Power rating engine ----------
# Tuned specifically for NBA (see verify.py section 2 for the calibration
# table this grid search produced; full search process described above).
ELO_K_FACTOR = 13.0             # NBA's 82-game season means more signal accumulates per team per season than NFL's 17 (arguing a smaller k, like MLB's 162-game season pushed k down to 6), landing between NFL's 20 and MLB's 6 - roughly proportional to how much regular-season data each sport provides per team
ELO_START_RATING = 1500.0
HOME_FIELD_ADV_ELO = 105.0      # notably LARGER than NFL's 48 or even CFB's 90 - surprising at first, but this is the rating engine's own Elo-scale parameter, not a direct read of the ~2-3 point real scoreboard home edge; the grid search selected it purely to minimize calibration error (0.0045, tighter than every other sport module here), and a real, well-documented NBA home-court edge (~60% win rate, confirmed below) is exactly what a large hfa-in-this-formula is fitting
SEASON_REGRESSION = 0.33        # matches NFL's exactly - the search tried lower values (MLB's 0.25, a CFB-like 0.27) and 0.33 still won, i.e. NBA rosters/rotations don't carry over across a season boundary any more reliably than NFL's do in this Elo formulation
MOV_MULT_BASE = 2.0
MOV_MULT_DIVISOR = 2.5          # close to NFL's 2.2/2.2 - NBA blowouts (20-30 point margins) are common but rarely reach CFB's 40-60 point extremes, so only a mild adjustment from NFL's own values was needed

# Points-per-elo-point conversion, used to turn a rating differential into a
# predicted scoring margin. Empirically fit (not guessed) by regressing
# actual_margin on rating_diff_pre at the tuned config above: slope 0.0343,
# intercept +3.82 (n=59,093), correlation 0.404 -> ~29.2 elo points per point
# of margin, rounded to 29. See verify.py for the same check re-run and
# printed at runtime.
ELO_POINTS_PER_MARGIN = 29.0


# ---------- Season boundaries ----------
# NBA seasons cross the Jan 1 boundary like NFL/CFB (regular season
# Oct/Nov -> April, playoffs into June). Verified directly against this
# dataset's own `season_year` column: applying `dt.year if dt.month >= 8
# else dt.year - 1` to every row with a valid (non-NaT) date and
# season_type in {Regular Season, Playoffs} reproduces `season_year` exactly
# (0 mismatches across 59,589 games) - including the 2011-12 lockout season,
# which started late (Dec 25, 2011) but is still correctly month>=8 so
# resolves to season_year=2011. The loader trusts the CSV's own
# `season_year` column directly (already correct) but this function is kept
# for interface parity with NFL/CFB and for scoring any future/manually
# entered game that has no season_year yet.
def season_for_date(dt) -> int:
    return dt.year if dt.month >= 8 else dt.year - 1


# ---------- Team identity ----------
# UNLIKE CFB (321 unrealigned team strings) and MLB (30 franchises behind 34
# raw codes needing a 4-entry relocation map), NBA's `team_id` in this data
# is ALREADY the NBA stats API's own stable, continuous identifier across
# every relocation and rename in the window this data covers - verified
# directly, not assumed:
#   - nba_teams_all.csv lists exactly 30 team_ids for the 30 currently-active
#     franchises, each carrying a `min_year` that reaches back to that
#     specific team_id's TRUE original founding/first-game year - e.g.
#     New Orleans Pelicans (team_id 1610612740) shows min_year=2002 (their
#     first season as the relocated Charlotte/New Orleans Hornets), while
#     Charlotte Hornets (team_id 1610612766) separately shows min_year=1988 -
#     correctly reflecting the NBA's 2014 ruling that returned the original
#     1988-2002 Charlotte Hornets franchise history to the CURRENT Charlotte
#     Hornets (formerly the 2004 Bobcats expansion team), not to New Orleans.
#     A naive "same city/nickname" mapping would get this exact case backwards.
#   - Spot-checked team_id 1610612751 (Brooklyn Nets): min_year=1976, one
#     continuous ID spanning the New Jersey Nets era (1977-2012) through the
#     2012 Brooklyn move - no separate "NJN" identity to reconcile.
#   - Spot-checked team_id 1610612760 (OKC): min_year=1967, one continuous ID
#     spanning the Seattle SuperSonics era (1967-2008) through the 2008 OKC
#     move.
# So no relocation-mapping table is needed or built here (unlike MLB's
# FRANCHISE_CANONICAL) - `canonical_team(team_id)` is a direct lookup of the
# CURRENT abbreviation for that already-continuous team_id.
#
# What IS filtered out (see loader.py): 733 rows across the full 1950-2018
# history use non-standard team_ids that are NOT among these 30 - almost all
# early-1950s defunct pre-shot-clock-era teams (Sheboygan Redskins,
# Indianapolis Olympians, etc.) plus a handful of All-Star-game placeholder
# "teams" in later years. None of these overlap the 2006-2018 odds-covered
# window (verified in verify.py) - they only affect the pre-2006 history that
# has no betting data to test against anyway.
def load_team_abbrev_map(teams_df) -> dict:
    """teams_df = the raw nba_teams_all.csv DataFrame. Returns {team_id: abbreviation}
    for the 30 team_ids that have a real (non-blank) abbreviation - i.e. exactly
    the currently-active franchises, each already continuous back to its true
    founding year per the verification above."""
    valid = teams_df.dropna(subset=["abbreviation"])
    valid = valid[valid["abbreviation"].astype(str).str.strip() != ""]
    return dict(zip(valid["team_id"], valid["abbreviation"]))


# ---------- ESPN display-name resolution (live harness only) ----------
# Unlike NFL/MLB, this module never needed a canonical-name function before
# now — `home_franchise` is populated directly from `load_team_abbrev_map`
# at load time (a team_id -> abbreviation lookup, not a display-name one),
# so there was nothing that took a display-name string as input. ESPN's
# scoreboard API only gives full display names ("Los Angeles Lakers"), so
# the live forward-test harness (core/matchup.py) needs this bridge into
# the same abbreviation space `home_franchise` already uses.
ESPN_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def canonical_franchise(espn_display_name: str) -> str:
    """ESPN display name -> the abbreviation `load_team_abbrev_map` already
    resolves every real team_id to. Falls back to the raw name unchanged on
    a miss (caller defaults to ELO_START_RATING) rather than crashing."""
    return ESPN_NAME_TO_ABBREV.get(espn_display_name, espn_display_name)


# Reverse of ESPN_NAME_TO_ABBREV — for display purposes only (e.g. the
# Ratings tab), since `home_franchise` here is always an abbreviation
# (via `load_team_abbrev_map`) with no display-name column to fall back to.
# "Los Angeles Clippers" and "LA Clippers" both map to LAC; dict
# construction keeps whichever is defined LAST in ESPN_NAME_TO_ABBREV above
# ("Los Angeles Clippers"), overridden explicitly below to prefer "LA
# Clippers" — the team's own current official branding, dropped "Los
# Angeles" in 2024.
ABBREV_TO_DISPLAY_NAME = {abbrev: name for name, abbrev in ESPN_NAME_TO_ABBREV.items()}
ABBREV_TO_DISPLAY_NAME["LAC"] = "LA Clippers"


def franchise_display_name(abbrev: str) -> str:
    """This module's abbreviation -> a real display name, for the UI. Falls
    back to the raw abbreviation unchanged if unrecognized rather than
    crashing."""
    return ABBREV_TO_DISPLAY_NAME.get(abbrev, abbrev)


# ---------- Season type filtering ----------
# Only Regular Season and Playoffs games are kept for rating/feature purposes
# (see loader.py). Pre Season games are dropped - rotations are experimental,
# starters rest, and results are not a reliable signal of true team strength
# (the same spirit as why other sports' loaders drop incomplete/placeholder
# rows rather than mishandle them). All Star games are dropped - exhibition,
# not competitive, and mostly carry non-standard team_ids anyway (see above).
KEPT_SEASON_TYPES = ("Regular Season", "Playoffs")
