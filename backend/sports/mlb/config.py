"""
MLB-specific constants. Mirrors sports/nfl/config.py and sports/cfb/config.py's
structure so all three sports share the exact same downstream pipeline shape.
Every constant that's "tuned" rather than derived lives here so it's obvious
what's a modeling assumption vs. a fact pulled from data.

IMPORTANT CONTEXT NOT PRESENT IN THE OTHER TWO SPORTS' config.py: the primary
dataset here (Retrosheet game logs, Datasets/MLB/gl{1990_99,2000_09,2010_19,
2020_25}/*.txt, loaded by loader.py) carries ZERO betting-line data - no
spread, no total, no moneyline, in any season, 1990-2025. A SEPARATE real
odds archive (Datasets/MLB/archive/, loaded/joined by odds_loader.py) was
found later and covers 2012-2021 only - real, two-sided moneyline/run-line/
total prices for ~22.7k of the 83k+ games in this module. Power ratings and
the walk-forward ML model below are trained on the FULL 1990-2025 history
(walk-forward-honest either way), and a real backtest now runs over the
2012-2021 odds-covered window - see sports/mlb/verify.py for the actual
numbers and sports/mlb/odds_loader.py for exactly what the odds source is
and isn't.
"""

from pathlib import Path

# Retrosheet game logs are split across four decade-ish directories, one file
# per season, no header row. DATA_DIR (not a single DATA_PATH like NFL/CFB)
# because the loader needs to glob every file across all four subdirectories.
DATA_DIR = Path(__file__).parent.parent.parent.parent / "Datasets" / "MLB"
DATA_GLOB_DIRS = ["gl1990_99", "gl2000_09", "gl2010_19", "gl2020_25"]


# ---------- Power rating engine ----------
# Tuned specifically for MLB, NOT copy-pasted from NFL or CFB. Two things
# make baseball structurally different from both:
#   1. A 162-game season is ~10x NFL's 17 and ~12x CFB's ~13 - far more
#      signal accumulates per team per season, so ratings should move much
#      more slowly per game (a small k_factor) or they'll overreact to
#      single-game noise that regresses out over a full season anyway.
#   2. Scoring margins are tiny and compressed (typically 1-5 runs, total
#      games usually 2-8 runs combined) compared to NFL/CFB's point scales -
#      nothing like CFB's 40-60 point blowouts. The margin-of-victory
#      multiplier needs a much gentler curve as a result.
# Home-field advantage in MLB is also historically well-documented at ~54%
# league-wide (much smaller than NFL's), confirmed directly against this
# dataset below.
#
# Selection process: grid search over 1,200 combinations of (k_factor in
# {3,4,6,8,10}, home_field_adv in {16,20,24,28,32}, season_regression in
# {0.15,0.20,0.25,0.30}, mov_mult_base in {0.5,1.0,1.5}, mov_mult_divisor in
# {0.8,1.2,1.6,2.2}), scored by mean |predicted - actual| home-win rate
# across walk-forward calibration deciles (see verify.py section 2 - the
# identical calibration-decile check used for NFL and CFB). Best score:
# k=4, hfa=28, season_regression=0.15, mov_base=1.0, mov_divisor=2.2 ->
# mean decile calibration error 0.0029 (a k=8/mov_base=0.5 combo tied it
# exactly, confirming k and mov_base trade off near-linearly at fixed
# k*mov_base; k=4 was kept as the simpler/smaller value). This is
# dramatically better calibration than either NFL or CFB achieved (NFL
# ~0.047, CFB's own tuned config ~0.022) - expected, given ~83k games here
# vs. NFL/CFB's low thousands, and MLB's much more stable, well-documented
# home-field effect. See verify.py's printed calibration table for the
# actual numbers behind this choice.
ELO_K_FACTOR = 4.0             # far smaller than NFL's 20 / CFB's 40 - 162 games/season means ratings need to move slowly per game or they whipsaw on single-game noise that a full season would otherwise average out
ELO_START_RATING = 1500.0
HOME_FIELD_ADV_ELO = 28.0      # reproduces MLB's real ~53.7% home win rate almost exactly (1/(1+10^(-28/400)) = 0.540) - see verify.py cross-check against this dataset's actual home_win rate; much smaller than NFL's 48 or CFB's 90
SEASON_REGRESSION = 0.15       # smaller than NFL's 0.33 and CFB's 0.30 - the grid search preferred less season-to-season regression than either, plausibly because a 162-game season already lets ratings converge to a team's true talent within-season, leaving less "noise" from the prior season that needs regressing away
MOV_MULT_BASE = 1.0
MOV_MULT_DIVISOR = 2.2          # needed alongside the small k_factor above so the margin-of-victory multiplier stays in a sane range for baseball's tiny actual margins (log(margin+1)) compared to football's

# Points-per-elo-point conversion, used to turn a rating differential into a
# predicted scoring margin (in RUNS, not points, for MLB). Empirically fit
# against this dataset's actual margin vs. rating_diff_pre relationship at
# the tuned config above (see verify.py) - NOT guessed: regressing
# actual_margin on rating_diff_pre gives a slope of ~0.0142 (~70.3 elo
# points per run of margin), rounded to 70. This is LARGER than NFL's 25 or
# CFB's 14 (i.e. it takes many more elo points to imply one run of margin)
# because the small k_factor/home_field_adv tuned above keep MLB ratings
# clustered much closer to the 1500 starting point than NFL/CFB ratings
# spread out to - the calibration-decile search optimized win-probability
# accuracy, not margin-scale spread, and win probability turned out to want
# a "tighter" rating distribution for this sport. Correlation of
# rating_diff_pre alone with actual_margin is a modest 0.147 as a direct
# consequence (see verify.py) - Elo's job here is primarily calibrated
# win/loss probability, not margin prediction; the ML model below uses many
# more features than rating_diff_pre alone to predict margin/total, and (per
# verify.py's printed comparison) clearly outperforms pure Elo-implied
# margin on that task. As with NFL/CFB, this constant is just the starting
# scale - `ensemble.compute_residual_stds` fits the actual out-of-sample
# residual std-dev at runtime.
ELO_POINTS_PER_MARGIN = 70.0


# ---------- Season boundaries ----------
# MLB seasons run entirely within one calendar year (unlike NFL/CFB, which
# both cross a Jan 1 boundary): regular season late March/April through
# early October. Verified directly against this dataset - every one of the
# 83,228 rows' dates fall within a single calendar year per season, and the
# postseason (which WOULD cross into November) is not present in these game
# logs at all (Retrosheet's regular-season game log files stop after the
# last regular-season game each year - confirmed by verify.py, which prints
# the actual min/max date per season and finds no November/December dates
# and no games logged as postseason series). So season_for_date is simply
# the calendar year - no month-based cutover logic needed, unlike NFL/CFB.
def season_for_date(dt) -> int:
    return dt.year


# ---------- Franchise continuity across relocations/rebrands ----------
# Built by cross-checking which 3-letter Retrosheet team codes appear in
# which date ranges (see verify.py's printed per-code date-range table) -
# not assumed from general baseball knowledge. The data shows exactly four
# clean code hand-offs (old code's last game immediately precedes new code's
# first game, same season boundary, no overlap) and everything else keeps
# one stable code across renames that didn't change the on-field
# organization's Retrosheet identity:
#
#   CAL -> ANA : California Angels -> Anaheim Angels franchise move, 1997.
#                CAL's last game is 1996-09-29; ANA's first is 1997-04-02.
#                (The LATER renames - Anaheim Angels of Anaheim in 2005,
#                Los Angeles Angels in 2016 - did NOT change the Retrosheet
#                code; "ANA" is used continuously 1997-2025 in this data, so
#                nothing further is needed for those.)
#   MON -> WAS : Montreal Expos -> Washington Nationals relocation, 2005.
#                MON's last game is 2004-10-03; WAS's first is 2005-04-04.
#   FLO -> MIA : Florida Marlins -> Miami Marlins rename/rebrand, 2012.
#                FLO's last game is 2011-09-28; MIA's first is 2012-04-04.
#   OAK -> ATH : Oakland Athletics -> Athletics (Sacramento) relocation,
#                2025. OAK's last game is 2024-09-29; ATH's first is
#                2025-03-27.
#
# Explicitly NOT a code change, verified by the SAME code appearing
# continuously straight through the rename with no gap/handoff:
#   - Tampa Bay Devil Rays -> Tampa Bay Rays (2008 rename): code stays TBA
#     continuously 1998-2025.
#   - Cleveland Indians -> Cleveland Guardians (2022 rename): code stays CLE
#     continuously 1990-2025.
#
# 34 distinct codes appear in the raw data across 1990-2025; collapsing the
# four hand-offs above resolves to exactly 30 franchises, matching the
# actual number of MLB organizations that have existed in this window
# (28 pre-1993, +COL/FLO in 1993, +ARI/TBA in 1998 = 30).
FRANCHISE_CANONICAL = {
    "CAL": "ANA",
    "MON": "WAS",
    "FLO": "MIA",
    "OAK": "ATH",
}


def canonical_team(code: str) -> str:
    return FRANCHISE_CANONICAL.get(code, code)


# ---------- ESPN display-name resolution (live harness only) ----------
# ESPN's scoreboard API returns full display names ("New York Yankees"), a
# completely different namespace from `canonical_team`'s input (always
# already a Retrosheet-style code — used throughout loader.py,
# starting_pitcher.py, odds_loader.py, the research_*.py modules). This is
# the ESPN-name -> Retrosheet-code bridge, used only by the live
# forward-test harness (core/matchup.py, via `canonical_franchise`, which it
# prefers over `canonical_team` when both exist) to score a brand-new ESPN
# game against the historical model. Deliberately separate from
# FRANCHISE_CANONICAL above — that dict resolves Retrosheet-code churn
# across historical relocations, this one resolves a different data
# source's naming convention into the same code space.
ESPN_NAME_TO_CODE = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHN", "Chicago White Sox": "CHA",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCA",
    "Los Angeles Angels": "ANA", "Los Angeles Dodgers": "LAN", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYN",
    "New York Yankees": "NYA", "Athletics": "ATH", "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDN",
    "San Francisco Giants": "SFN", "Seattle Mariners": "SEA", "St. Louis Cardinals": "SLN",
    "Tampa Bay Rays": "TBA", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WAS",
}


def canonical_franchise(espn_display_name: str) -> str:
    """ESPN display name -> this module's Retrosheet-style franchise code.
    Falls back to running the name through `canonical_team` unchanged if
    it's not a recognized ESPN name, so an unexpected input degrades to a
    harmless no-op lookup miss (caller defaults to ELO_START_RATING) rather
    than a crash."""
    return ESPN_NAME_TO_CODE.get(espn_display_name, canonical_team(espn_display_name))


# Reverse of ESPN_NAME_TO_CODE — for display purposes only (e.g. the Ratings
# tab), since this dataset's own `home_team`/`away_team` are Retrosheet codes
# with no display-name column anywhere to fall back to (unlike NFL/NHL,
# whose raw data already carries a real display name). Two ESPN names collide
# on ATH ("Athletics" post-2025 relocation, "Oakland Athletics" before it) —
# dict construction order keeps whichever is defined LAST in
# ESPN_NAME_TO_CODE above, which is "Oakland Athletics"; overridden
# explicitly below to prefer the current name instead.
CODE_TO_DISPLAY_NAME = {code: name for name, code in ESPN_NAME_TO_CODE.items()}
CODE_TO_DISPLAY_NAME["ATH"] = "Athletics"


def franchise_display_name(code: str) -> str:
    """This module's franchise code -> a real display name, for the UI.
    Falls back to the raw code unchanged if it's not recognized (a franchise
    that predates ESPN's current team list, e.g. an old FRANCHISE_CANONICAL
    source code) rather than crashing."""
    return CODE_TO_DISPLAY_NAME.get(code, code)
