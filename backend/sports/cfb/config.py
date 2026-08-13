"""
CFB-specific constants. Mirrors sports/nfl/config.py's structure so the two
sports share the exact same downstream pipeline shape. Every constant that's
"tuned" rather than derived lives here so it's obvious what's a modeling
assumption vs. a fact pulled from data.

Tuning note: these Elo constants are NOT copy-pasted from NFL. College
football runs ~130 FBS teams (plus FCS/D-II opponents some seasons) with a
much wider talent gap than the 32-team NFL, so home margins and blowouts run
far larger. On this dataset (2004-2025, completed=1 games): home teams win
58.9% of games, average home margin is +5.6 points (NFL is roughly +2 to
+2.5), and the std-dev of margin is ~21.6 points (NFL is roughly 13-14).

Selection process: a small grid search over (k_factor, home_field_adv,
season_regression, mov_mult_base, mov_mult_divisor) was run, scoring each
combination by mean-absolute-error between predicted and actual home-win
rate across walk-forward calibration deciles (see cfb/verify.py section 2 -
the exact same decile-bucket check used to validate the NFL ratings). NFL's
own constants (k=20, hfa=48, regression=0.33, mov 2.2/2.2) scored a 0.047
mean decile calibration error on this data; the values below scored 0.022 -
roughly half the error - and were kept. This is a legitimate hyperparameter
search on the *rating engine's own probability calibration*, not a search
over backtest ROI/profitability, so it doesn't cross into the "tune
thresholds until it looks profitable" territory the project is explicitly
trying to avoid; see verify.py's printed calibration table for the actual
numbers behind this choice.
"""

from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent.parent / "Datasets" / "College Football" / "cfb-games.csv"

# Player-level box scores, keyed by event_id/team/opponent (see loader.py's
# load_team_game_boxscores() docstring for the join-coverage caveat: only
# ~92% of completed games get a full team-vs-team yardage match, ~95%
# get at least one side matched - partial like every other source here,
# not filtered out, just documented).
PASSING_BOX_SCORE_PATH = DATA_PATH.parent / "cfb-passing-box-scores.csv"
RUSHING_BOX_SCORE_PATH = DATA_PATH.parent / "cfb-rushing-box-scores.csv"

# 2015-season odds/game enrichment, adopted via hypothesis test
# "cfb_misc1g_2015_enrichment" (see decision_log.jsonl,
# sports/cfb/research_misc_enrichment.py, and docs/METHODOLOGY.md's dated
# CFB subsection for the full before/after numbers). Real per-game
# score+line data for the 2015 NCAAF season, independently sourced - see
# loader.py's merge_misc_2015_enrichment() for what this does to load_games().
MISC_2015_ENRICHMENT_PATH = DATA_PATH.parent.parent / "Misc." / "Misc 1" / "ncaaf_game_scores_1g.csv"

# ---------- Power rating engine ----------
# Tuned for CFB's much larger margins/blowouts and larger home-win rate
# relative to NFL; see verify.py's calibration-decile output for the check
# that motivated these vs. just reusing NFL's numbers (grid search summary
# above).
ELO_K_FACTOR = 40.0            # much faster-moving than NFL's 20 - CFB talent gaps are wider and shift more within a season (young teams, transfer portal, cupcake/blue-blood scheduling) so ratings need to separate faster
ELO_START_RATING = 1500.0
HOME_FIELD_ADV_ELO = 90.0      # nearly double NFL's 48 - CFB home-field/crowd effects and travel mismatches (small schools traveling to blue-bloods) run much bigger
SEASON_REGRESSION = 0.30       # close to NFL's 0.33 - roster turnover pressure (graduation/transfer portal) and NFL's own year-to-year uncertainty turned out to want similar regression once k/hfa were re-tuned
MOV_MULT_BASE = 2.5
MOV_MULT_DIVISOR = 3.5         # larger divisor than NFL's 2.2 to damp the multiplier for CFB's much more common 40-60 point blowouts

# Points-per-elo-point conversion, used to turn a rating differential into a
# predicted scoring margin. Empirically fit (not guessed): regressing actual
# margin on rating_diff_pre at the final tuned rating config gives a slope
# of ~0.072, i.e. ~13.8 elo points per 1 point of margin; rounded to 14.
# This is SMALLER than NFL's 25 (fewer elo points needed per margin point)
# because the larger k_factor/home_field_adv above already spread CFB
# ratings out further for the same underlying skill gap. As with NFL, this
# is just the starting scale - `ensemble.compute_residual_stds` fits the
# actual out-of-sample residual std-dev at runtime.
ELO_POINTS_PER_MARGIN = 14.0


# ---------- Season boundaries ----------
# CFB seasons run Aug -> Jan (regular season Aug-Dec, bowls/playoff into
# Jan). A game in Jan-Jul belongs to the season that started the previous
# autumn. Verified against this dataset's own `season` column: applying
# `dt.year if dt.month >= 8 else dt.year - 1` to every one of the 3,688 rows'
# `date` reproduces the provided `season` column exactly (0 mismatches) -
# including the COVID-shifted spring-2021 FCS games (Feb-May 2021 dates,
# correctly falling under season=2020). The loader trusts the CSV's own
# `season` column directly (it's already correct and handles that edge case)
# but this function is kept for interface parity with NFL and for scoring
# any future/manually-entered game that has no season column yet.
def season_for_date(dt) -> int:
    return dt.year if dt.month >= 8 else dt.year - 1


# ---------- Team identity ----------
# Deliberately NOT attempting a historical realignment/rename map. FBS
# college football has far more conference realignment, name changes, and
# program churn over 2004-2025 than the NFL's 4 relocations (e.g. this CSV
# has 321 unique home_team strings) - building and validating an exhaustive
# mapping is a real research project on its own and explicitly out of scope
# here. `canonical_team` is therefore a passthrough: each distinct team
# string in the source data is treated as its own entity for rating/feature
# purposes. Kept as a function (rather than inlining `home_team` directly)
# purely so this module's shape matches sports/nfl/config.py's
# canonical_franchise pattern and so a real mapping could be dropped in here
# later without touching loader.py or features.py.
#
# Known consequence, observed directly in this data and not fixed: the same
# real program sometimes appears under multiple strings close together in
# time, e.g. "East Tennessee St. Buccaneers" vs "East Tennessee State
# Buccaneers", "Southern Miss Golden Eagles" vs "Southern Mississippi Golden
# Eagles", "Florida Intl Golden Panthers" vs "Florida International
# Panthers", "San Jose State Spartans" vs "San José St Spartans" vs "San
# José State Spartans", and "St. Thomas (MN) Tommies" vs "St. Thomas -
# Minnesota Tommies" vs "St. Thomas-Minnesota Tommies". Each variant is
# rated as a brand-new team starting at 1500, which understates that team's
# true strength for a stretch after a naming inconsistency. Not corrected
# per scope above.
def canonical_team(display_name: str) -> str:
    return display_name


# ---------- Bowl / postseason & neutral-site approximation ----------
# The CSV has no explicit "neutral venue" flag (unlike the NFL source file's
# "Neutral Venue?" column). `season_type == 3` is this dataset's postseason
# bowl/playoff marker (600 of 3,688 rows); the overwhelming majority of those
# are played at a neutral site (bowl games, CFP games), so `is_bowl` is used
# as a neutral-site proxy. This is an approximation, not a fact pulled from
# a venue field - a handful of true "postseason" rows (e.g. some conference
# championship games hosted at a higher seed's home stadium) will be
# mislabeled neutral when they weren't. Documented here rather than silently
# assumed correct.
POSTSEASON_SEASON_TYPE = 3


def is_bowl_game(season_type) -> bool:
    return season_type == POSTSEASON_SEASON_TYPE


# ---------- Ratings display filter ----------
# This CSV has no conference/division column at all (see features.py's
# is_divisional docstring), so the Elo engine treats every opponent —
# ~134 FBS programs plus whatever FCS/D-II teams show up as a handful of
# real games' buy-game opponents — as one undivided pool. A small program
# that's simply undefeated across a thin handful of games (sometimes
# against equally small opponents it never had to prove itself against a
# real contender in) can outrank an actual top-25 blue-blood whose rating
# is built on far more games. Real, reported example: West Georgia Wolves
# (a Division II program) ranked #9, above most of the SEC/Big Ten.
#
# A games-played minimum alone doesn't fix this — West Georgia had 20
# recorded games, comfortably clearing any reasonable threshold, while a
# real blue-blood like Michigan has only 15 in this dataset (odds/game
# coverage here skews toward a roughly 2012-2022 window — see the CFB
# section of docs/METHODOLOGY.md). The actual fix is an allow-list: every
# string below is a current FBS program's exact display-name string AS IT
# APPEARS IN THIS DATASET, cross-referenced by hand against the full list
# of ~460 distinct team strings the loader produces (several real FBS
# programs appear under more than one spelling — e.g. "App State
# Mountaineers" / "Appalachian State Mountaineers" — both included so
# neither is silently dropped). Built from known current FBS membership,
# not derived from the data itself (this data has no division field to
# derive it from) — if a program moves to/from FBS this list won't
# self-update, that's a real, accepted limitation of a hand-built list
# rather than a computed one.
FBS_TEAMS = frozenset({
    "Air Force Falcons", "Akron Zips", "Alabama Crimson Tide",
    "Appalachian State Mountaineers", "App State Mountaineers",
    "Arizona State Sun Devils", "Arizona Wildcats", "Arkansas Razorbacks",
    "Arkansas State Red Wolves", "Army Black Knights", "Auburn Tigers",
    "Ball State Cardinals", "Baylor Bears", "Boise State Broncos",
    "Boston College Eagles", "Bowling Green Falcons", "Buffalo Bulls",
    "BYU Cougars", "California Golden Bears", "Central Michigan Chippewas",
    "Charlotte 49ers", "Cincinnati Bearcats", "Clemson Tigers",
    "Coastal Carolina Chanticleers", "Colorado Buffaloes", "Colorado State Rams",
    "Connecticut Huskies", "UConn Huskies", "Delaware Blue Hens", "Duke Blue Devils",
    "East Carolina Pirates", "Eastern Michigan Eagles", "Florida Atlantic Owls",
    "Florida Gators", "Florida International Panthers", "Florida Intl Golden Panthers",
    "Florida State Seminoles", "Fresno State Bulldogs", "Georgia Bulldogs",
    "Georgia Southern Eagles", "Georgia State Panthers", "Georgia Tech Yellow Jackets",
    "Hawai'i Rainbow Warriors", "Houston Cougars", "Illinois Fighting Illini",
    "Indiana Hoosiers", "Iowa Hawkeyes", "Iowa State Cyclones",
    "Jacksonville State Gamecocks", "James Madison Dukes", "Kansas Jayhawks",
    "Kansas State Wildcats", "Kennesaw State Owls", "Kent State Golden Flashes",
    "Kentucky Wildcats", "Liberty Flames", "Louisiana Monroe Warhawks",
    "UL Monroe Warhawks", "Louisiana Ragin' Cajuns", "Louisiana Tech Bulldogs",
    "Louisville Cardinals", "LSU Tigers", "Marshall Thundering Herd",
    "Maryland Terrapins", "Massachusetts Minutemen", "Memphis Tigers",
    "Miami (OH) RedHawks", "Miami Hurricanes", "Michigan State Spartans",
    "Michigan Wolverines", "Middle Tennessee Blue Raiders", "Minnesota Golden Gophers",
    "Mississippi State Bulldogs", "Missouri State Bears", "Missouri Tigers", "Navy Midshipmen",
    "NC State Wolfpack", "Nebraska Cornhuskers", "Nevada Wolf Pack",
    "New Mexico Lobos", "New Mexico State Aggies", "North Carolina Tar Heels",
    "North Texas Mean Green", "Northern Illinois Huskies", "Northwestern Wildcats",
    "Notre Dame Fighting Irish", "Ohio Bobcats", "Ohio State Buckeyes",
    "Oklahoma Sooners", "Oklahoma State Cowboys", "Old Dominion Monarchs",
    "Ole Miss Rebels", "Oregon Ducks", "Oregon State Beavers",
    "Penn State Nittany Lions", "Pittsburgh Panthers", "Purdue Boilermakers",
    "Rice Owls", "Rutgers Scarlet Knights", "Sam Houston Bearkats",
    "Sam Houston State Bearkats", "San Diego State Aztecs",
    "San José State Spartans", "San José St Spartans", "SMU Mustangs",
    "South Alabama Jaguars", "South Carolina Gamecocks", "South Florida Bulls",
    "Southern Miss Golden Eagles", "Southern Mississippi Golden Eagles",
    "Stanford Cardinal", "Syracuse Orange", "TCU Horned Frogs", "Temple Owls",
    "Tennessee Volunteers", "Texas A&M Aggies", "Texas Longhorns",
    "Texas State Bobcats", "Texas Tech Red Raiders", "Toledo Rockets",
    "Troy Trojans", "Tulane Green Wave", "Tulsa Golden Hurricane",
    "UAB Blazers", "UCF Knights", "UCLA Bruins", "UNLV Rebels", "USC Trojans",
    "UTEP Miners", "UTSA Roadrunners", "Utah State Aggies", "Utah Utes",
    "Vanderbilt Commodores", "Virginia Cavaliers", "Virginia Tech Hokies",
    "Wake Forest Demon Deacons", "Washington Huskies", "Washington State Cougars",
    "West Virginia Mountaineers", "Western Kentucky Hilltoppers",
    "Western Michigan Broncos", "Wisconsin Badgers", "Wyoming Cowboys",
})
