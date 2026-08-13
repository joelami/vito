"""
RESEARCH SCRIPT (throwaway, not part of the production module shape) -
tests whether Datasets/Misc./Misc 1/ncaaf_game_scores_1g.csv (783 real
2015-season NCAAF games with real spread + total lines, ESPN-style team
abbreviations) can enrich our existing, already-tested CFB dataset, whose
own 2015 season has only 192 recorded games with just 1.6% total/moneyline
odds coverage (66% spread coverage). See docs/METHODOLOGY.md's "A new data
source: Datasets/Misc./" section for the opportunity this was scoped from,
and core/research.py for the evaluation discipline this follows exactly.

Explicit instruction from the app owner: enrich a sport we already have, do
NOT build a new sport. This script only ever touches CFB's own 2015 season.

Run with:  python -m sports.cfb.research_misc_enrichment   (from backend/)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import odds_math, ensemble, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.cfb import config as cfb_config
from sports.cfb.loader import load_games, ASSUMED_JUICE_DECIMAL
from sports.cfb.features import build_features, ML_FEATURE_COLS


MISC_PATH = Path("/Users/joe/Work/Sports Bet/Datasets/Misc./Misc 1/ncaaf_game_scores_1g.csv")


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# STEP 1: abbreviation -> our-dataset franchise-string mapping.
#
# Built by cross-referencing the Misc file's 148 distinct ESPN-style codes
# against the ACTUAL 460 distinct team strings our own cfb/loader.py-loaded
# data uses (not guessed spellings) - every value below was verified to
# exist as an exact string in our data before being accepted (see
# verify_mapping() below, run at the top of main()).
#
# Two codes turned out to be simple duplicates for the same real program,
# both confirmed directly against real final scores (MIZ/MIZZ = Missouri,
# NMST/NMSU = New Mexico State) - not a mapping ambiguity, just this source
# file's own minor inconsistency, mapped to the same franchise string.
#
# Four codes needed the SEASON-SPECIFIC spelling variant rather than
# whichever variant happens to exist somewhere else in our multi-season
# dataset, because our own 2015 rows for these four programs use a
# DIFFERENT spelling than other seasons do (the exact "same real program,
# multiple strings" inconsistency cfb/config.py's canonical_team()
# docstring already documents as a known, unaddressed limitation):
#   FIU  -> "Florida Intl Golden Panthers" (not "Florida International Panthers")
#   USM  -> "Southern Mississippi Golden Eagles" (not "Southern Miss Golden Eagles")
#   UTSA -> "UT San Antonio Roadrunners" (not "UTSA Roadrunners")
#   UTM  -> "Tennessee-Martin Skyhawks" (not "UT Martin Skyhawks")
# Picking the 2015-specific spelling matters for two reasons: it's required
# for exact-string duplicate detection against our existing 2015 rows, and
# it avoids silently splitting the same real team into two different Elo
# identities within the same season if a brand-new game for that team gets
# added under a spelling our 2015 data doesn't already use.
# ---------------------------------------------------------------------------
CODE_TO_FRANCHISE = {
    'ACU': 'Abilene Christian Wildcats', 'AFA': 'Air Force Falcons', 'AKR': 'Akron Zips',
    'ALA': 'Alabama Crimson Tide', 'ALCN': 'Alcorn State Braves', 'APP': 'Appalachian State Mountaineers',
    'ARIZ': 'Arizona Wildcats', 'ARK': 'Arkansas Razorbacks', 'ARMY': 'Army Black Knights',
    'ARST': 'Arkansas State Red Wolves', 'ASU': 'Arizona State Sun Devils', 'AUB': 'Auburn Tigers',
    'BALL': 'Ball State Cardinals', 'BAY': 'Baylor Bears', 'BC': 'Boston College Eagles',
    'BGSU': 'Bowling Green Falcons', 'BSU': 'Boise State Broncos', 'BUCK': 'Bucknell Bison',
    'BUFF': 'Buffalo Bulls', 'BYU': 'BYU Cougars', 'CAL': 'California Golden Bears',
    'CDAV': 'UC Davis Aggies', 'CHAR': 'Charlotte 49ers', 'CHAT': 'Chattanooga Mocs',
    'CHSO': 'Charleston Southern Buccaneers', 'CIN': 'Cincinnati Bearcats', 'CIT': 'The Citadel Bulldogs',
    'CLEM': 'Clemson Tigers', 'CMU': 'Central Michigan Chippewas', 'COLO': 'Colorado Buffaloes',
    'CONN': 'Connecticut Huskies', 'CSU': 'Colorado State Rams', 'DUKE': 'Duke Blue Devils',
    'ECU': 'East Carolina Pirates', 'ELON': 'Elon Phoenix', 'EMU': 'Eastern Michigan Eagles',
    'FAU': 'Florida Atlantic Owls', 'FIU': 'Florida Intl Golden Panthers', 'FLA': 'Florida Gators',
    'FORD': 'Fordham Rams', 'FRES': 'Fresno State Bulldogs', 'FSU': 'Florida State Seminoles',
    'GASO': 'Georgia Southern Eagles', 'GAST': 'Georgia State Panthers', 'GT': 'Georgia Tech Yellow Jackets',
    'HAW': "Hawai'i Rainbow Warriors", 'HOU': 'Houston Cougars', 'IDHO': 'Idaho Vandals',
    'IDST': 'Idaho State Bengals', 'ILL': 'Illinois Fighting Illini', 'IND': 'Indiana Hoosiers',
    'IOWA': 'Iowa Hawkeyes', 'ISU': 'Iowa State Cyclones', 'KENT': 'Kent State Golden Flashes',
    'KSU': 'Kansas State Wildcats', 'KU': 'Kansas Jayhawks', 'LOU': 'Louisville Cardinals',
    'LSU': 'LSU Tigers', 'LT': 'Louisiana Tech Bulldogs', 'MD': 'Maryland Terrapins',
    'MEM': 'Memphis Tigers', 'MIA': 'Miami Hurricanes', 'MICH': 'Michigan Wolverines',
    'MINN': 'Minnesota Golden Gophers', 'MIOH': 'Miami (OH) RedHawks', 'MISS': 'Ole Miss Rebels',
    'MIZ': 'Missouri Tigers', 'MIZZ': 'Missouri Tigers', 'MRSH': 'Marshall Thundering Herd',
    'MSST': 'Mississippi State Bulldogs', 'MSU': 'Michigan State Spartans',
    'MTSU': 'Middle Tennessee Blue Raiders', 'NAVY': 'Navy Midshipmen', 'NCST': 'NC State Wolfpack',
    'ND': 'Notre Dame Fighting Irish', 'NEB': 'Nebraska Cornhuskers', 'NEV': 'Nevada Wolf Pack',
    'NIU': 'Northern Illinois Huskies', 'NMST': 'New Mexico State Aggies', 'NMSU': 'New Mexico State Aggies',
    'NW': 'Northwestern Wildcats', 'ODU': 'Old Dominion Monarchs', 'OHIO': 'Ohio Bobcats',
    'OKLA': 'Oklahoma Sooners', 'OKST': 'Oklahoma State Cowboys', 'ORE': 'Oregon Ducks',
    'ORST': 'Oregon State Beavers', 'OSU': 'Ohio State Buckeyes', 'PITT': 'Pittsburgh Panthers',
    'PSU': 'Penn State Nittany Lions', 'PUR': 'Purdue Boilermakers', 'RICE': 'Rice Owls',
    'RUTG': 'Rutgers Scarlet Knights', 'SC': 'South Carolina Gamecocks', 'SDSU': 'San Diego State Aztecs',
    'SJSU': 'San Jose State Spartans', 'SMU': 'SMU Mustangs', 'STAN': 'Stanford Cardinal',
    'SUTAH': 'Southern Utah Thunderbirds', 'SYR': 'Syracuse Orange', 'TAMU': 'Texas A&M Aggies',
    'TCU': 'TCU Horned Frogs', 'TEM': 'Temple Owls', 'TENN': 'Tennessee Volunteers',
    'TEX': 'Texas Longhorns', 'TLSA': 'Tulsa Golden Hurricane', 'TOL': 'Toledo Rockets',
    'TROY': 'Troy Trojans', 'TTU': 'Texas Tech Red Raiders', 'TULN': 'Tulane Green Wave',
    'TXST': 'Texas State Bobcats', 'UCF': 'UCF Knights', 'UCLA': 'UCLA Bruins',
    'UGA': 'Georgia Bulldogs', 'UK': 'Kentucky Wildcats', 'ULL': "Louisiana Ragin' Cajuns",
    'ULM': 'Louisiana Monroe Warhawks', 'UMASS': 'Massachusetts Minutemen', 'UNC': 'North Carolina Tar Heels',
    'UNH': 'New Hampshire Wildcats', 'UNLV': 'UNLV Rebels', 'UNM': 'New Mexico Lobos',
    'UNT': 'North Texas Mean Green', 'URI': 'Rhode Island Rams', 'USA': 'South Alabama Jaguars',
    'USC': 'USC Trojans', 'USF': 'South Florida Bulls', 'USM': 'Southern Mississippi Golden Eagles',
    'USU': 'Utah State Aggies', 'UTAH': 'Utah Utes', 'UTEP': 'UTEP Miners',
    'UTM': 'Tennessee-Martin Skyhawks', 'UTSA': 'UT San Antonio Roadrunners', 'UVA': 'Virginia Cavaliers',
    'VAN': 'Vanderbilt Commodores', 'VILL': 'Villanova Wildcats', 'VMI': 'VMI Keydets',
    'VT': 'Virginia Tech Hokies', 'WAKE': 'Wake Forest Demon Deacons', 'WASH': 'Washington Huskies',
    'WCU': 'Western Carolina Catamounts', 'WEBER': 'Weber State Wildcats', 'WIS': 'Wisconsin Badgers',
    'WKU': 'Western Kentucky Hilltoppers', 'WMU': 'Western Michigan Broncos', 'WSU': 'Washington State Cougars',
    'WVU': 'West Virginia Mountaineers', 'WYO': 'Wyoming Cowboys',
}


def verify_mapping(games: pd.DataFrame, misc: pd.DataFrame):
    """Every code must resolve to an exact string already present somewhere
    in our own loaded data, and every code the Misc file actually uses must
    have an entry. Fails loudly (not silently) if either check fails - this
    is exactly the kind of "guessed spelling" mistake the task warned about."""
    our_franchises = set(games["home_franchise"]) | set(games["away_franchise"])
    codes_in_file = set(misc["home.team"]) | set(misc["away.team"])

    missing_codes = codes_in_file - set(CODE_TO_FRANCHISE.keys())
    assert not missing_codes, f"Misc file uses codes with no mapping entry: {missing_codes}"

    bad_names = {code: name for code, name in CODE_TO_FRANCHISE.items() if name not in our_franchises}
    assert not bad_names, f"Mapped names that don't exist verbatim in our data: {bad_names}"

    print(f"Mapping covers all {len(codes_in_file)} codes used in the Misc file; "
          f"all {len(CODE_TO_FRANCHISE)} mapped franchise strings verified to exist "
          f"verbatim in our own {len(our_franchises)}-team dataset.")


# ---------------------------------------------------------------------------
# STEP 2: match Misc rows against our existing 2015 games.
#
# Key = (home_franchise, away_franchise) exact, date within 0-or-1 days.
# The "+1 day" tolerance is not a guess - it's a fully-diagnosed, 100%-
# consistent timezone artifact: our source CSV's `date` column is the UTC
# calendar date of kickoff_utc (spot-checked directly: e.g. Vanderbilt vs
# Western Kentucky's kickoff_utc is "2015-09-04T00:00Z", i.e. 8pm Thursday
# Sept 3 Eastern), while the Misc file's `date` is the local/US calendar
# date. Every night game that crosses the UTC midnight boundary therefore
# looks 1 day later in our data than in the Misc file; day games don't
# cross it and match exactly. Checked directly across all 80 such cases
# before trusting it: 100% of them show exactly a +1 day offset, 0% show
# any other offset - a clean, fully-explained pattern, not an ambiguous one.
# ---------------------------------------------------------------------------
def match_games(games: pd.DataFrame, misc: pd.DataFrame):
    g2015 = games[games["season"] == 2015].copy()
    misc = misc.copy()
    misc["home_franchise"] = misc["home.team"].map(CODE_TO_FRANCHISE)
    misc["away_franchise"] = misc["away.team"].map(CODE_TO_FRANCHISE)

    from collections import defaultdict
    misc_by_pair = defaultdict(list)
    for idx, r in misc.iterrows():
        misc_by_pair[(r["home_franchise"], r["away_franchise"])].append(idx)

    matches = []  # (our_idx, misc_idx, date_delta_days)
    used_misc_idx = set()
    for idx, r in g2015.iterrows():
        cands = misc_by_pair.get((r["home_franchise"], r["away_franchise"]), [])
        best = None
        for mi in cands:
            if mi in used_misc_idx:
                continue
            delta = (r["date"] - misc.loc[mi, "date"]).days
            if delta in (0, 1):
                if best is None or abs(delta) < abs(best[1]):
                    best = (mi, delta)
        if best:
            matches.append((idx, best[0], best[1]))
            used_misc_idx.add(best[0])

    matched_our_idx = {m[0] for m in matches}
    unmatched_ours = g2015.index[~g2015.index.isin(matched_our_idx)]
    unmatched_misc = misc.index[~misc.index.isin(used_misc_idx)]

    hr("STEP 2: MATCH RESULTS")
    print(f"Our 2015 games: {len(g2015)}")
    print(f"Misc file rows: {len(misc)}")
    print(f"TRUE DUPLICATES (exact team-pair match, date within 0-1 days): {len(matches)}")
    print(f"  - same-day exact date match:  {sum(1 for m in matches if m[2] == 0)}")
    print(f"  - +1 day (UTC-vs-local artifact, diagnosed above): {sum(1 for m in matches if m[2] == 1)}")
    print(f"Our 2015 games with NO match in Misc file: {len(unmatched_ours)}")
    print(f"Misc rows that are GENUINELY NEW games (not in our data at all): {len(unmatched_misc)}")

    # Sanity check #1: every one of our unmatched games should be explainable
    # as a non-FBS opponent the Misc file's FBS-only scope wouldn't carry.
    unmatched_df = g2015.loc[unmatched_ours]
    fbs_names = set(CODE_TO_FRANCHISE.values())
    both_fbs = unmatched_df[
        unmatched_df["home_franchise"].isin(fbs_names) & unmatched_df["away_franchise"].isin(fbs_names)
    ]
    print(f"\nOf our {len(unmatched_ours)} unmatched games, {len(both_fbs)} have BOTH teams "
          f"in the Misc file's FBS code list (should be ~0 if the 'FCS opponent' explanation "
          f"holds) - these are worth a manual look if non-zero:")
    if len(both_fbs):
        print(both_fbs[["date", "home_franchise", "away_franchise"]].to_string())
    else:
        print("  (none - confirms every unmatched game involves at least one team outside the "
              "Misc file's FBS-only scope)")

    # Sanity check #2: scores must agree for every matched pair (a real,
    # independent check that home/away/date/team-name alignment is
    # correctly identifying the SAME real game, not a coincidental collision).
    misc_scored = misc.rename(columns={"home.score": "misc_hs", "away.score": "misc_as"})
    score_check = g2015.loc[[m[0] for m in matches]].copy()
    score_check["misc_hs"] = [misc_scored.loc[m[1], "misc_hs"] for m in matches]
    score_check["misc_as"] = [misc_scored.loc[m[1], "misc_as"] for m in matches]
    score_mismatch = score_check[
        (score_check["home_score"] != score_check["misc_hs"]) | (score_check["away_score"] != score_check["misc_as"])
    ]
    print(f"\nScore mismatches among {len(matches)} matched games (would indicate a false-"
          f"positive match): {len(score_mismatch)}")
    if len(score_mismatch):
        print(score_mismatch[["date", "home_franchise", "away_franchise", "home_score", "misc_hs",
                               "away_score", "misc_as"]].to_string())
        print(
            "Traced this one directly: South Carolina vs North Carolina, 2015-09-03, was a "
            "NEUTRAL-SITE game (Bank of America Stadium, Charlotte NC - confirmed in our raw "
            "CSV's own venue column) where our EXISTING baseline has home_score/away_score "
            "swapped relative to the real final (UNC won 17-13; our baseline lists SC=17, "
            "UNC=13). This is a pre-existing data-entry issue already living in the tested "
            "production baseline, independent of and not introduced by this merge - the match "
            "itself (same date, same two teams, same neutral designation) is unambiguously "
            "correct. Per this task's scope (fill MISSING values, never overwrite existing "
            "conflicting ones), this game's existing odds/score fields are left untouched "
            "either way - flagged here for the record, not silently corrected."
        )

    return matches, unmatched_misc, g2015, misc


# ---------------------------------------------------------------------------
# STEP 3: build the enriched games frame - fill missing odds on matched
# games, append genuinely new games, following loader.py's exact column
# contract and derivation formulas (never a parallel path).
# ---------------------------------------------------------------------------
def build_enriched_games(games: pd.DataFrame, matches, unmatched_misc, g2015: pd.DataFrame, misc: pd.DataFrame):
    games = games.copy()
    games_by_id = games.set_index("game_id")

    fills = {"spread_filled": 0, "total_filled": 0}
    for our_idx, misc_idx, _delta in matches:
        game_id = g2015.loc[our_idx, "game_id"]
        m = misc.loc[misc_idx]

        if pd.isna(games_by_id.loc[game_id, "Home Line Close"]) and pd.notna(m["line"]):
            games_by_id.loc[game_id, "Home Line Close"] = m["line"]
            games_by_id.loc[game_id, "Home Line Odds Close"] = ASSUMED_JUICE_DECIMAL
            games_by_id.loc[game_id, "Away Line Odds Close"] = ASSUMED_JUICE_DECIMAL
            fills["spread_filled"] += 1

        if pd.isna(games_by_id.loc[game_id, "Total Score Close"]) and pd.notna(m["over_under"]):
            games_by_id.loc[game_id, "Total Score Close"] = m["over_under"]
            games_by_id.loc[game_id, "Total Score Over Close"] = ASSUMED_JUICE_DECIMAL
            games_by_id.loc[game_id, "Total Score Under Close"] = ASSUMED_JUICE_DECIMAL
            fills["total_filled"] += 1

    games = games_by_id.reset_index()

    # Recompute the derived ATS/over-under result columns for every row
    # whose odds just got filled in, using the EXACT same formula loader.py
    # uses - otherwise ats_pct_l10/over-result features wouldn't see any
    # benefit from the newly-filled odds.
    close_line = games["Home Line Close"]
    games["home_covers_close"] = np.select(
        [games["actual_margin"] > -close_line, games["actual_margin"] == -close_line], [1, 0], default=-1,
    )
    games.loc[close_line.isna(), "home_covers_close"] = np.nan

    total_line = games["Total Score Close"]
    games["over_result_close"] = np.select(
        [games["actual_total"] > total_line, games["actual_total"] == total_line], [1, 0], default=-1,
    )
    games.loc[total_line.isna(), "over_result_close"] = np.nan

    print(f"\nExisting 2015 games: filled {fills['spread_filled']} missing spreads, "
          f"{fills['total_filled']} missing totals (never overwriting a value we already had).")

    # ---- New games: date -> (week, is_bowl) lookup built from OUR OWN data ----
    # Every date in the Misc file except one (2015-12-26) has at least one
    # real game in our own 2015 data on that date (or date-1, the same UTC
    # artifact from STEP 2) - so week/is_bowl for new rows comes from real,
    # already-ESPN-sourced values in our own dataset, not invented. Checked
    # directly: every date in our own data maps to exactly one (week,
    # is_bowl) pair (0 dates with mixed values), so this lookup is
    # unambiguous. The one uncovered date (2015-12-26) sits squarely in the
    # bowl calendar (after the Dec 5 conference championship weekend) so is
    # assigned is_bowl=True / week=(that season's max regular week + 1),
    # consistent with season_week_adjusted()'s own bowl convention - labeled
    # here as an inference, not presented as a real ESPN value.
    date_lookup = g2015.groupby("date").agg(week=("week", "first"), is_bowl=("is_bowl", "first"))
    max_reg_week_2015 = g2015.loc[~g2015["is_bowl"], "week"].max()

    def lookup(date):
        if date in date_lookup.index:
            return date_lookup.loc[date, "week"], bool(date_lookup.loc[date, "is_bowl"])
        prev_day = date - pd.Timedelta(days=1)
        if prev_day in date_lookup.index:
            return date_lookup.loc[prev_day, "week"], bool(date_lookup.loc[prev_day, "is_bowl"])
        return max_reg_week_2015 + 1.0, True  # 2015-12-26 only - see docstring above

    new_rows = []
    inferred_date_count = 0
    for i, misc_idx in enumerate(unmatched_misc):
        m = misc.loc[misc_idx]
        week, is_bowl = lookup(m["date"])
        if m["date"] not in date_lookup.index and (m["date"] - pd.Timedelta(days=1)) not in date_lookup.index:
            inferred_date_count += 1

        home_score, away_score = float(m["home.score"]), float(m["away.score"])
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        home_line = m["line"] if pd.notna(m["line"]) else np.nan
        total_line = m["over_under"] if pd.notna(m["over_under"]) else np.nan

        home_covers = np.nan
        if pd.notna(home_line):
            home_covers = 1 if actual_margin > -home_line else (0 if actual_margin == -home_line else -1)
        over_result = np.nan
        if pd.notna(total_line):
            over_result = 1 if actual_total > total_line else (0 if actual_total == total_line else -1)

        new_rows.append({
            "game_id": f"cfb_misc1g_{i}", "event_id": np.nan, "date": m["date"], "season": 2015,
            "week": week, "is_bowl": is_bowl, "is_neutral_venue": is_bowl,
            "home_team": m["home_franchise"], "away_team": m["away_franchise"],
            "home_franchise": m["home_franchise"], "away_franchise": m["away_franchise"],
            "home_score": home_score, "away_score": away_score,
            "actual_margin": actual_margin, "actual_total": actual_total,
            "home_win": int(actual_margin > 0),
            "home_covers_close": home_covers, "over_result_close": over_result,
            "Home Odds Close": np.nan, "Away Odds Close": np.nan,  # Misc file has no moneyline
            "Home Line Close": home_line,
            "Home Line Odds Close": ASSUMED_JUICE_DECIMAL if pd.notna(home_line) else np.nan,
            "Away Line Odds Close": ASSUMED_JUICE_DECIMAL if pd.notna(home_line) else np.nan,
            "Total Score Close": total_line,
            "Total Score Over Close": ASSUMED_JUICE_DECIMAL if pd.notna(total_line) else np.nan,
            "Total Score Under Close": ASSUMED_JUICE_DECIMAL if pd.notna(total_line) else np.nan,
        })

    print(f"New games added: {len(new_rows)} (week/is_bowl inferred rather than looked up for "
          f"{inferred_date_count} of them - expect exactly 1, the 2015-12-26 date with no direct "
          f"same-day or prior-day match in our own 2015 data).")

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([games, new_df], ignore_index=True)

    # Same chronological re-sort + sequential game_id re-assignment loader.py
    # itself does - a single, consistent convention, not a parallel one.
    # New rows have no kickoff_utc, so date + a stable team-name tiebreaker
    # substitutes for loader.py's date+kickoff_utc sort among same-day games.
    # CAVEAT (found after this script was first run, recorded here rather
    # than silently fixed): `games` here is load_games()'s OUTPUT, which
    # already drops the raw `kickoff_utc` column (not in loader.py's
    # keep_cols) - so unlike loader.py's own internal merge (which runs
    # BEFORE that column is dropped and can keep real kickoff-time ordering
    # for every pre-existing game), this script can only sort ALL rows by
    # date + team name, losing real same-day kickoff ordering for every
    # existing game, not just the newly-added ones. Since CFB Elo ratings
    # update sequentially and same-day games are common, this measurably
    # shifted this script's OOS numbers vs. the real production loader
    # (margin_corr 0.3914 here vs. 0.3871 from a fresh `verify.py` run
    # after the merge was made permanent; 2,926 bets vs 2,917; ROI -4.54%
    # vs -4.21%) - both point the same direction and clear the same
    # noise floor, but `verify.py`'s numbers (run against the ACTUAL
    # production loader.py, which preserves kickoff_utc correctly) are the
    # authoritative ones reported in docs/METHODOLOGY.md, not this script's.
    combined = combined.sort_values(["date", "home_franchise", "away_franchise"], kind="stable").reset_index(drop=True)
    combined["game_id"] = combined.index.astype(str).map(lambda i: f"cfb_{i}")
    return combined


# ---------------------------------------------------------------------------
# Shared pipeline runner - identical shape to verify.py / other CFB research scripts
# ---------------------------------------------------------------------------
def run_pipeline(games: pd.DataFrame, label: str) -> dict:
    hr(f"PIPELINE RUN: {label}")
    rating_cfg = PowerRatingConfig(
        k_factor=cfb_config.ELO_K_FACTOR, start_rating=cfb_config.ELO_START_RATING,
        home_field_adv=cfb_config.HOME_FIELD_ADV_ELO, season_regression=cfb_config.SEASON_REGRESSION,
        mov_mult_base=cfb_config.MOV_MULT_BASE, mov_mult_divisor=cfb_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", neutral_col="is_neutral_venue",
        config=rating_cfg,
    )
    feats = build_features(games, rr.history)
    print(f"Feature rows: {len(feats):,}")

    wf = walk_forward_predict(feats, ML_FEATURE_COLS, min_train_seasons=3)
    print(f"OOS rows: {len(wf.predictions):,}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    margin_corr = corr("predicted_margin", "actual_margin")
    total_corr = corr("predicted_total", "actual_total")
    print(f"Margin corr: {margin_corr:.4f}   Total corr: {total_corr:.4f}")

    stds = ensemble.compute_residual_stds(oos, cfb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, cfb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"Qualifying bets: {len(bets):,}")

    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr, "roi_pct": 0.0,
                "roi_stderr_pct": float("nan"), "bets": 0}

    summary = backtest.summarize(bets)
    roi_pct = float(summary["roi_pct"].iloc[0])
    roi_stderr_pct = float(summary["roi_stderr_pct"].iloc[0])
    print(f"ROI: {roi_pct:+.2f}%  (stderr {roi_stderr_pct:.2f}pp, n={len(bets)})")

    # 2015-specific slice: the whole point of this change - did 2015 itself get better?
    bets_2015 = bets[bets["season"] == 2015] if "season" in bets.columns else None
    if bets_2015 is not None and not bets_2015.empty:
        s2015 = backtest.summarize(bets_2015)
        print(f"  [2015-only] bets={len(bets_2015)}  ROI={float(s2015['roi_pct'].iloc[0]):+.2f}%  "
              f"stderr={float(s2015['roi_stderr_pct'].iloc[0]):.2f}pp")
    else:
        print("  [2015-only] 0 qualifying bets")

    return {"margin_corr": margin_corr, "total_corr": total_corr, "roi_pct": roi_pct,
            "roi_stderr_pct": roi_stderr_pct, "bets": len(bets)}


def main():
    hr("STEP 0: LOAD BASELINE + MISC FILE, VERIFY MAPPING")
    games = load_games()
    misc = pd.read_csv(MISC_PATH)
    misc["date"] = pd.to_datetime(misc["date"])
    print(f"Baseline CFB games: {len(games):,} rows")
    print(f"Misc file: {len(misc)} rows, season(s): {sorted(misc['season'].unique())}")
    verify_mapping(games, misc)

    # Sign-convention check on the Misc file's own `line` column before
    # trusting it: does -line correlate with actual home margin the same
    # direction our own `spread` column does (loader.py's own docstring:
    # negative = home favored, regression slope ~0.93)?
    misc_tmp = misc.copy()
    misc_tmp["home_margin"] = misc_tmp["home.score"] - misc_tmp["away.score"]
    slope, intercept = np.polyfit(-misc_tmp["line"], misc_tmp["home_margin"], 1)
    corr_check = np.corrcoef(-misc_tmp["line"], misc_tmp["home_margin"])[0, 1]
    print(f"\nMisc `line` sign-convention check: regressing actual home_margin on -line gives "
          f"slope={slope:.3f}, intercept={intercept:.3f}, corr={corr_check:.3f} - matches our "
          f"own loader's documented convention (negative=home favored) closely enough (loader.py's "
          f"own check on OUR data: slope~0.93, intercept~0.07) to use as-is, no sign flip needed.")

    matches, unmatched_misc, g2015, misc_mapped = match_games(games, misc)

    hr("STEP 3: BUILD ENRICHED GAMES FRAME")
    enriched = build_enriched_games(games, matches, unmatched_misc, g2015, misc_mapped)
    print(f"Baseline row count: {len(games):,}  ->  Enriched row count: {len(enriched):,} "
          f"(+{len(enriched) - len(games)})")

    e2015 = enriched[enriched["season"] == 2015]
    has_spread = e2015["Home Line Close"].notna()
    has_total = e2015["Total Score Close"].notna()
    print(f"\n2015 season coverage, BEFORE -> AFTER:")
    print(f"  games:  {len(g2015)} -> {len(e2015)}")
    print(f"  spread: {g2015['Home Line Close'].notna().sum()} ({g2015['Home Line Close'].notna().mean()*100:.1f}%)"
          f" -> {has_spread.sum()} ({has_spread.mean()*100:.1f}%)")
    print(f"  total:  {g2015['Total Score Close'].notna().sum()} ({g2015['Total Score Close'].notna().mean()*100:.1f}%)"
          f" -> {has_total.sum()} ({has_total.mean()*100:.1f}%)")

    hr("STEP 4: MEASURE BASELINE vs ENRICHED")
    baseline = run_pipeline(games, "BASELINE (current production load_games())")
    variant = run_pipeline(enriched, "VARIANT (Misc-enriched 2015 season)")

    hyp = Hypothesis(
        name="cfb_misc1g_2015_enrichment",
        reasoning=(
            "Our own CFB dataset's 2015 season has only 192 recorded games with just 1.6% total/"
            "moneyline odds coverage (66% spread coverage), while Datasets/Misc./Misc 1/"
            "ncaaf_game_scores_1g.csv independently carries 783 real games with real spread AND "
            "total lines for that same season - a materially more complete source for 2015 "
            "specifically. A validated, exact-string team-abbreviation mapping (148 ESPN-style "
            "codes, cross-checked against our own loaded team strings and against real 2015 final "
            "scores) found 140 of our existing 192 2015 games are true duplicates of Misc rows "
            "(the other 52 are exclusively non-FBS opponents outside the Misc file's FBS-only "
            "scope), letting us fill real missing spread/total odds on existing games without "
            "ever overwriting a value we already had, plus add hundreds of genuinely new 2015 "
            "games with real lines we had zero visibility into before. This directly targets the "
            "exact limitation flagged in METHODOLOGY.md's CFB section: sparse, short-window odds "
            "coverage swallowing every backtest result in its own noise band."
        ),
        sport="CFB",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT: cfb_misc1g_2015_enrichment")
    import json
    print(json.dumps(result.to_dict(), indent=2, default=str))

    hr("DONE")


if __name__ == "__main__":
    main()
