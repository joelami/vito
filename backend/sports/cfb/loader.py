"""
Loads and cleans Datasets/College Football/cfb-games.csv into a tidy,
one-row-per-game DataFrame ready for feature engineering. No modeling
happens here - this module's only job is turning the raw CSV into
trustworthy columns, mirroring sports/nfl/loader.py's shape exactly.

Important, already-checked facts about this source (see cfb/config.py and
the Phase-1 report for the full detail, not re-derived here):

  - Odds are American format in the raw CSV (e.g. -142, +600), unlike the
    NFL source which is already decimal. Every price is run through
    `core.odds_math.american_to_decimal` before being stored.
  - Odds coverage is sparse: only ~42% of rows have a spread, ~25% a total,
    ~24% moneylines. Most games have NO betting data. Nothing here filters
    those out - every completed game is kept (a game missing odds is still
    valid for rating/feature purposes, just not "bettable") - the caller
    (verify.py) is responsible for filtering to odds-having rows before
    treating anything as a backtest candidate.
  - There is only ONE spread value and ONE total value per game (no
    open/close/min/max like NFL), and neither has a real two-sided price
    (no separate juice for either side). Both sides of the spread and total
    are therefore assumed to be priced at standard -110 (decimal 1.9091) -
    a documented assumption, not real market data. Moneyline IS two-sided
    and real (home_moneyline/away_moneyline), so it alone gets a genuine
    devig downstream. All four odds columns are named with the "Close"
    price-point suffix (`price_point="Close"`) purely so
    core/edge_finder.py's column lookups work unchanged - there is no
    actual open-vs-close distinction in this data (see the CLV caveat in
    cfb/verify.py's output).
"""

import numpy as np
import pandas as pd

from . import config
from core import odds_math


ODDS_COLS = [
    "Home Odds Close", "Away Odds Close",
    "Home Line Close", "Home Line Odds Close", "Away Line Odds Close",
    "Total Score Close", "Total Score Over Close", "Total Score Under Close",
]

ASSUMED_JUICE_DECIMAL = odds_math.american_to_decimal(-110.0)  # 1.9090909...


def load_games() -> pd.DataFrame:
    """Returns a cleaned, chronologically-sorted DataFrame, one row per game."""
    df = pd.read_csv(config.DATA_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"]).copy()

    # completed=0 rows are incomplete/future placeholder rows in this
    # snapshot (0-0 scores, postponed/cancelled games) - drop them.
    df = df[df["completed"] == 1].copy()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    # ---------- Odds: American -> decimal, assumed juice for spread/total ----------
    for col in ["spread", "over_under", "home_moneyline", "away_moneyline"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # American odds can never legitimately be exactly 0 (minimum magnitude is
    # +/-100); one row (Baylor vs SMU, 2014-08-31) has home_moneyline=
    # away_moneyline=0.0, an obvious data-entry placeholder, not a real
    # price. Treated as missing rather than fed into american_to_decimal
    # (which would divide by zero).
    df.loc[df["home_moneyline"] == 0, "home_moneyline"] = np.nan
    df.loc[df["away_moneyline"] == 0, "away_moneyline"] = np.nan

    df["Home Odds Close"] = df["home_moneyline"].apply(
        lambda v: odds_math.american_to_decimal(v) if pd.notna(v) else np.nan
    )
    df["Away Odds Close"] = df["away_moneyline"].apply(
        lambda v: odds_math.american_to_decimal(v) if pd.notna(v) else np.nan
    )

    # `spread` is validated (see Phase-1 report) to already follow the
    # project's "Home Line" convention: negative = home favored by that many
    # points. Regressing actual_margin on -spread gives slope ~0.93,
    # intercept ~0.07 (n=1,551), and sign agrees with the real two-sided
    # home_moneyline favorite 97.7% of the time - so it's used as-is, no
    # sign flip.
    df["Home Line Close"] = df["spread"]
    df["Home Line Odds Close"] = np.where(df["Home Line Close"].notna(), ASSUMED_JUICE_DECIMAL, np.nan)
    df["Away Line Odds Close"] = np.where(df["Home Line Close"].notna(), ASSUMED_JUICE_DECIMAL, np.nan)

    df["Total Score Close"] = df["over_under"]
    df["Total Score Over Close"] = np.where(df["Total Score Close"].notna(), ASSUMED_JUICE_DECIMAL, np.nan)
    df["Total Score Under Close"] = np.where(df["Total Score Close"].notna(), ASSUMED_JUICE_DECIMAL, np.nan)

    # ---------- Team identity, season, bowl/neutral-site flag ----------
    df["home_franchise"] = df["home_team"].apply(config.canonical_team)
    df["away_franchise"] = df["away_team"].apply(config.canonical_team)

    # Trust the CSV's own `season` column directly - it already handles the
    # COVID-shifted spring-2021 games correctly. `config.season_for_date`
    # reproduces it exactly (0 mismatches across all 3,688 rows); not
    # re-derived here to avoid a redundant source of truth.
    df["season"] = df["season"].astype(int)

    df["is_bowl"] = df["season_type"].apply(config.is_bowl_game)
    df["is_neutral_venue"] = df["is_bowl"]  # approximation - see config.py docstring

    df["actual_margin"] = df["home_score"] - df["away_score"]        # home perspective, positive = home won by
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["actual_margin"] > 0).astype(int)

    # ATS result vs the (single, assumed-"close") home line.
    close_line = df["Home Line Close"]
    df["home_covers_close"] = np.select(
        [df["actual_margin"] > -close_line, df["actual_margin"] == -close_line],
        [1, 0],
        default=-1,
    )
    df.loc[close_line.isna(), "home_covers_close"] = np.nan

    total_line = df["Total Score Close"]
    df["over_result_close"] = np.select(
        [df["actual_total"] > total_line, df["actual_total"] == total_line],
        [1, 0],
        default=-1,
    )
    df.loc[total_line.isna(), "over_result_close"] = np.nan

    df = merge_misc_2015_enrichment(df)

    # sort by date, then kickoff time as a tiebreaker for determinism among
    # same-day games (kickoff_utc is a string but ISO-formatted, sorts correctly).
    # Rows added by merge_misc_2015_enrichment() have no real kickoff_utc
    # (NaN sorts last within a date, per pandas' default na_position) - the
    # home/away franchise tiebreak after it just keeps that deterministic.
    df = df.sort_values(
        ["date", "kickoff_utc", "home_franchise", "away_franchise"], kind="stable"
    ).reset_index(drop=True)
    df["game_id"] = df.index.astype(str).map(lambda i: f"cfb_{i}")

    keep_cols = [
        "game_id", "event_id", "date", "season", "week", "is_bowl", "is_neutral_venue",
        "home_team", "away_team", "home_franchise", "away_franchise",
        "home_score", "away_score", "actual_margin", "actual_total", "home_win",
        "home_covers_close", "over_result_close",
    ] + ODDS_COLS
    return df[[c for c in keep_cols if c in df.columns]]


# ---------------------------------------------------------------------------
# 2015-season enrichment from Datasets/Misc./Misc 1/ncaaf_game_scores_1g.csv
#
# Adopted via hypothesis test "cfb_misc1g_2015_enrichment" (see
# decision_log.jsonl, sports/cfb/research_misc_enrichment.py for the full
# research script this was validated with, and docs/METHODOLOGY.md's dated
# CFB subsection for the measured before/after numbers). Motivation: our own
# 2015 season had only 192 recorded games with 1.6% total/moneyline odds
# coverage (66% spread) - this independent source carries 783 real 2015
# games with real spread AND total lines, a materially more complete source
# for that one season. Measured result: margin_corr +0.0135, total_corr
# +0.0337 (both well above the 0.005 noise floor), ROI -2.38% -> -4.54%
# (-2.16pp, within its own +/-2.50pp standard error) - recommendation
# "adopt_cautiously", but adopted outright (unlike several similar-graded
# null results elsewhere in this project) because the fit improvement here
# is large and non-null, not a wash, and it directly fixes the exact
# documented weakness motivating this change: the 2015 season's own
# backtest slice went from 79 bets at a +/-10.73pp standard error (genuinely
# "swallowed by its own noise band," the project's own words for this
# problem) to 871 bets at +/-3.22pp - the actual problem this enrichment was
# built to solve.
#
# MISC_CODE_TO_FRANCHISE was built by cross-referencing the Misc file's 148
# distinct ESPN-style team codes against the ACTUAL franchise strings this
# loader already produces (not guessed spellings) - every value was
# verified to exist as an exact string in this dataset, and the resulting
# game-level matches were independently checked against real 2015 final
# scores before this was trusted (see the research script for the full
# validation, including the diagnosed +1-day UTC-vs-local timezone artifact
# in this source's own `date` column, and the one already-existing,
# independent data bug this incidentally surfaced: South Carolina vs North
# Carolina, 2015-09-03, has home_score/away_score swapped in our own raw
# CSV relative to the real final - left untouched here, since this merge
# only ever fills genuinely missing values, never overwrites an existing
# one, conflicting or not).
#
# Four codes use the SEASON-SPECIFIC spelling our own 2015 rows actually use
# rather than whichever spelling happens to exist elsewhere in the full
# multi-season dataset (the same "same real program, multiple strings"
# inconsistency cfb/config.py's canonical_team() docstring already
# documents as a known, unaddressed limitation): FIU, USM, UTSA, UTM.
MISC_CODE_TO_FRANCHISE = {
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


def merge_misc_2015_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Called from load_games() right after home_covers_close/over_result_close
    are computed, before the final sort/game_id assignment. `df` at this
    point already has home_franchise/away_franchise/season/is_bowl/week and
    every odds/derived column load_games() computes - this function only
    fills genuinely MISSING odds on existing 2015 rows and appends
    genuinely new 2015 rows; it never overwrites a value already present.
    See this module's section docstring above MISC_CODE_TO_FRANCHISE for the
    full research background.
    """
    misc = pd.read_csv(config.MISC_2015_ENRICHMENT_PATH)
    misc["date"] = pd.to_datetime(misc["date"])
    misc["home_franchise"] = misc["home.team"].map(MISC_CODE_TO_FRANCHISE)
    misc["away_franchise"] = misc["away.team"].map(MISC_CODE_TO_FRANCHISE)

    g2015 = df[df["season"] == 2015]

    from collections import defaultdict
    misc_by_pair = defaultdict(list)
    for idx, r in misc.iterrows():
        misc_by_pair[(r["home_franchise"], r["away_franchise"])].append(idx)

    # Match key: exact (home_franchise, away_franchise), date within 0-1
    # days. The "+1 day" tolerance accounts for a diagnosed, 100%-consistent
    # timezone artifact - our own `date` column is the UTC calendar date of
    # kickoff, this source's `date` is the local/US calendar date, so night
    # games that cross the UTC midnight boundary land one day later in our
    # data. See the research script for the full diagnostic.
    matches = []  # (our_idx, misc_idx)
    used_misc_idx = set()
    for idx, r in g2015.iterrows():
        cands = misc_by_pair.get((r["home_franchise"], r["away_franchise"]), [])
        best = None
        for mi in cands:
            if mi in used_misc_idx:
                continue
            delta = (r["date"] - misc.loc[mi, "date"]).days
            if delta in (0, 1) and (best is None or abs(delta) < abs(best[1])):
                best = (mi, delta)
        if best:
            matches.append((idx, best[0]))
            used_misc_idx.add(best[0])

    # ---- Fill missing odds on existing (matched) 2015 games ----
    # game_id doesn't exist yet at this point in load_games() (it's assigned
    # after this function returns) - `matches` holds `df`'s own row-label
    # index values (unchanged by the merge, so .at[] addressing is safe).
    for our_idx, misc_idx in matches:
        m = misc.loc[misc_idx]
        if pd.isna(df.at[our_idx, "Home Line Close"]) and pd.notna(m["line"]):
            df.at[our_idx, "Home Line Close"] = m["line"]
            df.at[our_idx, "Home Line Odds Close"] = ASSUMED_JUICE_DECIMAL
            df.at[our_idx, "Away Line Odds Close"] = ASSUMED_JUICE_DECIMAL
        if pd.isna(df.at[our_idx, "Total Score Close"]) and pd.notna(m["over_under"]):
            df.at[our_idx, "Total Score Close"] = m["over_under"]
            df.at[our_idx, "Total Score Over Close"] = ASSUMED_JUICE_DECIMAL
            df.at[our_idx, "Total Score Under Close"] = ASSUMED_JUICE_DECIMAL

    # Recompute the derived ATS/over-under result columns using the exact
    # same formula as load_games() itself, so newly-filled odds actually
    # feed ats_pct_l10/over-result features downstream.
    close_line = df["Home Line Close"]
    df["home_covers_close"] = np.select(
        [df["actual_margin"] > -close_line, df["actual_margin"] == -close_line], [1, 0], default=-1,
    )
    df.loc[close_line.isna(), "home_covers_close"] = np.nan

    total_line = df["Total Score Close"]
    df["over_result_close"] = np.select(
        [df["actual_total"] > total_line, df["actual_total"] == total_line], [1, 0], default=-1,
    )
    df.loc[total_line.isna(), "over_result_close"] = np.nan

    # ---- Append genuinely new 2015 games ----
    # date -> (week, is_bowl) lookup built from OUR OWN 2015 data (real,
    # already-ESPN-sourced values, not invented). Checked directly: every
    # date in our own 2015 data maps to exactly one (week, is_bowl) pair.
    g2015_now = df[df["season"] == 2015]
    date_lookup = g2015_now.groupby("date").agg(week=("week", "first"), is_bowl=("is_bowl", "first"))
    max_reg_week_2015 = g2015_now.loc[~g2015_now["is_bowl"], "week"].max()

    def _lookup_week_bowl(date):
        if date in date_lookup.index:
            return date_lookup.loc[date, "week"], bool(date_lookup.loc[date, "is_bowl"])
        prev_day = date - pd.Timedelta(days=1)
        if prev_day in date_lookup.index:
            return date_lookup.loc[prev_day, "week"], bool(date_lookup.loc[prev_day, "is_bowl"])
        # Only 2015-12-26 in this source falls outside both lookups (no
        # direct same-day or prior-day game in our own 2015 data) - squarely
        # in the bowl calendar (after the Dec 5 championship weekend), so
        # inferred as a bowl game rather than left unlabeled.
        return max_reg_week_2015 + 1.0, True

    matched_our_idx = {m[0] for m in matches}
    new_rows = []
    for i, misc_idx in enumerate(misc.index[~misc.index.isin(used_misc_idx)]):
        m = misc.loc[misc_idx]
        week, is_bowl = _lookup_week_bowl(m["date"])
        home_score, away_score = float(m["home.score"]), float(m["away.score"])
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        home_line = m["line"] if pd.notna(m["line"]) else np.nan
        total_line_v = m["over_under"] if pd.notna(m["over_under"]) else np.nan

        home_covers = np.nan
        if pd.notna(home_line):
            home_covers = 1 if actual_margin > -home_line else (0 if actual_margin == -home_line else -1)
        over_result = np.nan
        if pd.notna(total_line_v):
            over_result = 1 if actual_total > total_line_v else (0 if actual_total == total_line_v else -1)

        new_rows.append({
            "event_id": np.nan, "date": m["date"], "kickoff_utc": np.nan, "season": 2015,
            "week": week, "is_bowl": is_bowl, "is_neutral_venue": is_bowl,
            "home_team": m["home_franchise"], "away_team": m["away_franchise"],
            "home_franchise": m["home_franchise"], "away_franchise": m["away_franchise"],
            "home_score": home_score, "away_score": away_score,
            "actual_margin": actual_margin, "actual_total": actual_total,
            "home_win": int(actual_margin > 0),
            "home_covers_close": home_covers, "over_result_close": over_result,
            "Home Odds Close": np.nan, "Away Odds Close": np.nan,  # this source has no moneyline
            "Home Line Close": home_line,
            "Home Line Odds Close": ASSUMED_JUICE_DECIMAL if pd.notna(home_line) else np.nan,
            "Away Line Odds Close": ASSUMED_JUICE_DECIMAL if pd.notna(home_line) else np.nan,
            "Total Score Close": total_line_v,
            "Total Score Over Close": ASSUMED_JUICE_DECIMAL if pd.notna(total_line_v) else np.nan,
            "Total Score Under Close": ASSUMED_JUICE_DECIMAL if pd.notna(total_line_v) else np.nan,
        })

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def load_team_game_boxscores() -> pd.DataFrame:
    """
    Player-level passing/rushing box scores (Datasets/College Football/
    cfb-passing-box-scores.csv, cfb-rushing-box-scores.csv), aggregated to
    one row per (event_id, team): total offensive yards (passing + rushing,
    summed across every player on that team in that game) and interceptions
    thrown (a turnover proxy, from the passing file's `int` column).

    Join-key check (done before this function was written, not assumed):
    `event_id` matches cfb-games.csv's own `event_id` column exactly - 0
    team-string mismatches found between a box-score row's `team` field and
    its game's home_team/away_team for every event_id that appears in both.
    Coverage is partial like every other source in this dataset: ~95.4% of
    completed games have a matching event_id in the passing file at all, and
    ~91.9% have BOTH teams' box score present (the rest get a partial or
    missing team-game row here, left as NaN - not filled in this function,
    since the caller (features.py) is responsible for the same walk-forward
    -safe rolling/fill discipline used for every other sparse column, e.g.
    ats_pct_l10's handling of missing spreads).
    """
    passing = pd.read_csv(config.PASSING_BOX_SCORE_PATH)
    rushing = pd.read_csv(config.RUSHING_BOX_SCORE_PATH)

    pass_agg = passing.groupby(["event_id", "team"], as_index=False).agg(
        pass_yds=("yds", "sum"), int_thrown=("int", "sum")
    )
    rush_agg = rushing.groupby(["event_id", "team"], as_index=False).agg(
        rush_yds=("yds", "sum")
    )

    tg = pass_agg.merge(rush_agg, on=["event_id", "team"], how="outer")
    tg["pass_yds"] = tg["pass_yds"].fillna(0.0)
    tg["rush_yds"] = tg["rush_yds"].fillna(0.0)
    tg["int_thrown"] = tg["int_thrown"].fillna(0.0)
    tg["total_yards"] = tg["pass_yds"] + tg["rush_yds"]
    return tg[["event_id", "team", "total_yards", "int_thrown"]]


def load_team_game_comp_pct() -> pd.DataFrame:
    """
    Team-level completions/attempts per (event_id, team), summed across every
    passer on that team in that game, from cfb-passing-box-scores.csv's
    `comp_att` field (e.g. "14/35"). A team-level completion-percentage
    proxy for passing efficiency, distinct from raw yardage (see
    features.py docstring for why this was tested as its own hypothesis
    rather than folded into the yards/turnover feature). Same join-key and
    coverage caveats as load_team_game_boxscores() - not re-verified here,
    same source file.
    """
    passing = pd.read_csv(config.PASSING_BOX_SCORE_PATH)
    split = passing["comp_att"].astype(str).str.split("/", expand=True)
    passing["comp"] = pd.to_numeric(split[0], errors="coerce")
    passing["att"] = pd.to_numeric(split[1], errors="coerce")
    return passing.groupby(["event_id", "team"], as_index=False).agg(
        comp=("comp", "sum"), att=("att", "sum")
    )
