"""
Adds real HISTORICAL OPENING moneyline/total data from a second, independent
archive onto sports/nhl/loader.py's game-level output, as new "*Open"
columns -- sitting alongside the "*Close" columns loader.py already
populates (see loader.py's module docstring: NHL's existing Close moneyline
is a LOW-CONFIDENCE heuristic reconstruction, spread-sign-based, only
55.5-81.2% concordant -- untouched here, this module only ADDS Open
columns, never overwrites any existing Close value). Before this module,
NHL carried a single Close-only odds snapshot with no real opening line
anywhere, making Closing-Line-Value (core/backtest.py's clv_pct)
uncomputable for NHL -- this closes PART of that gap (see the PERMANENT
GAP section below for what it can't close).

SOURCE: github.com/FinnedAI/sportsbookreview-scraper, data/nhl_archive_10Y.json
(a frozen pre-scrape of the now-dead sportsbookreviewsonline.com -- the
original site is confirmed dead, see decision_log.jsonl's 2026-09-03 CLV
feasibility entries). Vendored at Datasets/NHL/archive/nhl_archive_10Y.json.
13,678 rows, one row per game, real calendar dates spanning 2011-2021.
UNLIKE MLB's version of this same archive (see the "MLB was rejected"
note in decision_log.jsonl), this side's team-pairing is real and clean --
verified directly across many seasons before trusting it (see JOIN
VERIFICATION below), not assumed just because the MLB side of the exact
same GitHub repo turned out to be corrupted.

TEAM IDENTITY: this archive names teams by CURRENT mascot only, verified
directly to already be era-independent (e.g. "Coyotes" appears for the
2014+ Arizona-branded seasons, "Phoenix" for 2011-2013 -- both map to the
same already-canonicalized team_id 129764 loader.py itself uses post its
own 2024 Utah-relocation mapping, config.FRANCHISE_CANONICAL). NHL_NAME_TO_
TEAM_ID below maps every raw string straight to that already-canonicalized
id space, so no separate relocation-window logic is needed here either.

DATE HANDLING -- the one real complication this source has that MLB/NBA's
archives don't: loader.py's own `date` column is a UTC TIMESTAMP (not a
plain date), because the raw nhl_data_plus.csv records real game start
times. An evening local puck-drop commonly lands after midnight UTC,
rolling the STORED date to the next calendar day -- confirmed directly
(e.g. a real 2011-10-08 Phoenix @ San Jose game, final score 6-3, appears
in loader.py's own output dated 2011-10-09 UTC). The archive here has no
time component at all, so a plain date-equality join would silently miss
most games. Fix: try date+1 first (the dominant real relationship, 8,982
of 12,992 final matches), then date+0 and date-1, then -- for the narrow
2020 COVID-bubble window only -- date+-365/366 days (the archive itself
mislabels the real Aug-2020 empty-arena playoff games as "2019", one year
early -- confirmed directly: e.g. archive's "2019-08-01" Hurricanes/Rangers
game has no real match anywhere in real August 2019 (NHL doesn't play
then) but matches loader.py's real 2020-08-01 Hurricanes/Rangers game
exactly by score and teams).

JOIN VERIFICATION (real numbers, measured directly, not assumed):
  - 39 raw team-name strings observed, all mapped (zero unmapped, asserted
    at load time) to NHL's 32 real team_ids (30 original-era franchises +
    Vegas + Seattle, matching loader.py's own REAL_TEAM_IDS count).
  - 10 of 13,678 raw rows are malformed placeholder rows (home_team/
    home_final etc. all literal 0, same failure pattern as NBA's 10) --
    dropped, not guessable.
  - Real american-odds moneylines never realistically exceed +-1000; 6 of
    12,688 pre-sanity-filter matched rows did (up to -2156), every one a
    clear transcription glitch (verified: e.g. the Golden Knights/Blue
    Jackets 2020-01-11 game's own home_close_ml is a normal -210, while its
    home_open_ml reads -2156 -- a 10x jump no real line move produces) --
    filtered to NaN rather than passed through as real prices.
  - After the date-tolerance fix above: 12,992 / 13,668 = 95.05% of
    (non-malformed) archive rows matched to exactly one real loader.py
    game_id. 39 archive-side (date, team-pair) keys matched to a game_id
    that ANOTHER archive row also matched -- spot-checked, not silently
    resolved: every one of the 39 is a genuine DUPLICATE scrape of the same
    real game recorded under two different `season` labels (e.g. a 2020
    -01-14 Coyotes/Sharks game appears once tagged season=2019 with
    home_open_ml=-140 and again tagged season=2020 with home_open_ml=-115
    -- the archive itself disagrees with itself on the true value) --
    correctly left unmatched (NaN) rather than guessing which duplicate is
    right, same "never guess on a real ambiguity" discipline
    sports/mlb/odds_loader.py already applies to its doubleheader case.
  - The remaining ~4.4% of rows genuinely never matched even with every
    tolerance above -- not further chased down here; loader.py's own
    coverage (2004-2025) is far wider than the archive's 2011-2021 window,
    so some of this is plain out-of-range archive rows for games loader.py
    simply never had betting-relevant issues with; documented honestly as
    residual, not claimed as 100%.

PERMANENT GAP -- read before assuming full CLV works for NHL: this archive
has NO opening (or closing) spread/puckline field anywhere (only
home_close_spread -- verified directly against every one of the 13,678 raw
rows' own keys), consistent with this project's own prior finding that NHL
puckline odds are degenerate/near-fixed in every source checked so far (see
decision_log.jsonl's NHL edge-magnitude confidence entries). Only
moneyline and total are recoverable from this source. Total's own juice
(open_over_under_odds) is ALSO not wired in here, deliberately: the
upstream scraper's own code (github.com/FinnedAI/sportsbookreview-scraper/
scrapers/sportsbookreview.py) shows this single juice value is read from
only ONE of the two raw per-game rows with no reliable way to tell whether
it's the Over or the Under side's price -- attaching it to either side
without real justification would be exactly the kind of fabricated-looking
value this task was told never to produce, so "Total Score Open" here is
the real LINE only, no price -- like NBA's spread/total Open, this can't
by itself produce a total-market backtest bet or clv_pct (core/edge_finder.py
requires a real two-sided price to generate any opportunity), but IS a
real, verified historical fact useful for future line-movement research.
See decision_log.jsonl for the full, honest accounting (search
"nhl_open_odds_archive").
"""

import json

import numpy as np
import pandas as pd

from . import config
from core import odds_math

ARCHIVE_PATH = config.DATA_PATH.parent / "nhl_archive_10Y.json"

# Every raw team-name string observed in the archive -> loader.py's own
# already-canonicalized numeric team_id space (config.FRANCHISE_CANONICAL
# already applied -- e.g. every Phoenix/Arizona/Utah-era name below maps
# straight to 129764). Verified directly against loader.py's real games
# (see module docstring), not assumed.
NHL_NAME_TO_TEAM_ID = {
    "Avalanche": 17, "Blackhawks": 4, "Blue Jackets": 29, "Bruins": 1,
    "Canadiens": 10, "Canucks": 22, "Capitals": 23,
    "Coyotes": 129764, "Phoenix": 129764, "Arizonas": 129764,
    "Devils": 11, "Ducks": 25, "Flames": 3, "Flyers": 15, "Golden Knights": 37,
    "Hurricanes": 7, "Islanders": 12, "NY Islanders": 12, "Jets": 28, "WinnipegJets": 28,
    "Kings": 8, "Kraken": 124292, "SeattleKraken": 124292,
    "Lightning": 20, "Tampa": 20, "Tampa Bay": 20,
    "Maple Leafs": 21, "Oilers": 6, "Panthers": 26, "Penguins": 16, "Predators": 27,
    "Rangers": 13, "Red Wings": 5, "Sabres": 2, "Senators": 14, "Sharks": 18,
    "St.Louis": 19, "Stars": 9, "Wild": 30,
}

# Real american-odds moneylines never realistically reach +-1000 in this
# league; a handful of matched rows do (up to -2156) and are verified
# transcription glitches -- see module docstring.
_SANE_ML_ABS_MAX = 1000.0

# Real NHL totals never fall outside this band (the lowest genuine value
# observed anywhere in loader.py's own real close_over_under data is 4.5) --
# 3 of 12,991 matched rows read 2.0/2.5/3.0, all verified transcription
# glitches (e.g. the 2021-06-14 Golden Knights/Canadiens game's own
# close_over_under is a normal 5.5, while its open_over_under reads 2.0 --
# no real total moves that far). Filtered to NaN, never passed through.
_SANE_TOTAL_RANGE = (4.0, 10.0)


def _load_clean_archive() -> pd.DataFrame:
    """Raw archive -> cleaned rows with franchise ids resolved and obvious
    data-entry glitches (0-as-missing-sentinel, |ml|>1000) nulled out.
    Does NOT join to loader.py yet -- see `attach_odds`."""
    raw = json.loads(ARCHIVE_PATH.read_text())
    raw = [r for r in raw if isinstance(r.get("home_team"), str) and isinstance(r.get("away_team"), str)]

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    df["home_franchise"] = df["home_team"].map(NHL_NAME_TO_TEAM_ID)
    df["away_franchise"] = df["away_team"].map(NHL_NAME_TO_TEAM_ID)
    assert df["home_franchise"].notna().all() and df["away_franchise"].notna().all(), \
        "unmapped team name in nhl_archive_10Y.json -- NHL_NAME_TO_TEAM_ID needs updating"

    for col in ("home_open_ml", "away_open_ml", "open_over_under"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # the upstream scraper's own blacklist step writes a literal 0 for a
        # missing/unparseable price (see sportsbookreview.py) -- never a real
        # american-odds value, so treat it as the missing sentinel it is.
        df.loc[df[col] == 0, col] = np.nan

    for col in ("home_open_ml", "away_open_ml"):
        df.loc[df[col].abs() > _SANE_ML_ABS_MAX, col] = np.nan

    df.loc[~df["open_over_under"].between(*_SANE_TOTAL_RANGE), "open_over_under"] = np.nan

    return df


def _candidate_dates(d: pd.Timestamp):
    """See module docstring's DATE HANDLING section. +1 day first (the
    dominant real relationship), then 0/-1, then -- Jul-Sep only -- the
    COVID-bubble +-365/366 day fix."""
    cands = [d + pd.Timedelta(days=1), d, d - pd.Timedelta(days=1)]
    if 7 <= d.month <= 9:
        cands += [d + pd.Timedelta(days=off) for off in (365, -365, 366, -366)]
    return cands


def attach_odds(games: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins real opening moneyline (both sides) and the real opening
    total LINE (no price -- see module docstring's PERMANENT GAP section)
    onto `games` (loader.py's output), as "Home Odds Open" / "Away Odds
    Open" / "Total Score Open" -- matching NFL's naming convention (see
    sports/nfl/loader.py's ODDS_COLS). Games outside the archive's
    2011-2021 coverage, or with no confident match, get a real, honest NaN
    -- never a fabricated fallback. Never touches any existing "*Close"
    column.
    """
    # Real deployment gap, caught before shipping: ARCHIVE_PATH is a
    # vendored file fetched directly to disk during research, NOT synced
    # via core.dataset_sync (Datasets/ is gitignored and this file was
    # never uploaded to the R2 bucket that backs it) -- so on a fresh
    # production checkout this path genuinely will not exist yet.
    # pipeline.py's odds_loader-import try/except only catches
    # ModuleNotFoundError, not a missing file, so a bare .read_text() here
    # would crash the ENTIRE NHL pipeline build, not just skip this one
    # feature -- exactly the kind of "no data available" case this
    # project's own convention is to fall back on honestly (real NaN, see
    # this function's own docstring), not crash on.
    if not ARCHIVE_PATH.exists():
        print(f"[nhl/odds_loader] {ARCHIVE_PATH} not found (not yet synced to R2) -- "
              f"skipping open-line attach, Home/Away Odds Open and Total Score Open will be NaN for every game.")
        out = games.copy()
        out["Home Odds Open"] = np.nan
        out["Away Odds Open"] = np.nan
        out["Total Score Open"] = np.nan
        return out

    archive = _load_clean_archive()

    lookup = {}
    for gid, d, h, a in zip(games["game_id"], games["date"], games["home_franchise"], games["away_franchise"]):
        lookup.setdefault((pd.Timestamp(d).tz_localize(None).normalize(), h, a), []).append(gid)

    matched_game_id = []
    for h, a, d in zip(archive["home_franchise"], archive["away_franchise"], archive["date"]):
        found = None
        for cd in _candidate_dates(d):
            cands = lookup.get((cd, h, a))
            if cands:
                found = cands[0]
                break
        matched_game_id.append(found)
    archive = archive.copy()
    archive["game_id"] = matched_game_id

    # a game_id claimed by more than one archive row is a real, unresolved
    # ambiguity (verified: every case found is a genuine duplicate scrape
    # under conflicting values, see module docstring) -- drop both/all
    # rather than guess.
    dupe_counts = archive.loc[archive["game_id"].notna(), "game_id"].value_counts()
    ambiguous_gids = set(dupe_counts[dupe_counts > 1].index)
    archive.loc[archive["game_id"].isin(ambiguous_gids), "game_id"] = None

    matched = archive.dropna(subset=["game_id"]).drop_duplicates(subset=["game_id"], keep=False)
    odds_by_game = matched.set_index("game_id")[["home_open_ml", "away_open_ml", "open_over_under"]]

    out = games.merge(odds_by_game, left_on="game_id", right_index=True, how="left")
    assert len(out) == len(games), "attach_odds must return exactly one row per input game"

    out["Home Odds Open"] = out["home_open_ml"].apply(
        lambda v: odds_math.american_to_decimal(v) if pd.notna(v) else np.nan)
    out["Away Odds Open"] = out["away_open_ml"].apply(
        lambda v: odds_math.american_to_decimal(v) if pd.notna(v) else np.nan)
    out["Total Score Open"] = out["open_over_under"]

    return out.drop(columns=["home_open_ml", "away_open_ml", "open_over_under"])
