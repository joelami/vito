"""
Adds real HISTORICAL OPENING spread/total lines from a second, independent
archive onto sports/nba/loader.py's game-level output, as new "*Open"
columns -- sitting alongside the "*Close" columns loader.py already
populates from the Pinnacle 2006-2018 archive (see loader.py's module
docstring). Before this module, NBA carried a single Close-only odds
snapshot with no real opening line anywhere, making Closing-Line-Value
(core/backtest.py's clv_pct) uncomputable for NBA -- this closes PART of
that gap (see the PERMANENT GAP section below for what it can't close).

SOURCE: github.com/FinnedAI/sportsbookreview-scraper, data/nba_archive_10Y.json
(a frozen pre-scrape of the now-dead sportsbookreviewsonline.com -- the
original site is confirmed dead, see decision_log.jsonl's 2026-09-03 CLV
feasibility entries). Vendored at Datasets/NBA/archive/nba_archive_10Y.json.
13,903 rows, one row per game, real calendar dates spanning 2011-2021
(the archive's own "season" field uses a "season STARTS in this year"
convention, the opposite of this project's own "season ENDS in this year"
convention -- irrelevant here since every join key below uses the real
calendar date, never the archive's own season label).

TEAM IDENTITY: this archive names teams by CURRENT mascot only (e.g.
"Hornets", "Pelicans"), never a franchise-history-aware name -- verified
directly (not assumed) that this matches loader.py's own convention: a
sample 2011-12-26 game confirmed the archive's "Hornets" (that early, the
real Charlotte team was still named the Bobcats) resolves to the exact same
real game as loader.py's CHA row (96-95 vs Bucks), and the archive's
"Pelicans" (that early, the real New Orleans team was still named the
Hornets) resolves to loader.py's NOP row (85-84 vs Suns) -- i.e. the
archive retroactively labels every game by the CURRENT franchise's mascot,
exactly like loader.py's own current-abbreviation-forever convention (see
nba/config.py's team-identity section) -- so NBA_NAME_TO_ABBREV below is a
flat, era-independent lookup, no relocation-window logic needed.

JOIN VERIFICATION (real numbers, measured directly, not assumed):
  - 34 raw team-name strings observed (30 real teams + 4 one-off scraper
    variants: "LA Clippers", "Golden State", "Oklahoma City", "NewJersey"
    -- all mapped to their real team's normal abbreviation below). Zero
    unmapped strings after the dict below is applied (asserted at load time).
  - 10 of 13,903 raw rows are malformed placeholder rows (home_team/
    home_final/etc. all literal 0 -- Finals-clinching road-win games where
    the scraper failed to capture the home side's row) -- dropped, not
    guessable.
  - Join key is (date, home_franchise, away_franchise) -- exact date, NO
    tolerance needed: unlike NHL (see sports/nhl/odds_loader.py), this
    archive's `date` field carries no time component and lines up with
    loader.py's own date exactly (confirmed directly: adding a +-1 day
    fallback found zero additional matches beyond a plain exact-date merge).
  - Real match rate, measured against loader.py's ACTUAL coverage window
    (loader.py's Pinnacle-backed odds end 2018-06-08, the close of the
    2017-18 season -- games before 2006-07 or after 2017-18 have no real
    odds in loader.py's own source regardless of this join): 8,704 / 8,948
    = 97.27% of in-window games matched. The 2.73% (244 games) that didn't
    were spot-checked, not silently dropped: they cluster almost entirely
    on 2012-10-30/31 and 2013-10-29/30 (opening week of the 2012-13 and
    2013-14 seasons) -- confirmed those specific dates are simply MISSING
    from loader.py's own nba_games_all.csv entirely, a pre-existing gap in
    NBA's primary data source with nothing to do with this archive or this
    join. Every archive row outside loader.py's 2006-2018 coverage window
    is correctly left unmatched (NaN), not counted as a failure.
  - Sign convention verified directly, not assumed: the archive's
    home_open_spread agrees in sign with loader.py's own independently
    -sourced "Home Line Close" (Pinnacle, negative = home favored) in
    99.07% of matched games (n=8,478) -- both use the same
    home-favored-is-negative convention.
  - 3 of 8,704 matched open_over_under values are obvious data-entry
    glitches (19.5, 20.5, 1955.5 -- no real NBA total is outside
    ~140-280) -- filtered to NaN rather than passed through as real
    prices, same discipline as this project's other sanity-bound cleaning
    (e.g. MLB's true-tie drop, NBA loader's own pts>0 filter).

PERMANENT GAP -- read before assuming CLV works for NBA: this archive has
NO opening moneyline field anywhere (only home_close_ml -- verified
directly against every one of the 13,903 raw rows' own keys) and NO
juice/price field for spread or total AT ALL, open OR close (only the LINE
numbers -- no home_open_spread_odds, no open_over_under_odds, unlike MLB/
NHL's version of this same archive). core/edge_finder.py requires a real
two-sided PRICE (not just a line number) to generate ANY betting
opportunity at a given price_point -- so even though this module adds a
real, verified "Home Line Open" and "Total Score Open" (the actual number
the market opened at), NEITHER can produce a backtest bet or a clv_pct
value under the current edge_finder mechanics, since there is no opening
PRICE to place a bet at. This is a genuine, disclosed limitation of the
source, not a bug here or in edge_finder: the two new columns are real,
verified historical facts (useful for future line-movement-only research)
but do NOT make NBA CLV computable today. See decision_log.jsonl for the
full, honest accounting (search "nba_open_odds_archive").
"""

import json

import numpy as np
import pandas as pd

from . import config

ARCHIVE_PATH = config.DATA_DIR / "nba_archive_10Y.json"

# Every raw team-name string observed in the archive -> the SAME abbreviation
# space loader.py's home_franchise/away_franchise already uses (see
# config.ESPN_NAME_TO_ABBREV for the canonical spelling of each franchise's
# current name) -- verified directly against loader.py's real games, not
# assumed to line up (see module docstring's Hornets/Pelicans check).
NBA_NAME_TO_ABBREV = {
    "Bucks": "MIL", "Bulls": "CHI", "Cavaliers": "CLE", "Celtics": "BOS",
    "Clippers": "LAC", "LA Clippers": "LAC", "Golden State": "GSW", "Warriors": "GSW",
    "Grizzlies": "MEM", "Hawks": "ATL", "Heat": "MIA", "Hornets": "CHA",
    "Jazz": "UTA", "Kings": "SAC", "Knicks": "NYK", "Lakers": "LAL",
    "Magic": "ORL", "Mavericks": "DAL", "Nets": "BKN", "NewJersey": "BKN",
    "Nuggets": "DEN", "Oklahoma City": "OKC", "Thunder": "OKC", "Pacers": "IND",
    "Pelicans": "NOP", "Pistons": "DET", "Raptors": "TOR", "Rockets": "HOU",
    "Seventysixers": "PHI", "Spurs": "SAS", "Suns": "PHX", "Timberwolves": "MIN",
    "Trailblazers": "POR", "Wizards": "WAS",
}

# Real NBA totals never fall outside this band -- 3 of 8,704 matched rows
# (19.5, 20.5, 1955.5) are obvious scrape/transcription glitches, not real
# opening totals. See module docstring.
_SANE_TOTAL_RANGE = (140.0, 280.0)


def load_open_odds() -> pd.DataFrame:
    """Returns one row per real, matchable archive game: date, home/away
    franchise (already translated to loader.py's abbreviation space), and
    the two real fields this source has -- home_open_spread, open_over_under.
    Malformed placeholder rows (10 of 13,903, see module docstring) are
    dropped."""
    raw = json.loads(ARCHIVE_PATH.read_text())
    raw = [r for r in raw if isinstance(r.get("home_team"), str) and isinstance(r.get("away_team"), str)]

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    df["home_franchise"] = df["home_team"].map(NBA_NAME_TO_ABBREV)
    df["away_franchise"] = df["away_team"].map(NBA_NAME_TO_ABBREV)
    assert df["home_franchise"].notna().all() and df["away_franchise"].notna().all(), \
        "unmapped team name in nba_archive_10Y.json -- NBA_NAME_TO_ABBREV needs updating"

    df["home_open_spread"] = pd.to_numeric(df["home_open_spread"], errors="coerce")
    df["open_over_under"] = pd.to_numeric(df["open_over_under"], errors="coerce")
    df.loc[~df["open_over_under"].between(*_SANE_TOTAL_RANGE), "open_over_under"] = np.nan

    return df[["date", "home_franchise", "away_franchise", "home_open_spread", "open_over_under"]]


def attach_odds(games: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins real opening spread/total LINES (no price -- see module
    docstring's PERMANENT GAP section) onto `games` (loader.py's output),
    as "Home Line Open" / "Total Score Open" -- matching NFL's naming
    convention for an opening line (see sports/nfl/loader.py's ODDS_COLS).
    Exact-date join, no tolerance needed (verified in module docstring).
    Games outside the archive's 2011-2021 coverage, or with no match, get a
    real, honest NaN -- never a fabricated fallback. Never touches any
    existing "*Close" column.
    """
    # Real deployment gap, caught before shipping: ARCHIVE_PATH is a
    # vendored file fetched directly to disk during research, NOT synced
    # via core.dataset_sync (Datasets/ is gitignored and this file was
    # never uploaded to the R2 bucket that backs it) -- so on a fresh
    # production checkout this path genuinely will not exist yet.
    # pipeline.py's odds_loader-import try/except only catches
    # ModuleNotFoundError, not a missing file, so a bare .read_text() here
    # would crash the ENTIRE NBA pipeline build, not just skip this one
    # feature -- exactly the kind of "no data available" case this
    # project's own convention is to fall back on honestly (real NaN, see
    # this function's own docstring), not crash on.
    if not ARCHIVE_PATH.exists():
        print(f"[nba/odds_loader] {ARCHIVE_PATH} not found (not yet synced to R2) -- "
              f"skipping open-line attach, Home Line Open/Total Score Open will be NaN for every game.")
        out = games.copy()
        out["Home Line Open"] = np.nan
        out["Total Score Open"] = np.nan
        return out

    open_odds = load_open_odds()

    merged = games.merge(
        open_odds, on=["date", "home_franchise", "away_franchise"], how="left",
    )

    # a real (date, home, away) key should be unique -- defensively guard
    # against any future archive update introducing a duplicate, same
    # pattern sports/mlb/odds_loader.py already uses for its own join.
    dupe_counts = merged.groupby("game_id")["home_franchise"].transform("size")
    ambiguous = dupe_counts > 1
    if ambiguous.any():
        merged.loc[ambiguous, ["home_open_spread", "open_over_under"]] = np.nan
    merged = merged.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)
    assert len(merged) == len(games), "attach_odds must return exactly one row per input game"

    merged["Home Line Open"] = merged["home_open_spread"]
    merged["Total Score Open"] = merged["open_over_under"]

    keep_extra = ["Home Line Open", "Total Score Open"]
    return pd.concat([games.reset_index(drop=True), merged[keep_extra]], axis=1)
