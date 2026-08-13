"""
Starting-pitcher rolling run-prevention feature, joined onto the MLB games
frame. ADOPTED (adopt_cautiously) after a hypothesis test via core.research
- see research_starting_pitcher.py for the full investigation writeup
(exact event-file format verification, parser diagnostics, and the
evaluate_hypothesis() result this module's adoption is based on) and
decision_log.jsonl for the logged outcome.

SCOPING NOTE, carried over from the research script and worth repeating
here since it governs what this feature actually means: this is NOT "ERA"
(earned runs per 9 innings pitched). Computing true innings-normalized ERA
per start requires tracking every substitution event in the play-by-play to
know exactly which outs each pitcher was responsible for - a materially
harder parse, out of scope for this pass. What's computed is EARNED RUNS
ALLOWED PER START (a per-appearance total, not innings-normalized) - a
real, if cruder, proxy for pitcher quality, same spirit as CFB's
is_bowl-as-neutral-site proxy. Column names use `sp_er_lN` throughout
(never "era") to keep that honest.

Source: Datasets/MLB/2010seve/*.EVA|*.EVN and Datasets/MLB/2020seve/*.EVA|
*.EVN - real Retrosheet play-by-play EVENT files (one file per HOME team
per season), covering ~2010-2025. Team codes in these files are the SAME
3-letter Retrosheet codes loader.py's game logs use - config.canonical_team
is reused as-is, no separate translation table needed (verified in
research_starting_pitcher.py).

COVERAGE: event files span ~2010-2025. Games from 1990-2009 (and any
pitcher's true first tracked start within the covered window) have no real
starter-quality signal and get the league-average fallback below -
documented, not silently blended in as if it were real. The
2012-2021 odds-covered backtest window is essentially fully covered by
these event files (verified: 100% of backtest-eligible games matched to a
real event-file start record for both sides in the research run).
"""

import csv
import glob

import pandas as pd

from . import config

EVENT_SUBDIRS = ["2010seve", "2020seve"]

# Rolling window: pitcher's last N starts. 8 is the documented choice for
# this feature - inside the 5-10 range this experiment scoped for, and
# roughly 5-6 weeks of a normal 5-man rotation, enough to smooth one bad/
# great outing without going so wide it stops reflecting recent form.
ROLL_N_STARTS = 8

# League-average earned runs allowed per start, computed directly from every
# clean pitcher-start extracted from the 2010seve/2020seve event files
# (74,682 starts, mean 2.578, rounded here same convention as
# features.py's LEAGUE_AVG_TEAM_SCORE). Used as the fallback for (a) a
# pitcher's true first tracked start in this window, and (b) any game
# outside the ~2010-2025 event-file coverage entirely (e.g. 1990-2009).
LEAGUE_AVG_SP_ER_PER_START = 2.58


def _iter_event_files():
    for sub in EVENT_SUBDIRS:
        for ext in ("EVA", "EVN"):
            for f in sorted(glob.glob(str(config.DATA_DIR / sub / f"*.{ext}"))):
                yield f


def _parse_file(path):
    """Returns a list of dicts, one per game, parsed from a single .EVA/.EVN file."""
    games = []
    cur = None

    with open(path, encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            tag = row[0]

            if tag == "id":
                if cur is not None:
                    games.append(cur)
                cur = {"starters": {}, "er": {}}
            elif cur is None:
                continue
            elif tag == "info":
                key = row[1]
                val = row[2] if len(row) > 2 else ""
                if key in ("visteam", "hometeam", "date", "number"):
                    cur[key] = val
            elif tag == "start":
                # start,playerid,name,home_flag(0/1),battingpos,fieldpos
                player_id, home_flag, fieldpos = row[1], row[3], row[5]
                if fieldpos == "1":
                    side = "home" if home_flag == "1" else "away"
                    cur["starters"][side] = player_id
            elif tag == "data" and len(row) >= 4 and row[1] == "er":
                cur["er"][row[2]] = row[3]

    if cur is not None:
        games.append(cur)
    return games


def load_event_games() -> pd.DataFrame:
    """
    One row per game found in the event files: date, game_number, canonical
    franchises, starter ids, and their earned-runs-allowed that game. Games
    missing a clean single starter per side, or missing an ER entry for a
    found starter, are dropped (see research_starting_pitcher.py for exact
    counts - 0 and 2 respectively out of 37,343 games at the time this
    module was adopted).
    """
    rows = []
    for f in _iter_event_files():
        for g in _parse_file(f):
            if "home" not in g["starters"] or "away" not in g["starters"]:
                continue
            home_sp, away_sp = g["starters"]["home"], g["starters"]["away"]
            home_er, away_er = g["er"].get(home_sp), g["er"].get(away_sp)
            if home_er is None or away_er is None:
                continue
            rows.append({
                "date": g.get("date"),
                "game_number": g.get("number", "0"),
                "home_team_raw": g.get("hometeam"),
                "away_team_raw": g.get("visteam"),
                "home_sp_id": home_sp,
                "away_sp_id": away_sp,
                "home_sp_er": float(home_er),
                "away_sp_er": float(away_er),
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["home_franchise"] = df["home_team_raw"].apply(config.canonical_team)
    df["away_franchise"] = df["away_team_raw"].apply(config.canonical_team)
    return df


def _build_rolling_starter_quality(event_games: pd.DataFrame, n: int = ROLL_N_STARTS) -> pd.DataFrame:
    """
    Long-format per-pitcher-start log, sorted chronologically per pitcher,
    with a shift(1)-then-rolling(n, min_periods=1) mean of earned runs
    allowed per start - the same walk-forward discipline (current game's
    own result never included in its own feature) every other rolling
    feature in this project follows (see sports/nfl/features.py's
    _rolling_form for the identical pattern).
    """
    home = event_games[["date", "game_number", "home_franchise", "away_franchise", "home_sp_id", "home_sp_er"]].rename(
        columns={"home_franchise": "team", "away_franchise": "opponent",
                 "home_sp_id": "pitcher", "home_sp_er": "er"}
    )
    home["side"] = "home"
    away = event_games[["date", "game_number", "home_franchise", "away_franchise", "away_sp_id", "away_sp_er"]].rename(
        columns={"away_franchise": "team", "home_franchise": "opponent",
                 "away_sp_id": "pitcher", "away_sp_er": "er"}
    )
    away["side"] = "away"

    log = pd.concat([home, away], ignore_index=True)
    log = log.sort_values(["pitcher", "date", "game_number"], kind="stable").reset_index(drop=True)

    grp = log.groupby("pitcher", group_keys=False)
    log["sp_er_lN"] = grp["er"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    log["sp_er_lN"] = log["sp_er_lN"].fillna(LEAGUE_AVG_SP_ER_PER_START)

    home_roll = log[log["side"] == "home"][["date", "game_number", "team", "opponent", "sp_er_lN"]].rename(
        columns={"team": "home_franchise", "opponent": "away_franchise", "sp_er_lN": "home_sp_er_lN"}
    )
    away_roll = log[log["side"] == "away"][["date", "game_number", "team", "opponent", "sp_er_lN"]].rename(
        columns={"team": "away_franchise", "opponent": "home_franchise", "sp_er_lN": "away_sp_er_lN"}
    )

    out = event_games.merge(home_roll, on=["date", "game_number", "home_franchise", "away_franchise"], how="left")
    out = out.merge(away_roll, on=["date", "game_number", "home_franchise", "away_franchise"], how="left")
    return out


def attach_starter_quality(games: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins two new columns onto `games` (sports/mlb/loader.py's output,
    or anything carrying date/game_number/home_franchise/away_franchise):
    `home_sp_er_lN` / `away_sp_er_lN`, each starter's rolling earned-runs-
    allowed-per-start over their last ROLL_N_STARTS starts, walk-forward
    -safe. Any game outside the ~2010-2025 event-file coverage, or not
    cleanly matched, gets the LEAGUE_AVG_SP_ER_PER_START fallback -
    documented, not left as a silent NaN a downstream ML model would choke
    on or a merge would quietly propagate.
    """
    event_games = load_event_games()
    rolled = _build_rolling_starter_quality(event_games)

    key = ["date", "game_number", "home_franchise", "away_franchise"]
    dupe_counts = rolled.groupby(key).size()
    if (dupe_counts > 1).any():
        rolled = rolled.drop_duplicates(subset=key, keep="first")

    merged = games.merge(rolled[key + ["home_sp_er_lN", "away_sp_er_lN"]], on=key, how="left")
    assert len(merged) == len(games), "attach_starter_quality must return exactly one row per input game"

    merged["home_sp_er_lN"] = merged["home_sp_er_lN"].fillna(LEAGUE_AVG_SP_ER_PER_START)
    merged["away_sp_er_lN"] = merged["away_sp_er_lN"].fillna(LEAGUE_AVG_SP_ER_PER_START)
    return merged
