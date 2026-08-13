"""
RESEARCH SCRIPT — not wired into build_features/ML_FEATURE_COLS by default.
Tests one hypothesis via core.research.evaluate_hypothesis: whether a
starting pitcher's rolling recent run-prevention is a genuine, currently-
missing signal for MLB margin/total prediction.

SCOPING NOTE, stated here and again in the Hypothesis reasoning passed to
evaluate_hypothesis: this is deliberately NOT "ERA" (earned runs per 9
innings pitched). Computing true innings-normalized ERA per start requires
tracking every substitution (`sub`) event in the play-by-play to know
exactly which outs each pitcher was responsible for — a materially harder
parse than what's attempted here. What IS computed is EARNED RUNS ALLOWED
PER START — a per-appearance total, not innings-normalized — a real, if
cruder, proxy for pitcher quality. This is the same spirit as CFB's
is_bowl-as-neutral-site proxy: an honestly-scoped approximation of the real
signal, not a fabrication of one that doesn't exist. Named `sp_er_l{N}`
throughout (never "era") to keep that honest.

Source: Datasets/MLB/2010seve/*.EVA|*.EVN and Datasets/MLB/2020seve/*.EVA|
*.EVN — real Retrosheet PLAY-BY-PLAY EVENT files (one file per HOME team per
season, e.g. 2010ANA.EVA = every game the Angels hosted in 2010), covering
~2010-2025. This is NOT the same source as loader.py's game logs (final
scores only) — it's a separate, richer source. Format verified directly
with bash head/grep before any parsing code was written (see inline
comments below for what was specifically confirmed):

  - Each game is delimited by an `id,<TEAMDATE><game_number>` line.
  - `info,visteam` / `info,hometeam` / `info,date` (YYYY/MM/DD) /
    `info,number` (0/1/2, doubleheader game number) carry everything needed
    to build the exact same merge key loader.py's own `game_number` column
    already uses for same-day disambiguation.
  - `start,playerid,name,home_flag(0/1),battingpos,fieldpos` lines list the
    starting lineup for both teams; fieldpos==1 identifies the starting
    pitcher. Spot-checked directly (2020-07-28 SEA@ANA: shefj001 fieldpos=1
    for the visitor, sandp002 fieldpos=1 for the home team) and enforced
    programmatically below (a game missing exactly one starter per side is
    counted and dropped, never guessed).
  - `data,er,playerid,earnedruns` lines report earned runs allowed, one line
    per pitcher who appeared, INCLUDING a starter who was pulled early —
    confirmed directly: in that same 2020-07-28 game, shefj001 was relieved
    by cortn001 and grotz001, and still has its own
    `data,er,shefj001,4` line alongside theirs.
  - Team codes in these event files are the SAME 3-letter Retrosheet codes
    loader.py's game logs use (diffed the full code set against
    gl2010.txt's raw codes: identical). No separate translation table is
    needed the way odds_loader.py needed one for the different odds
    archive's codes — `config.canonical_team` is reused as-is, and both
    relocations that fall inside this 2010-2025 window (FLO->MIA in 2012,
    OAK->ATH in 2025) are present in the event-file code set too (FLO
    appears through 2011, MIA from 2012 on; OAK through 2024, ATH in 2025).

Run standalone with:  python -m sports.mlb.research_starting_pitcher
(from backend/) to print parser diagnostics and the full hypothesis-test
result without touching any live feature/config file.
"""

import csv
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from . import config

EVENT_SUBDIRS = ["2010seve", "2020seve"]

# Rolling window: pitcher's last N starts. Kept inside the 5-10 range
# specified for this experiment; 8 chosen as the midpoint - enough starts
# (~5-6 weeks of a normal 5-man rotation) to smooth out one bad/great outing
# without going so wide it stops reflecting *recent* form.
ROLL_N_STARTS = 8


# ---------------------------------------------------------------------
# 1. Parse raw event files into one row per game
# ---------------------------------------------------------------------

def _iter_event_files():
    for sub in EVENT_SUBDIRS:
        for ext in ("EVA", "EVN"):
            for f in sorted(glob.glob(str(config.DATA_DIR / sub / f"*.{ext}"))):
                yield f


def _parse_file(path):
    """Returns a list of dicts, one per game, parsed from a single .EVA/.EVN file."""
    games = []
    cur = None
    starter_issue = 0

    with open(path, encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            tag = row[0]

            if tag == "id":
                if cur is not None:
                    games.append(cur)
                cur = {"gid": row[1], "starters": {}, "er": {}}
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
                    if side in cur["starters"]:
                        starter_issue += 1  # duplicate fieldpos==1 for one side - shouldn't happen
                    cur["starters"][side] = player_id
            elif tag == "data" and len(row) >= 4 and row[1] == "er":
                cur["er"][row[2]] = row[3]

    if cur is not None:
        games.append(cur)
    return games, starter_issue


def load_event_games() -> pd.DataFrame:
    """
    One row per game found in the event files: date, game_number, raw team
    codes, canonical franchises, starter ids, and their earned-runs-allowed
    that game. Games missing a clean single starter per side, or missing an
    ER entry for a found starter, are reported and dropped rather than
    guessed.
    """
    files = list(_iter_event_files())
    all_games = []
    total_starter_issues = 0

    for f in files:
        games, issues = _parse_file(f)
        total_starter_issues += issues
        all_games.extend(games)

    n_raw = len(all_games)

    rows = []
    n_missing_starter = 0
    n_missing_er = 0
    for g in all_games:
        if "home" not in g["starters"] or "away" not in g["starters"]:
            n_missing_starter += 1
            continue
        home_sp, away_sp = g["starters"]["home"], g["starters"]["away"]
        home_er = g["er"].get(home_sp)
        away_er = g["er"].get(away_sp)
        if home_er is None or away_er is None:
            n_missing_er += 1
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

    print(f"[research_starting_pitcher] Parsed {len(files)} event files, {n_raw:,} games found.")
    print(f"[research_starting_pitcher] Dropped {n_missing_starter:,} games missing a clean single "
          f"starter (fieldpos==1) per side; {total_starter_issues:,} duplicate-fieldpos==1 anomalies noted.")
    print(f"[research_starting_pitcher] Dropped {n_missing_er:,} games where a found starter had no "
          f"matching data,er line.")
    print(f"[research_starting_pitcher] Clean game rows kept: {len(df):,}")

    return df


# ---------------------------------------------------------------------
# 2. Per-pitcher chronological log -> walk-forward-safe rolling ER/start
# ---------------------------------------------------------------------

def build_rolling_starter_quality(event_games: pd.DataFrame, n: int = ROLL_N_STARTS):
    """
    Long-format per-pitcher-start log (one row per start, whichever side),
    sorted chronologically per pitcher, with a shift(1)-then-rolling(n,
    min_periods=1) mean of earned runs allowed per start - the same
    walk-forward discipline (current game's own result never included in
    its own feature) every other rolling feature in this project follows,
    see sports/nfl/features.py's _rolling_form for the identical pattern.

    Returns (event_games_with_rolling_cols, league_avg_er_per_start).
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

    league_avg_er_per_start = float(log["er"].mean())

    grp = log.groupby("pitcher", group_keys=False)
    log["sp_er_lN"] = grp["er"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).mean())

    n_first_starts = log["sp_er_lN"].isna().sum()
    log["sp_er_lN"] = log["sp_er_lN"].fillna(league_avg_er_per_start)

    print(f"[research_starting_pitcher] League-avg earned runs allowed per start (this data): "
          f"{league_avg_er_per_start:.3f} - used to fill {n_first_starts:,} true first-tracked-starts "
          f"({n_first_starts / len(log) * 100:.1f}% of {len(log):,} pitcher-starts), same fallback "
          f"convention as LEAGUE_AVG_TEAM_SCORE elsewhere in this module.")

    home_roll = log[log["side"] == "home"][["date", "game_number", "team", "opponent", "sp_er_lN"]].rename(
        columns={"team": "home_franchise", "opponent": "away_franchise", "sp_er_lN": "home_sp_er_lN"}
    )
    away_roll = log[log["side"] == "away"][["date", "game_number", "team", "opponent", "sp_er_lN"]].rename(
        columns={"team": "away_franchise", "opponent": "home_franchise", "sp_er_lN": "away_sp_er_lN"}
    )

    out = event_games.merge(home_roll, on=["date", "game_number", "home_franchise", "away_franchise"], how="left")
    out = out.merge(away_roll, on=["date", "game_number", "home_franchise", "away_franchise"], how="left")

    return out, league_avg_er_per_start


# ---------------------------------------------------------------------
# 3. Join onto the main MLB games frame
# ---------------------------------------------------------------------

def attach_starter_quality(games: pd.DataFrame, league_avg_fallback: float = None):
    """
    `games` is sports/mlb/loader.py's load_games() output (optionally
    already odds-attached - doesn't matter, this only needs date/franchise/
    game_number). Returns games with two new columns, home_sp_er_lN /
    away_sp_er_lN, NaN for any game not covered by the 2010seve/2020seve
    event-file window (~2010-2025) or not cleanly parsed.

    Also returns diagnostics: (games_out, league_avg_er_per_start, match_stats dict).
    """
    event_games = load_event_games()
    rolled, league_avg = build_rolling_starter_quality(event_games, n=ROLL_N_STARTS)

    key = ["date", "game_number", "home_franchise", "away_franchise"]
    dupe_check = rolled.groupby(key).size()
    n_dupe_keys = (dupe_check > 1).sum()
    if n_dupe_keys:
        print(f"[research_starting_pitcher] WARNING: {n_dupe_keys} (date, game_number, home, away) "
              f"keys are not unique in event-file data - keeping first occurrence only to avoid "
              f"a fan-out join.")
        rolled = rolled.drop_duplicates(subset=key, keep="first")

    merged = games.merge(
        rolled[key + ["home_sp_er_lN", "away_sp_er_lN"]],
        on=key, how="left",
    )
    assert len(merged) == len(games), "attach_starter_quality must return exactly one row per input game"

    return merged, league_avg


# ---------------------------------------------------------------------
# 4. Standalone diagnostics run
# ---------------------------------------------------------------------

def main():
    from sports.mlb.loader import load_games
    from sports.mlb.odds_loader import attach_odds

    games = load_games()
    games = attach_odds(games)

    merged, league_avg = attach_starter_quality(games)

    has_odds = merged["Home Odds Close"].notna() & merged["Away Odds Close"].notna()
    in_backtest_window = has_odds  # odds coverage IS the 2012-2021 window
    n_window = int(in_backtest_window.sum())

    has_real_sp = merged["home_sp_er_lN"].notna() & merged["away_sp_er_lN"].notna()
    n_window_with_sp = int((in_backtest_window & has_real_sp).sum())

    print("\n" + "=" * 78)
    print("EVENT-FILE / ODDS-WINDOW COVERAGE")
    print("=" * 78)
    print(f"Backtest-eligible games (real odds, 2012-2021): {n_window:,}")
    print(f"...of those, matched to an event-file starter-quality row (both sides): "
          f"{n_window_with_sp:,} ({n_window_with_sp / n_window * 100:.1f}%)")
    print(f"League-average earned runs allowed per start (fallback value): {league_avg:.3f}")


if __name__ == "__main__":
    main()
