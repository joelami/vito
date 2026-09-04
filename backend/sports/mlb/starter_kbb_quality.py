"""
Starting-pitcher rolling K-BB% feature, joined onto the MLB games frame.
ADOPTED (plain "adopt" -- a real, noise-floor-clearing statistical fit
improvement, not just a harmless null) after a hypothesis test via
core.research -- see research_starter_kbb_pct.py for the full investigation
writeup (play-by-play parsing verification, parser diagnostics, and the
evaluate_hypothesis() result this module's adoption is based on) and
decision_log.jsonl for the logged outcome ("Hypothesis test:
starter_kbb_pct_rolling").

WHAT THIS IS: K-BB% = (strikeouts - walks) / batters faced, rolled over a
starting pitcher's trailing 8 starts, walk-forward-safe. A DIFFERENT, more
skill-isolated metric than the already-adopted home_sp_er_lN/away_sp_er_lN
(rolling earned-runs-allowed PER START -- see starting_pitcher.py) -- K-BB%
only counts outcomes the pitcher himself overwhelmingly controls (striking a
batter out, walking him), stripping out defense/ballpark/sequencing-luck
noise that a runs-allowed-based proxy can't separate from true skill. NOT
full FIP/xFIP -- those need innings-pitched (exact out-attribution per
pitcher), a materially harder parse this dataset's event files don't cleanly
support (same reason ERA itself was scoped down for the sp_er_lN feature);
this uses BATTERS FACED as an honest, unambiguous denominator instead.
Column names use `kbb_pct` throughout (never "FIP"/"xFIP") to keep that
honest.

Source: same Datasets/MLB/2010seve/*.EVA|*.EVN and Datasets/MLB/2020seve/
*.EVA|*.EVN Retrosheet play-by-play event files as starting_pitcher.py,
covering ~2010-2025. This module additionally walks each game's `play`
lines in order (not just `start`/`sub`/`data,er`) to attribute every plate
appearance to whichever pitcher was on the mound at the time -- see
research_starter_kbb_pct.py's docstring for the exact base-event-code
classification (verified against every distinct code appearing in this
dataset, spot-checked against a real game) and for why sports/mlb/research_
bullpen_arm_quality.py imports this module's build_pitcher_pa_log() rather
than re-parsing the same play-by-play a second, riskier time.

COVERAGE: same ~2010-2025 window as sp_er_lN. Games from 1990-2009 (and any
pitcher's true first tracked start within the covered window) get the
league-average fallback below -- 99.9% of the 2012-2021 backtest-eligible
window had a real (non-fallback) value on at least one side at adoption
time (research_starter_kbb_pct.py's own diagnostic run).
"""

import csv
import glob

import numpy as np
import pandas as pd

from . import config

EVENT_SUBDIRS = ["2010seve", "2020seve"]

# Same rolling window (in STARTS) as the adopted ER/start feature -- see
# research_starter_kbb_pct.py for why this is kept identical rather than
# independently tuned (keeps the two metrics comparable on one axis, not
# two, and avoids a second free parameter chosen purely to make a backtest
# look better -- exactly the "keep tuning until it turns positive" trap
# this project's methodology exists to reject).
ROLL_N_STARTS = 8

# League-average K-BB% over a trailing-8-start window, computed directly
# from every clean plate appearance extracted from the 2010seve/2020seve
# event files at adoption time (2,885,530 total PA across 74,686 starts +
# 236,632 relief appearances; starts-only K-BB% = 0.1218). Used as the
# fallback for (a) a pitcher's true first tracked start in this window, and
# (b) any game outside the ~2010-2025 event-file coverage entirely.
LEAGUE_AVG_SP_KBB_PCT = 0.1218

# Verified (research_starter_kbb_pct.py) as the COMPLETE set of Retrosheet
# event base-codes in this dataset that do NOT represent a completed plate
# appearance -- baserunning-only or record-keeping-only events.
NON_PA_BASE_CODES = {
    "NP", "SB", "SBH", "CS", "CSH", "PO", "POCS", "POCSH", "WP", "PB", "BK", "DI", "OA",
}
WALK_BASE_CODES = {"W", "IW"}


def _iter_event_files():
    for sub in EVENT_SUBDIRS:
        for ext in ("EVA", "EVN"):
            for f in sorted(glob.glob(str(config.DATA_DIR / sub / f"*.{ext}"))):
                yield f


def _extract_base_code(event: str) -> str:
    return event.split(".")[0].split("/")[0].split("+")[0].split("#")[0]


def _parse_file_pa_log(path):
    """Same play-by-play walk as research_starter_kbb_pct.py's function of
    the same name -- see that module for the verification this is based on.
    Returns a list of (game_meta, pitcher_stats, pitcher_meta) per game."""
    games = []
    cur = None
    cur_pitcher = {"home": None, "away": None}
    pstats = None
    pmeta = None

    def _new_stat():
        return {"k": 0, "bb": 0, "pa": 0}

    with open(path, encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            tag = row[0]

            if tag == "id":
                if cur is not None:
                    games.append((cur, pstats, pmeta))
                cur, cur_pitcher, pstats, pmeta = {}, {"home": None, "away": None}, {}, {}
            elif cur is None:
                continue
            elif tag == "info":
                key = row[1]
                val = row[2] if len(row) > 2 else ""
                if key in ("visteam", "hometeam", "date", "number"):
                    cur[key] = val
            elif tag == "start":
                player_id, home_flag, fieldpos = row[1], row[3], row[5]
                if fieldpos == "1":
                    side = "home" if home_flag == "1" else "away"
                    cur_pitcher[side] = player_id
                    pmeta[player_id] = {"side": side, "is_starter": True}
                    pstats.setdefault(player_id, _new_stat())
            elif tag == "sub":
                player_id, home_flag, fieldpos = row[1], row[3], row[5]
                if fieldpos == "1":
                    side = "home" if home_flag == "1" else "away"
                    cur_pitcher[side] = player_id
                    if player_id not in pmeta:
                        pmeta[player_id] = {"side": side, "is_starter": False}
                        pstats.setdefault(player_id, _new_stat())
            elif tag == "play" and len(row) >= 7:
                vh_flag, event = row[2], row[6]
                pitching_side = "home" if vh_flag == "0" else "away"
                pid = cur_pitcher.get(pitching_side)
                if pid is None:
                    continue
                base = _extract_base_code(event)
                if not base or base in NON_PA_BASE_CODES:
                    continue
                st = pstats.setdefault(pid, _new_stat())
                st["pa"] += 1
                if base == "K":
                    st["k"] += 1
                elif base in WALK_BASE_CODES:
                    st["bb"] += 1

    if cur is not None:
        games.append((cur, pstats, pmeta))
    return games


def build_pitcher_pa_log() -> pd.DataFrame:
    """
    One row per (game, pitcher-who-appeared) -- both starters and relievers
    (sports/mlb/research_bullpen_arm_quality.py's production counterpart
    reuses this exact function for its relief-appearance rows, same
    single-shared-parse discipline as the research script). Columns: date,
    game_number, team_franchise, opponent_franchise, pitcher_id, is_starter,
    k, bb, pa.
    """
    rows = []
    for f in _iter_event_files():
        for game_meta, pstats, pmeta in _parse_file_pa_log(f):
            date, number = game_meta.get("date"), game_meta.get("number", "0")
            home_raw, away_raw = game_meta.get("hometeam"), game_meta.get("visteam")
            if date is None or home_raw is None or away_raw is None:
                continue
            for pid, meta in pmeta.items():
                st = pstats.get(pid, {"k": 0, "bb": 0, "pa": 0})
                side = meta["side"]
                team_raw = home_raw if side == "home" else away_raw
                opp_raw = away_raw if side == "home" else home_raw
                rows.append({
                    "date": date, "game_number": number,
                    "team_raw": team_raw, "opponent_raw": opp_raw, "side": side,
                    "pitcher_id": pid, "is_starter": meta["is_starter"],
                    "k": st["k"], "bb": st["bb"], "pa": st["pa"],
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["team_franchise"] = df["team_raw"].apply(config.canonical_team)
    df["opponent_franchise"] = df["opponent_raw"].apply(config.canonical_team)
    return df


def _build_rolling_starter_kbb(pa_log: pd.DataFrame, n: int = ROLL_N_STARTS) -> pd.DataFrame:
    """
    Starts-only subset, sorted chronologically per pitcher, with a
    shift(1)-then-rolling(n, min_periods=1) SUM of K/BB/PA over the
    pitcher's last `n` starts (summed counts, then a ratio -- not an
    average of per-start percentages) -- same walk-forward discipline
    (current start's own result never included in its own feature) every
    other rolling feature in this project follows.
    """
    starts = pa_log[pa_log["is_starter"]].sort_values(
        ["pitcher_id", "date", "game_number"], kind="stable"
    ).reset_index(drop=True)

    grp = starts.groupby("pitcher_id", group_keys=False)
    roll_k = grp["k"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    roll_bb = grp["bb"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    roll_pa = grp["pa"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        kbb = (roll_k - roll_bb) / roll_pa
    starts["sp_kbb_pct_lN"] = kbb.where(roll_pa.fillna(0) > 0, LEAGUE_AVG_SP_KBB_PCT)

    cols = ["date", "game_number", "team_franchise", "opponent_franchise", "sp_kbb_pct_lN"]
    home_side = starts.loc[starts["side"] == "home", cols].rename(columns={
        "team_franchise": "home_franchise", "opponent_franchise": "away_franchise", "sp_kbb_pct_lN": "home_sp_kbb_pct_lN",
    })
    away_side = starts.loc[starts["side"] == "away", cols].rename(columns={
        "team_franchise": "away_franchise", "opponent_franchise": "home_franchise", "sp_kbb_pct_lN": "away_sp_kbb_pct_lN",
    })
    return home_side, away_side


def attach_starter_kbb_pct(games: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins two new columns onto `games`: `home_sp_kbb_pct_lN` /
    `away_sp_kbb_pct_lN`, each starter's rolling K-BB% over their last
    ROLL_N_STARTS starts, walk-forward-safe. Any game outside the
    ~2010-2025 event-file coverage, or not cleanly matched, gets the
    LEAGUE_AVG_SP_KBB_PCT fallback -- same discipline as
    starting_pitcher.attach_starter_quality.
    """
    pa_log = build_pitcher_pa_log()
    home_side, away_side = _build_rolling_starter_kbb(pa_log)

    key = ["date", "game_number", "home_franchise", "away_franchise"]
    home_side = home_side.dropna(subset=["home_franchise", "away_franchise"])
    away_side = away_side.dropna(subset=["home_franchise", "away_franchise"])
    if (home_side.groupby(key).size() > 1).any():
        home_side = home_side.drop_duplicates(subset=key, keep="first")
    if (away_side.groupby(key).size() > 1).any():
        away_side = away_side.drop_duplicates(subset=key, keep="first")

    merged = games.merge(home_side[key + ["home_sp_kbb_pct_lN"]], on=key, how="left")
    merged = merged.merge(away_side[key + ["away_sp_kbb_pct_lN"]], on=key, how="left")
    assert len(merged) == len(games), "attach_starter_kbb_pct must return exactly one row per input game"

    merged["home_sp_kbb_pct_lN"] = merged["home_sp_kbb_pct_lN"].fillna(LEAGUE_AVG_SP_KBB_PCT)
    merged["away_sp_kbb_pct_lN"] = merged["away_sp_kbb_pct_lN"].fillna(LEAGUE_AVG_SP_KBB_PCT)
    return merged
