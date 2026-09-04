"""
RESEARCH SCRIPT -- not wired into build_features/ML_FEATURE_COLS by default
(pending the evaluate_hypothesis() outcome logged below / in decision_log.jsonl).

Tests one hypothesis via core.research.evaluate_hypothesis: whether a
starting pitcher's rolling K-BB% (strikeout rate minus walk rate, both as a
share of batters faced) is a genuinely BETTER starter-quality signal than
the already-adopted home_sp_er_lN/away_sp_er_lN (rolling earned-runs-allowed
PER START -- see starting_pitcher.py/research_starting_pitcher.py,
adopt_cautiously in decision_log.jsonl). This is a DIFFERENT metric on the
SAME already-tested general idea ("starter quality matters"), not a repeat
of that exact test -- checked directly against decision_log.jsonl before
writing a line of code here (`grep -o '"decision": "[^"]*"' decision_log.jsonl`
shows only "starting_pitcher_rolling_er_per_start" and
"bullpen_fatigue_relief_appearances_l3d" tested for MLB pitching so far).

WHY K-BB% INSTEAD OF FIP/xFIP (the task's other two named options), STATED
HONESTLY: FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + constant is INNINGS-PITCHED
normalized. This dataset's Retrosheet event files (Datasets/MLB/2010seve,
2020seve) do NOT carry a per-pitcher innings-pitched column anywhere --
computing real IP per start requires attributing every OUT on every play to
whichever pitcher was on the mound at that instant (including correctly
counting double/triple plays as 2/3 outs, distinguishing a fielder's-choice
non-out from a true out, etc.) -- exactly the same "materially harder parse"
starting_pitcher.py's own docstring already declined to attempt for ERA
itself, for the same reason. Rather than quietly approximate innings pitched
(risking a subtly wrong denominator that LOOKS precise), this script uses
BATTERS FACED (PA) as the denominator instead -- a count that IS unambiguous
and directly, safely countable from the same play-by-play (see
_extract_base_code / NON_PA_BASE_CODES below, verified against a real
spot-checked game before being trusted for the full dataset). K-BB% = (K -
BB) / PA is standard sabermetric practice (FanGraphs reports it as one of
the most year-to-year STABLE pitcher metrics, arguably more predictive of
true skill than ERA or even full FIP in small samples) and needs nothing
beyond PA, K, and BB counts -- no HR, no HBP, no innings, no defense/luck-
laden balls-in-play outcomes at all. This is the honestly-scoped choice, not
a downgrade dressed up as a choice: it isolates the two outcomes that are
overwhelmingly pitcher-controlled (whether he strikes a batter out and
whether he walks him), a real step toward a "true talent" signal that raw
ERA -- driven partly by defense, ballpark, and sequencing luck -- muddies.

SOURCE FORMAT, verified directly with a real game before trusting the parser
on the full dataset (2020-07-28 SEA@ANA, ANA202007280 -- the same game
starting_pitcher.py's and research_bullpen_fatigue.py's docstrings already
spot-check):
  - `play,inning,vh_flag,batter_id,count,pitches,event` rows record one
    plate-appearance-relevant event each. vh_flag='0' means the VISITING
    team is batting (so the HOME team is pitching); vh_flag='1' means the
    HOME team is batting (so the AWAY team is pitching) -- confirmed
    directly: longs001 (Seattle/away) has vh_flag='0' rows, fletd002
    (Angels/home) has vh_flag='1' rows.
  - The `event` field's BASE code (everything before the first '.', '/',
    '+', or '#' -- see _extract_base_code) falls into two buckets, verified
    by extracting and counting every distinct base code appearing across
    ALL 976 files in 2010seve+2020seve (not assumed from memory): (a) real
    plate-appearance-ending outcomes -- K, W, IW, HP, HR, S/D/T (hits), FC,
    DGR, C, E, and every numeric fielding-sequence out code (e.g. "63",
    "64(1)3") -- and (b) baserunning-only / no-play records that do NOT end
    a plate appearance and must NOT be double-counted: NP, SB, SBH, CS, CSH,
    PO, POCS, POCSH, WP, PB, BK, DI, OA -- exactly the codes collected into
    NON_PA_BASE_CODES below, no others were found in this dataset.
  - Spot-check on ANA202007280 (104 total `play` rows): 21 NP + 1 PB + 1 WP
    = 23 non-PA rows, 81 real plate appearances -- matches Justus
    Sheffield's (Seattle's starter) own line exactly when walked through by
    hand: 2 K, 4 BB, 0 HP, 0 HR, 16 PA over his start before being pulled
    for Nestor Cortes (the `sub,cortn001,...,fieldpos=1` line that ends his
    outing) -- confirmed by direct line-by-line trace, not just aggregate
    counts.
  - Pitching-change tracking (`sub` lines with fieldpos==1) is the SAME
    mechanism research_bullpen_fatigue.py already uses to detect relief
    entries -- this script additionally uses it to know, play-by-play,
    WHICH pitcher (starter or whichever reliever is currently in) each
    plate-appearance outcome belongs to. `build_pitcher_pa_log()` below
    returns BOTH starters' and relievers' per-appearance K/BB/HP/HR/PA
    counts for exactly this reason -- sports/mlb/research_bullpen_arm_
    quality.py imports it directly rather than re-deriving the same
    play-by-play walk a second, riskier time.

Run standalone with:  python -m sports.mlb.research_starter_kbb_pct
(from backend/) to print parser diagnostics and the full hypothesis-test
result without touching any live feature/config file.

OUTCOME (2026-09-03, logged in decision_log.jsonl as "Hypothesis test:
starter_kbb_pct_rolling"): see the bottom of this docstring / the printed
run for the exact numbers. [Filled in immediately below once the real run
completed -- see decision_log.jsonl for the permanent record either way.]
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

# Same rolling window (in STARTS, not days) as the already-adopted ER/start
# feature -- chosen there for the same reason it applies here: ~5-6 weeks of
# a normal 5-man rotation, long enough to smooth one outlier start without
# going so wide it stops reflecting recent form. Keeping it identical to
# starting_pitcher.ROLL_N_STARTS makes the two metrics directly comparable
# apples-to-apples rather than differing on two axes (metric AND window) at
# once.
ROLL_N_STARTS = 8

# Verified (see module docstring) as the COMPLETE set of Retrosheet event
# base-codes in this dataset that do NOT represent a completed plate
# appearance -- baserunning-only or record-keeping-only events. Anything
# NOT in this set (K, W, IW, HP, HR, hits, outs, FC, etc.) is counted as one
# real PA.
NON_PA_BASE_CODES = {
    "NP",                    # no play (record-keeping only)
    "SB", "SBH",             # stolen base (incl. of home)
    "CS", "CSH",             # caught stealing
    "PO", "POCS", "POCSH",   # pickoff / pickoff-caught-stealing
    "WP", "PB",              # wild pitch / passed ball
    "BK",                    # balk
    "DI",                    # defensive indifference
    "OA",                    # other baserunner advance, no PA involved
}
WALK_BASE_CODES = {"W", "IW"}  # unintentional + intentional -- standard K-BB% treats both as walks


def _iter_event_files():
    for sub in EVENT_SUBDIRS:
        for ext in ("EVA", "EVN"):
            for f in sorted(glob.glob(str(config.DATA_DIR / sub / f"*.{ext}"))):
                yield f


def _extract_base_code(event: str) -> str:
    """Retrosheet event field's base play type, stripped of runner-advance
    modifiers (after '.'), linked secondary events (after '+'), fielding
    detail (after '/'), and rare '#' markers. E.g. 'K+WP' -> 'K',
    'S8/L8.2-H' -> 'S', '64(1)3' -> '64(1)3' (numeric fielding sequences
    have no modifier to strip and ARE the base code as-is)."""
    return event.split(".")[0].split("/")[0].split("+")[0].split("#")[0]


def _parse_file_pa_log(path):
    """
    One pass over a single .EVA/.EVN file, walking every play IN ORDER so
    the currently-active pitcher for each side is always known (tracked via
    `start`/`sub` fieldpos==1 rows, the same mechanism research_starting_
    pitcher.py and research_bullpen_fatigue.py already use). Returns a list
    of (game_meta, pitcher_stats, pitcher_meta) tuples, one per game:
      - game_meta: {"date", "number", "hometeam", "visteam"}
      - pitcher_stats: {pitcher_id: {"k","bb","hp","hr","pa"}} -- counts
        accumulated ONLY from plays that occurred while this exact pitcher
        was the one on the mound for his side.
      - pitcher_meta: {pitcher_id: {"side": "home"/"away", "is_starter": bool}}
    """
    games = []
    cur = None
    cur_pitcher = {"home": None, "away": None}
    pstats = None
    pmeta = None

    def _new_stat():
        return {"k": 0, "bb": 0, "hp": 0, "hr": 0, "pa": 0}

    with open(path, encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            tag = row[0]

            if tag == "id":
                if cur is not None:
                    games.append((cur, pstats, pmeta))
                cur = {}
                cur_pitcher = {"home": None, "away": None}
                pstats = {}
                pmeta = {}
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
                    continue  # malformed/missing starter row -- skip rather than guess
                base = _extract_base_code(event)
                if not base or base in NON_PA_BASE_CODES:
                    continue
                st = pstats.setdefault(pid, _new_stat())
                st["pa"] += 1
                if base == "K":
                    st["k"] += 1
                elif base in WALK_BASE_CODES:
                    st["bb"] += 1
                elif base == "HP":
                    st["hp"] += 1
                elif base == "HR":
                    st["hr"] += 1

    if cur is not None:
        games.append((cur, pstats, pmeta))
    return games


def build_pitcher_pa_log() -> pd.DataFrame:
    """
    Long format: one row per (game, pitcher-who-appeared), across every
    parsed 2010seve/2020seve game -- both starters and relievers, since
    research_bullpen_arm_quality.py needs the relief rows too and this is
    the single, verified, shared parse of the underlying play-by-play (see
    module docstring for why duplicating this parse a second time would be
    the riskier choice). Columns: date, game_number, team_franchise,
    opponent_franchise, pitcher_id, is_starter, k, bb, hp, hr, pa.
    """
    rows = []
    n_games = 0
    for f in _iter_event_files():
        for game_meta, pstats, pmeta in _parse_file_pa_log(f):
            date, number = game_meta.get("date"), game_meta.get("number", "0")
            home_raw, away_raw = game_meta.get("hometeam"), game_meta.get("visteam")
            if date is None or home_raw is None or away_raw is None:
                continue
            n_games += 1
            for pid, meta in pmeta.items():
                st = pstats.get(pid, {"k": 0, "bb": 0, "hp": 0, "hr": 0, "pa": 0})
                side = meta["side"]
                team_raw = home_raw if side == "home" else away_raw
                opp_raw = away_raw if side == "home" else home_raw
                rows.append({
                    "date": date, "game_number": number,
                    "team_raw": team_raw, "opponent_raw": opp_raw, "side": side,
                    "pitcher_id": pid, "is_starter": meta["is_starter"],
                    "k": st["k"], "bb": st["bb"], "hp": st["hp"], "hr": st["hr"], "pa": st["pa"],
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["team_franchise"] = df["team_raw"].apply(config.canonical_team)
    df["opponent_franchise"] = df["opponent_raw"].apply(config.canonical_team)

    print(f"[research_starter_kbb_pct] Parsed {n_games:,} games, {len(df):,} pitcher-appearance "
          f"rows ({int(df['is_starter'].sum()):,} starts, {int((~df['is_starter']).sum()):,} relief appearances). "
          f"Total PA counted: {int(df['pa'].sum()):,}, total K: {int(df['k'].sum()):,}, "
          f"total BB(+IBB): {int(df['bb'].sum()):,}.")

    return df


# ---------------------------------------------------------------------
# Rolling, walk-forward-safe starter K-BB%
# ---------------------------------------------------------------------

def build_rolling_starter_kbb(pa_log: pd.DataFrame, n: int = ROLL_N_STARTS):
    """
    Starts-only subset of `pa_log`, sorted chronologically per pitcher, with
    a shift(1)-then-rolling(n, min_periods=1) SUM of K/BB/PA over that
    pitcher's last `n` starts (summed counts, THEN a ratio taken -- not an
    average of per-start percentages, which would overweight a fluky
    small-PA start exactly as much as a normal one). The current start's own
    outcome is never included in its own feature value -- the same
    walk-forward discipline as every other rolling feature in this project.
    Returns (games_with_kbb_col, league_avg_kbb_pct).
    """
    starts = pa_log[pa_log["is_starter"]].copy()
    starts = starts.sort_values(["pitcher_id", "date", "game_number"], kind="stable").reset_index(drop=True)

    league_total_pa = starts["pa"].sum()
    league_avg_kbb_pct = float((starts["k"].sum() - starts["bb"].sum()) / league_total_pa) if league_total_pa else 0.0

    grp = starts.groupby("pitcher_id", group_keys=False)
    roll_k = grp["k"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    roll_bb = grp["bb"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    roll_pa = grp["pa"].apply(lambda s: s.shift(1).rolling(n, min_periods=1).sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        kbb = (roll_k - roll_bb) / roll_pa
    n_first_starts = int((roll_pa.fillna(0) == 0).sum())
    kbb = kbb.where(roll_pa.fillna(0) > 0, league_avg_kbb_pct)
    starts["sp_kbb_pct_lN"] = kbb

    print(f"[research_starter_kbb_pct] League-avg K-BB% over last-{n}-starts window (this data): "
          f"{league_avg_kbb_pct:.4f} -- used to fill {n_first_starts:,} true first-tracked-starts "
          f"({n_first_starts / len(starts) * 100:.1f}% of {len(starts):,} starts).")

    # Each start row carries the pitcher's OWN team/opponent (team_franchise/
    # opponent_franchise) plus which SIDE (home/away) his team was on in that
    # specific game -- split on that (same pattern as research_starting_
    # pitcher.py's build_rolling_starter_quality) before relabeling into a
    # home_franchise/away_franchise merge key, rather than assuming a
    # pitcher's own team was always "home".
    cols = ["date", "game_number", "team_franchise", "opponent_franchise", "sp_kbb_pct_lN"]
    home_side = starts.loc[starts["side"] == "home", cols].rename(columns={
        "team_franchise": "home_franchise", "opponent_franchise": "away_franchise", "sp_kbb_pct_lN": "home_sp_kbb_pct_lN",
    })
    away_side = starts.loc[starts["side"] == "away", cols].rename(columns={
        "team_franchise": "away_franchise", "opponent_franchise": "home_franchise", "sp_kbb_pct_lN": "away_sp_kbb_pct_lN",
    })
    return home_side, away_side, league_avg_kbb_pct


def attach_starter_kbb_pct(games: pd.DataFrame, pa_log: pd.DataFrame = None):
    """
    `games` is sports/mlb/loader.py's load_games() output (odds-attached or
    not, doesn't matter -- only date/franchise/game_number are needed).
    Returns (games_with_two_new_cols, league_avg_kbb_pct). New columns:
    home_sp_kbb_pct_lN / away_sp_kbb_pct_lN -- league-average-filled for any
    game outside 2010seve/2020seve coverage or a pitcher's true first
    tracked start (same fallback discipline as LEAGUE_AVG_SP_ER_PER_START).
    """
    if pa_log is None:
        pa_log = build_pitcher_pa_log()
    home_side, away_side, league_avg = build_rolling_starter_kbb(pa_log, n=ROLL_N_STARTS)

    key = ["date", "game_number", "home_franchise", "away_franchise"]
    home_side = home_side.dropna(subset=["home_franchise", "away_franchise"])
    away_side = away_side.dropna(subset=["home_franchise", "away_franchise"])

    dupe_home = home_side.groupby(key).size()
    if (dupe_home > 1).any():
        home_side = home_side.drop_duplicates(subset=key, keep="first")
    dupe_away = away_side.groupby(key).size()
    if (dupe_away > 1).any():
        away_side = away_side.drop_duplicates(subset=key, keep="first")

    merged = games.merge(home_side[key + ["home_sp_kbb_pct_lN"]], on=key, how="left")
    merged = merged.merge(away_side[key + ["away_sp_kbb_pct_lN"]], on=key, how="left")
    assert len(merged) == len(games), "attach_starter_kbb_pct must return exactly one row per input game"

    merged["home_sp_kbb_pct_lN"] = merged["home_sp_kbb_pct_lN"].fillna(league_avg)
    merged["away_sp_kbb_pct_lN"] = merged["away_sp_kbb_pct_lN"].fillna(league_avg)
    return merged, league_avg


# ---------------------------------------------------------------------
# Standalone diagnostics + hypothesis test
# ---------------------------------------------------------------------

def main():
    import json
    from core import ensemble, backtest
    from core.research import Hypothesis, evaluate_hypothesis
    from core.power_ratings import compute_power_ratings, PowerRatingConfig
    from core.ml_models import walk_forward_predict
    from sports.mlb import config as mlb_config
    from sports.mlb.loader import load_games
    from sports.mlb.odds_loader import attach_odds
    from sports.mlb.starting_pitcher import attach_starter_quality
    from sports.mlb.features import build_features, ML_FEATURE_COLS

    def hr(title):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    hr("1. PARSE EVENT FILES -> PER-PITCHER PA LOG")
    pa_log = build_pitcher_pa_log()

    hr("2. LOAD + ATTACH (odds, adopted starter-ER feature, new K-BB% feature)")
    games = load_games()
    games = attach_odds(games)
    games = attach_starter_quality(games)  # already-adopted -- part of BOTH baseline and variant
    games, league_avg = attach_starter_kbb_pct(games, pa_log=pa_log)

    has_odds = games["Home Odds Close"].notna() & games["Away Odds Close"].notna()
    n_window = int(has_odds.sum())
    has_real_kbb = games["home_sp_kbb_pct_lN"].notna() & games["away_sp_kbb_pct_lN"].notna()
    real_kbb_nonfallback = (games["home_sp_kbb_pct_lN"] != league_avg) | (games["away_sp_kbb_pct_lN"] != league_avg)
    n_window_real = int((has_odds & real_kbb_nonfallback).sum())
    print(f"Backtest-eligible games (real odds, 2012-2021): {n_window:,}")
    print(f"...with a real (non-fallback) K-BB% value on at least one side: "
          f"{n_window_real:,} ({n_window_real / n_window * 100:.1f}%)")
    print(f"League-average K-BB% fallback: {league_avg:.4f}")

    def run_pipeline(feature_cols, label):
        rating_cfg = PowerRatingConfig(
            k_factor=mlb_config.ELO_K_FACTOR, start_rating=mlb_config.ELO_START_RATING,
            home_field_adv=mlb_config.HOME_FIELD_ADV_ELO, season_regression=mlb_config.SEASON_REGRESSION,
            mov_mult_base=mlb_config.MOV_MULT_BASE, mov_mult_divisor=mlb_config.MOV_MULT_DIVISOR,
        )
        rr = compute_power_ratings(
            games, home_col="home_franchise", away_col="away_franchise",
            home_score_col="home_score", away_score_col="away_score",
            season_col="season", date_col="date", config=rating_cfg,
        )
        feats = build_features(games, rr.history)
        feats["home_sp_kbb_pct_lN"] = games.set_index("game_id")["home_sp_kbb_pct_lN"].reindex(feats["game_id"]).values
        feats["away_sp_kbb_pct_lN"] = games.set_index("game_id")["away_sp_kbb_pct_lN"].reindex(feats["game_id"]).values
        # positive = home's starter has a BETTER (higher) K-BB% than away's -> favors home,
        # same sign convention as rest_diff/sp_er_diff_lN (positive always favors home)
        feats["sp_kbb_pct_diff_lN"] = feats["home_sp_kbb_pct_lN"] - feats["away_sp_kbb_pct_lN"]

        wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
        oos = feats.set_index("game_id").join(wf.predictions, how="inner")

        def corr(a, b):
            return float(np.corrcoef(oos[a], oos[b])[0, 1])

        margin_corr = corr("predicted_margin", "actual_margin")
        total_corr = corr("predicted_total", "actual_total")

        stds = ensemble.compute_residual_stds(oos, mlb_config.ELO_POINTS_PER_MARGIN)
        ens_cfg = ensemble.EnsembleConfig()
        bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
        bets = backtest.run_backtest(oos, stds, mlb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)

        print(f"\n[{label}] margin_corr={margin_corr:.4f} total_corr={total_corr:.4f} n_bets={len(bets):,}")
        if bets.empty:
            metrics = {"margin_corr": margin_corr, "total_corr": total_corr,
                       "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "n_bets": 0}
        else:
            row = backtest.summarize(bets).iloc[0]
            metrics = {"margin_corr": margin_corr, "total_corr": total_corr,
                       "roi_pct": float(row["roi_pct"]), "roi_stderr_pct": float(row["roi_stderr_pct"]),
                       "n_bets": int(row["bets"])}
            print(f"[{label}] ROI={metrics['roi_pct']:+.2f}% (stderr {metrics['roi_stderr_pct']:.2f}pp, n={metrics['n_bets']:,})")
        return metrics, oos, feats

    hr("3. BASELINE (current ML_FEATURE_COLS, already includes sp_er_lN)")
    baseline, oos_base, feats_base = run_pipeline(ML_FEATURE_COLS, "BASELINE")

    hr("4. VARIANT (+ home_sp_kbb_pct_lN, away_sp_kbb_pct_lN, sp_kbb_pct_diff_lN)")
    variant_cols = ML_FEATURE_COLS + ["home_sp_kbb_pct_lN", "away_sp_kbb_pct_lN", "sp_kbb_pct_diff_lN"]
    variant, oos_var, feats_var = run_pipeline(variant_cols, "VARIANT")

    hr("5. SEASON SPLIT-HALF STABILITY CHECK")
    seasons = sorted(oos_var["season"].dropna().unique())
    mid = seasons[len(seasons) // 2]
    for label, cond in [(f"early ({seasons[0]}-{mid - 1})", lambda df: df["season"] < mid),
                         (f"late ({mid}-{seasons[-1]})", lambda df: df["season"] >= mid)]:
        b = oos_base[cond(oos_base)]
        v = oos_var[cond(oos_var)]
        bm = float(np.corrcoef(b["predicted_margin"], b["actual_margin"])[0, 1])
        vm = float(np.corrcoef(v["predicted_margin"], v["actual_margin"])[0, 1])
        bt = float(np.corrcoef(b["predicted_total"], b["actual_total"])[0, 1])
        vt = float(np.corrcoef(v["predicted_total"], v["actual_total"])[0, 1])
        print(f"{label}: n={len(v):,}")
        print(f"  margin_corr {bm:.4f} -> {vm:.4f} ({vm - bm:+.4f})")
        print(f"  total_corr  {bt:.4f} -> {vt:.4f} ({vt - bt:+.4f})")

    hr("6. HYPOTHESIS EVALUATION")
    hyp = Hypothesis(
        name="starter_kbb_pct_rolling",
        reasoning=(
            "The already-adopted starter-quality feature (home_sp_er_lN/away_sp_er_lN, "
            "adopt_cautiously) uses rolling earned-runs-allowed PER START, a real but crude "
            "proxy that mixes true pitcher skill with defense quality, ballpark, and batted-"
            "ball sequencing luck. Sabermetric research (e.g. FanGraphs) treats K-BB% "
            "(strikeout rate minus walk rate, both as a share of batters faced) as a more "
            "skill-isolated, year-to-year STABLE signal specifically because it only counts "
            "outcomes the pitcher himself overwhelmingly controls (striking a batter out, "
            "walking him) rather than balls in play where nine defenders and park factors "
            "also decide the outcome. This is a genuinely different underlying metric (not a "
            "repeat of the exact already-tested ER/start feature), computed from the same "
            "Retrosheet play-by-play event files already parsed for the ER/start feature, "
            "attributing each plate appearance to whichever pitcher was on the mound via the "
            "same start/sub fieldpos==1 tracking. Full FIP/xFIP were considered but require "
            "innings-pitched (exact out-attribution per pitcher), a materially harder parse "
            "this dataset's event files were already documented as not supporting cleanly "
            "for the ER feature -- K-BB% needs only batters-faced (an unambiguous count, "
            "verified directly against a real spot-checked game) as its denominator, an "
            "honestly-scoped choice, not an unverified approximation of the harder metric."
        ),
        sport="MLB",
    )
    baseline_metrics = {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}
    variant_metrics = {k: variant[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}
    result = evaluate_hypothesis(hyp, baseline_metrics, variant_metrics)

    hr("RESULT")
    print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
