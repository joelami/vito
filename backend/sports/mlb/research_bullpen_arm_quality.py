"""
RESEARCH SCRIPT -- not wired into build_features/ML_FEATURE_COLS by default
(pending the evaluate_hypothesis() outcome logged below / in decision_log.jsonl).

THIRD hypothesis test in the same MLB pitching-signal line of research as
starting_pitcher.py (adopt_cautiously) and research_bullpen_fatigue.py
(adopt_cautiously, NOT wired into ML_FEATURE_COLS -- confirmed directly via
`grep -n "bullpen" sports/mlb/features.py` before writing this file, which
returns nothing: bullpen_apps_l3d never made it into the live feature set).

HYPOTHESIS, and how it's DISTINCT from the already-tested bullpen_fatigue_
relief_appearances_l3d (a pure trailing-3-day APPEARANCE COUNT, silent on
who those pitchers actually are or how good they are): a team's bullpen
tonight is only as good as the SPECIFIC relievers actually available to
pitch, rest-adjusted -- a team with 3 lights-out relievers all rested is a
real asset regardless of how many total appearances the bullpen logged
recently; a team whose only good arms all pitched yesterday has a real
problem regardless of the raw appearance count. This script builds a
QUALITY-and-AVAILABILITY metric: for each team, each night, take the set of
relievers who look like they're on the current active bullpen (have
pitched in relief for this team recently) MINUS whoever pitched the
immediately preceding day (presumed unavailable/limited tonight under
standard bullpen-management practice), and average THEIR OWN individual
rolling skill level (K-BB%, the same defense/luck-stripped metric
research_starter_kbb_pct.py uses and justifies for starters -- reused here
rather than re-justified from scratch, and reused via direct import of that
script's build_pitcher_pa_log() rather than a second, riskier re-parse of
the same play-by-play).

SOURCE: same Retrosheet event files as every other MLB pitching feature in
this project (Datasets/MLB/2010seve, 2020seve). The per-pitcher, per-
appearance K/BB/PA counts this script rolls up are produced by
research_starter_kbb_pct.build_pitcher_pa_log() -- see that module's
docstring for the full play-by-play parsing verification (spot-checked
against the same 2020-07-28 SEA@ANA game every MLB pitching-feature script
in this project checks). This script's OWN new logic is (a) restricting to
relief appearances (is_starter == False) and computing each individual
reliever's own rolling K-BB%, and (b) the team-level roster/rest-availability
walk in `build_bullpen_arm_quality()` below.

Run standalone with:  python -m sports.mlb.research_bullpen_arm_quality
(from backend/) to print parser diagnostics and the full hypothesis-test
result without touching any live feature/config file.

OUTCOME (2026-09-03, logged in decision_log.jsonl as "Hypothesis test:
bullpen_arm_quality_available_tonight"): see the printed run / decision_log.jsonl
for the exact numbers -- filled in honestly whichever way it comes out.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from .research_starter_kbb_pct import build_pitcher_pa_log

# Relievers appear far more often than starters (who pitch on a ~5-day
# rotation) but face far fewer batters PER appearance (~3-5 vs. a starter's
# ~20+) -- a window sized in APPEARANCES needs to be wider than the
# starter feature's ROLL_N_STARTS=8 to gather a comparable total plate-
# appearance sample before the ratio is trusted. 15 trailing relief
# appearances is used here; the real average PA/appearance in this dataset
# is printed and checked in main() rather than assumed.
ROLL_N_RELIEF_APPS = 15

# How far back a reliever's most recent relief appearance can be and still
# count as "currently on this team's active bullpen" -- 30 days comfortably
# covers a full turn of a 5-man rotation's worth of team games without
# reaching back far enough to still be counting someone who was traded,
# DFA'd, or optioned to the minors since (real roster churn this dataset
# cannot see directly -- no transaction log, only appearance history -- so
# "hasn't pitched in relief for 30 days" is used as the honest, coarse
# proxy for "no longer on this specific bullpen").
ROSTER_WINDOW_DAYS = 30

# A reliever who pitched IN RELIEF on the calendar day immediately before
# tonight's game is treated as presumptively unavailable/limited -- the same
# "back-to-back-day usage" bullpen-management convention research_bullpen_
# fatigue.py's own reasoning cites, applied here as an AVAILABILITY filter
# on individual pitchers rather than a raw team-wide appearance count.
REST_EXCLUDE_DAYS = 1


def build_relief_rolling_kbb(pa_log: pd.DataFrame, n: int = ROLL_N_RELIEF_APPS):
    """
    Relief-appearances-only subset of `pa_log` (research_starter_kbb_pct's
    per-pitcher-appearance PA/K/BB log), with each pitcher's own rolling
    K-BB% over their trailing `n` relief appearances INCLUDING the current
    one -- deliberately NOT shift(1)-excluded the way the starter feature's
    own next-start prediction is, because this value is never attached to
    the SAME appearance's own game outcome. It is only ever looked up LATER,
    from `build_bullpen_arm_quality()`, by a DIFFERENT, strictly-later game
    date for a DIFFERENT team-level prediction -- at that point this
    appearance is genuinely in the past, so including its own outcome in
    "this pitcher's current form" is real information, not leakage (using
    the shift(1)-excluded value instead would just make the roster snapshot
    one appearance stale for no honesty benefit). Returns
    (relief_log_with_col, league_avg_reliever_kbb_pct).
    """
    relief = pa_log[~pa_log["is_starter"]].copy()
    relief = relief.sort_values(["pitcher_id", "date", "game_number"], kind="stable").reset_index(drop=True)

    league_total_pa = relief["pa"].sum()
    league_avg = float((relief["k"].sum() - relief["bb"].sum()) / league_total_pa) if league_total_pa else 0.0

    grp = relief.groupby("pitcher_id", group_keys=False)
    roll_k = grp["k"].apply(lambda s: s.rolling(n, min_periods=1).sum())
    roll_bb = grp["bb"].apply(lambda s: s.rolling(n, min_periods=1).sum())
    roll_pa = grp["pa"].apply(lambda s: s.rolling(n, min_periods=1).sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        kbb = (roll_k - roll_bb) / roll_pa
    kbb = kbb.where(roll_pa.fillna(0) > 0, league_avg)
    relief["relief_kbb_pct_asof"] = kbb

    print(f"[research_bullpen_arm_quality] {len(relief):,} relief-appearance rows, "
          f"avg PA/relief-appearance {relief['pa'].mean():.2f}, league-avg relief K-BB%: {league_avg:.4f}.")

    return relief[["date", "game_number", "team_franchise", "pitcher_id", "pa", "relief_kbb_pct_asof"]], league_avg


def build_bullpen_arm_quality(relief_rolling: pd.DataFrame, team_dates: pd.DataFrame,
                               roster_window_days: int = ROSTER_WINDOW_DAYS,
                               rest_exclude_days: int = REST_EXCLUDE_DAYS) -> pd.DataFrame:
    """
    Walk-forward-safe, per-(team, date) team roster/availability simulation.
    `relief_rolling`: date/team_franchise/pitcher_id/relief_kbb_pct_asof, one
    row per real relief appearance (from build_relief_rolling_kbb).
    `team_dates`: every (team_franchise, date) a feature value is needed for
    (both sides of every game in the main `games` frame).

    For each team, processed independently and chronologically: maintain a
    running dict of {pitcher_id: (last_relief_date, relief_kbb_pct_asof)},
    ingesting only relief appearances with date STRICTLY BEFORE the game
    date being evaluated (never today's own not-yet-happened bullpen usage).
    At each game date d:
      - "roster" = pitchers whose last relief appearance for this team was
        within `roster_window_days` of d (still plausibly on the active
        bullpen).
      - "available" = roster MINUS anyone whose last relief appearance was
        exactly `rest_exclude_days` (or fewer) days before d (presumptively
        gassed/unavailable).
      - bullpen_arm_kbb_pct = mean(relief_kbb_pct_asof over `available`); if
        `available` is empty, falls back to the full `roster` mean (still
        real, recency-bounded team-specific data, just without the rest
        filter); if `roster` is ALSO empty (no relief data for this team in
        the window at all -- e.g. entirely outside 2010seve/2020seve
        coverage), bullpen_arm_kbb_pct is left NaN here and filled with the
        league-average reliever K-BB% by the caller (attach_bullpen_arm_
        quality), same fallback discipline as every other event-file-derived
        MLB feature in this project.
    """
    relief_rolling = relief_rolling.sort_values(["team_franchise", "date"], kind="stable")
    by_team = {t: g[["date", "pitcher_id", "relief_kbb_pct_asof"]].reset_index(drop=True).to_dict("records")
               for t, g in relief_rolling.groupby("team_franchise")}

    results = []
    for team, dates_df in team_dates.groupby("team_franchise"):
        apps = by_team.get(team, [])
        app_idx = 0
        roster = {}  # pitcher_id -> {"last_date": Timestamp, "kbb": float}
        for d in sorted(dates_df["date"].unique()):
            while app_idx < len(apps) and apps[app_idx]["date"] < d:
                a = apps[app_idx]
                roster[a["pitcher_id"]] = {"last_date": a["date"], "kbb": a["relief_kbb_pct_asof"]}
                app_idx += 1

            active = {pid: v for pid, v in roster.items() if (d - v["last_date"]).days <= roster_window_days}
            available = {pid: v for pid, v in active.items() if (d - v["last_date"]).days > rest_exclude_days}

            if available:
                q, source = float(np.mean([v["kbb"] for v in available.values()])), "available_subset"
            elif active:
                q, source = float(np.mean([v["kbb"] for v in active.values()])), "full_roster_no_rest_filter"
            else:
                q, source = np.nan, "no_data"

            results.append({
                "team_franchise": team, "date": d, "bullpen_arm_kbb_pct": q,
                "arm_quality_source": source, "n_available": len(available), "n_roster": len(active),
            })

    return pd.DataFrame(results)


def attach_bullpen_arm_quality(games: pd.DataFrame, pa_log: pd.DataFrame = None):
    """
    `games` is sports/mlb/loader.py's load_games() output. Returns
    (games_with_two_new_cols, league_avg_reliever_kbb_pct). New columns:
    home_bullpen_arm_kbb_pct / away_bullpen_arm_kbb_pct -- higher is a
    BETTER available bullpen (same direction as the starter K-BB% feature).
    League-average-filled for any (team, date) with zero relief data in the
    lookback window (out-of-coverage games, or a team's first-ever tracked
    date). Merge key is (team_franchise, date) only -- NOT game_number --
    the same convention research_bullpen_fatigue.py's attach_bullpen_
    fatigue() already uses, so a doubleheader's two games intentionally
    share one pre-game bullpen-availability snapshot.
    """
    if pa_log is None:
        pa_log = build_pitcher_pa_log()
    relief_rolling, league_avg = build_relief_rolling_kbb(pa_log)

    team_dates = pd.concat([
        games[["home_franchise", "date"]].rename(columns={"home_franchise": "team_franchise"}),
        games[["away_franchise", "date"]].rename(columns={"away_franchise": "team_franchise"}),
    ]).drop_duplicates()

    quality = build_bullpen_arm_quality(relief_rolling, team_dates)

    n_available = int((quality["arm_quality_source"] == "available_subset").sum())
    n_fallback_roster = int((quality["arm_quality_source"] == "full_roster_no_rest_filter").sum())
    n_no_data = int((quality["arm_quality_source"] == "no_data").sum())
    print(f"[research_bullpen_arm_quality] (team, date) coverage: {n_available:,} used a real rested-"
          f"available subset, {n_fallback_roster:,} fell back to the full roster (everyone was within "
          f"rest_exclude_days), {n_no_data:,} had no relief data at all in the lookback window "
          f"(out-of-coverage or first tracked date) and get the league-average fallback.")

    home_q = quality.rename(columns={"team_franchise": "home_franchise", "bullpen_arm_kbb_pct": "home_bullpen_arm_kbb_pct"})
    away_q = quality.rename(columns={"team_franchise": "away_franchise", "bullpen_arm_kbb_pct": "away_bullpen_arm_kbb_pct"})

    merged = games.merge(home_q[["home_franchise", "date", "home_bullpen_arm_kbb_pct"]], on=["date", "home_franchise"], how="left")
    merged = merged.merge(away_q[["away_franchise", "date", "away_bullpen_arm_kbb_pct"]], on=["date", "away_franchise"], how="left")
    assert len(merged) == len(games), "attach_bullpen_arm_quality must return exactly one row per input game"

    merged["home_bullpen_arm_kbb_pct"] = merged["home_bullpen_arm_kbb_pct"].fillna(league_avg)
    merged["away_bullpen_arm_kbb_pct"] = merged["away_bullpen_arm_kbb_pct"].fillna(league_avg)
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

    hr("1. PARSE EVENT FILES -> PER-PITCHER PA LOG (shared w/ research_starter_kbb_pct)")
    pa_log = build_pitcher_pa_log()

    hr("2. LOAD + ATTACH (odds, adopted starter-ER feature, new bullpen arm-quality feature)")
    games = load_games()
    games = attach_odds(games)
    games = attach_starter_quality(games)  # already-adopted -- part of BOTH baseline and variant
    games, league_avg = attach_bullpen_arm_quality(games, pa_log=pa_log)

    has_odds = games["Home Odds Close"].notna() & games["Away Odds Close"].notna()
    n_window = int(has_odds.sum())
    real_bpq = (games["home_bullpen_arm_kbb_pct"] != league_avg) | (games["away_bullpen_arm_kbb_pct"] != league_avg)
    n_window_real = int((has_odds & real_bpq).sum())
    print(f"Backtest-eligible games (real odds, 2012-2021): {n_window:,}")
    print(f"...with a real (non-fallback) bullpen arm-quality value on at least one side: "
          f"{n_window_real:,} ({n_window_real / n_window * 100:.1f}%)")
    print(f"League-average reliever K-BB% fallback: {league_avg:.4f}")
    print(games[["home_bullpen_arm_kbb_pct", "away_bullpen_arm_kbb_pct"]].describe().to_string())

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
        feats["home_bullpen_arm_kbb_pct"] = games.set_index("game_id")["home_bullpen_arm_kbb_pct"].reindex(feats["game_id"]).values
        feats["away_bullpen_arm_kbb_pct"] = games.set_index("game_id")["away_bullpen_arm_kbb_pct"].reindex(feats["game_id"]).values
        # positive = home's AVAILABLE bullpen is better (higher K-BB%) than away's -> favors home,
        # same sign convention as rest_diff/sp_er_diff_lN/sp_kbb_pct_diff_lN (positive always favors home)
        feats["bullpen_arm_kbb_pct_diff"] = feats["home_bullpen_arm_kbb_pct"] - feats["away_bullpen_arm_kbb_pct"]

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
        return metrics, oos

    hr("3. BASELINE (current ML_FEATURE_COLS -- bullpen_fatigue was NOT adopted, stays out)")
    baseline, oos_base = run_pipeline(ML_FEATURE_COLS, "BASELINE")

    hr("4. VARIANT (+ home_bullpen_arm_kbb_pct, away_bullpen_arm_kbb_pct, bullpen_arm_kbb_pct_diff)")
    variant_cols = ML_FEATURE_COLS + ["home_bullpen_arm_kbb_pct", "away_bullpen_arm_kbb_pct", "bullpen_arm_kbb_pct_diff"]
    variant, oos_var = run_pipeline(variant_cols, "VARIANT")

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
        name="bullpen_arm_quality_available_tonight",
        reasoning=(
            "The already-tested bullpen_fatigue_relief_appearances_l3d (adopt_cautiously, NOT wired "
            "into ML_FEATURE_COLS) is a pure trailing-3-day appearance COUNT -- it is silent on WHO the "
            "relievers actually are or how good they are, treating a bullpen of three lights-out arms "
            "identically to a bullpen of replacement-level arms as long as both logged the same number "
            "of recent appearances. Real bullpen usage is a rest-adjusted AVAILABILITY-and-QUALITY "
            "question: a team's real bullpen strength tonight depends on which specific relievers are "
            "both (a) actually on the current active bullpen and (b) rested enough to be used, weighted "
            "by how good those specific pitchers currently are. This is a genuinely different mechanism "
            "from the appearance-count feature, not a repeat of it (confirmed: bullpen_fatigue_relief_"
            "appearances_l3d is the only bullpen hypothesis in decision_log.jsonl before this one; "
            "grep -o '\"decision\": \"[^\"]*\"' decision_log.jsonl confirms no 'bullpen_arm_quality' or "
            "similar name has been tested). Built from the same Retrosheet play-by-play already parsed "
            "for the starter K-BB% feature (research_starter_kbb_pct.py), reusing its verified per-"
            "pitcher-appearance K/BB/PA counts (imported directly, not re-parsed) restricted to relief "
            "appearances, rolled per individual reliever, then combined into a team-level snapshot via a "
            "walk-forward-safe roster/rest-exclusion simulation (build_bullpen_arm_quality) -- excluding "
            "any reliever who pitched the immediately preceding calendar day as presumptively "
            "unavailable, the same back-to-back-day bullpen-management convention already cited as this "
            "project's own reasoning for the fatigue-count feature, applied here to WHO is available "
            "instead of just HOW MANY total appearances occurred."
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
