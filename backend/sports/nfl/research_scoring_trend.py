"""
THROWAWAY research script — NOT wired into main.py/pipeline.py/harness.py.
Tests one hypothesis via core/research.py's disciplined evaluate_hypothesis()
loop: does adding short-window (L3) scoring-trend features to the totals-side
feature set improve NFL total-points prediction?

Mirrors sports/cfb/verify.py's shape (loader -> power ratings -> features ->
walk-forward ML -> residual stds -> backtest -> summarize) but runs it TWICE
in one session — once with baseline features.py unmodified, once with a
local variant that adds pf_l3/pa_l3/pf_trend/pa_trend — so baseline and
variant are directly comparable (same data load, same walk-forward config,
same backtest config).

pipeline.py's build_nfl_pipeline() was read ONLY as a reference for how NFL
wires loader/features/ratings/config together (weather attach, config
constants). Not imported, not modified.

Run with:  python -m sports.nfl.research_scoring_trend   (from backend/)

OUTCOME (2026-08-10, logged in decision_log.jsonl): recommendation
"inconclusive". Statistical fit barely moved (margin_corr +0.0006, total_corr
+0.0003 — both below the 0.005 noise floor, i.e. "didn't move") while
backtest ROI dropped from -0.30% to -1.89% (-1.60pp), a delta LARGER than the
baseline's own +-1.54pp standard error, so it can't be waved off as noise
either. Per core/research.py's rules this is not "reject" (ROI didn't
improve, so it isn't the suspicious overfit signature) and not "adopt"/
"adopt_cautiously" (the ROI move isn't within noise) — genuinely
inconclusive. NOT ADOPTED. features.py was left untouched; this file is kept
only as a record of a tested, non-adopted experiment.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nfl import config as nfl_config
from sports.nfl.loader import load_games
from sports.nfl import features as nfl_features
from sports.nfl.features import ML_FEATURE_COLS, naive_score_features, pythagorean_win_pct
from sports.nfl.weather import attach_weather

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

TREND_WINDOW = 3  # short "recent form" window, contrasted against the existing 10-game window


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Variant feature builder: same shape as features.build_features, but adds
# pf_l3/pa_l3 (short trailing window) and pf_trend/pa_trend (L3 minus L10,
# a "is scoring pace heating up or cooling down" signal) for both teams.
# Reuses features._team_game_log/_rolling_form (same package) so the shared
# base columns (pf_l10, pa_l10, ats_pct_l10, etc.) are bit-for-bit identical
# to the baseline path - only the new trend columns are added on top, same
# shift(1)-before-rolling walk-forward-safe discipline as every other
# rolling feature in features.py.
# ---------------------------------------------------------------------------
def build_features_with_trend(games: pd.DataFrame, rating_history: pd.DataFrame) -> pd.DataFrame:
    log = nfl_features._team_game_log(games)
    log = nfl_features._rolling_form(log)  # adds ats_pct_l10, win_pct_l10, pf_l10, pa_l10, rest_days, streak

    grp = log.groupby("team", group_keys=False)
    log["pf_l3"] = grp["points_for"].apply(
        lambda s: s.shift(1).rolling(TREND_WINDOW, min_periods=1).mean()
    )
    log["pa_l3"] = grp["points_against"].apply(
        lambda s: s.shift(1).rolling(TREND_WINDOW, min_periods=1).mean()
    )
    log["pf_trend"] = log["pf_l3"] - log["pf_l10"]
    log["pa_trend"] = log["pa_l3"] - log["pa_l10"]

    base_cols = ["ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak",
                 "pf_l3", "pa_l3", "pf_trend", "pa_trend"]

    home_feats = log[log["is_home"]][["game_id"] + base_cols].rename(columns={
        "ats_pct_l10": "home_ats_pct_l10", "win_pct_l10": "home_win_pct_l10",
        "pf_l10": "home_pf_l10", "pa_l10": "home_pa_l10",
        "rest_days": "home_rest_days", "streak": "home_streak",
        "pf_l3": "home_pf_l3", "pa_l3": "home_pa_l3",
        "pf_trend": "home_pf_trend", "pa_trend": "home_pa_trend",
    })
    away_feats = log[~log["is_home"]][["game_id"] + base_cols].rename(columns={
        "ats_pct_l10": "away_ats_pct_l10", "win_pct_l10": "away_win_pct_l10",
        "pf_l10": "away_pf_l10", "pa_l10": "away_pa_l10",
        "rest_days": "away_rest_days", "streak": "away_streak",
        "pf_l3": "away_pf_l3", "pa_l3": "away_pa_l3",
        "pf_trend": "away_pf_trend", "pa_trend": "away_pa_trend",
    })

    out = games.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out = out.merge(rating_history[["home_rating_pre", "away_rating_pre", "rating_diff_pre",
                                     "elo_home_win_prob"]], left_on="game_id", right_index=True, how="left")

    out["home_rest_days"] = out["home_rest_days"].fillna(7)
    out["away_rest_days"] = out["away_rest_days"].fillna(7)
    out["home_ats_pct_l10"] = out["home_ats_pct_l10"].fillna(0.5)
    out["away_ats_pct_l10"] = out["away_ats_pct_l10"].fillna(0.5)
    out["home_win_pct_l10"] = out["home_win_pct_l10"].fillna(0.5)
    out["away_win_pct_l10"] = out["away_win_pct_l10"].fillna(0.5)
    out["home_streak"] = out["home_streak"].fillna(0)
    out["away_streak"] = out["away_streak"].fillna(0)
    out["home_pf_l10"] = out["home_pf_l10"].fillna(22.0)
    out["home_pa_l10"] = out["home_pa_l10"].fillna(22.0)
    out["away_pf_l10"] = out["away_pf_l10"].fillna(22.0)
    out["away_pa_l10"] = out["away_pa_l10"].fillna(22.0)
    # a team's first 1-2 games have no L3 history either - same neutral league-average fill as L10
    out["home_pf_l3"] = out["home_pf_l3"].fillna(22.0)
    out["home_pa_l3"] = out["home_pa_l3"].fillna(22.0)
    out["away_pf_l3"] = out["away_pf_l3"].fillna(22.0)
    out["away_pa_l3"] = out["away_pa_l3"].fillna(22.0)
    # no L3 vs L10 history yet -> neutral "no trend" fill, not 0.0-by-coincidence
    out["home_pf_trend"] = out["home_pf_trend"].fillna(0.0)
    out["home_pa_trend"] = out["home_pa_trend"].fillna(0.0)
    out["away_pf_trend"] = out["away_pf_trend"].fillna(0.0)
    out["away_pa_trend"] = out["away_pa_trend"].fillna(0.0)

    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]

    out["naive_total"], out["naive_margin"] = naive_score_features(
        out["home_pf_l10"], out["home_pa_l10"], out["away_pf_l10"], out["away_pa_l10"]
    )
    out["home_pyth_pct"] = pythagorean_win_pct(out["home_pf_l10"], out["home_pa_l10"])
    out["away_pyth_pct"] = pythagorean_win_pct(out["away_pf_l10"], out["away_pa_l10"])
    out["pyth_pct_diff"] = out["home_pyth_pct"] - out["away_pyth_pct"]
    return out


TREND_FEATURE_COLS = [
    "home_pf_l3", "home_pa_l3", "away_pf_l3", "away_pa_l3",
    "home_pf_trend", "home_pa_trend", "away_pf_trend", "away_pa_trend",
]
VARIANT_FEATURE_COLS = ML_FEATURE_COLS + TREND_FEATURE_COLS


# ---------------------------------------------------------------------------
# One shared "run the pipeline, return the 4 metrics" routine, called once
# per feature-builder (baseline, variant) so both go through byte-identical
# power ratings / walk-forward config / backtest config.
# ---------------------------------------------------------------------------
def run_pipeline(games: pd.DataFrame, rr, feats: pd.DataFrame, feature_cols: list, label: str) -> dict:
    hr(f"WALK-FORWARD ML — {label}")
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    print(f"Seasons predicted out-of-sample: {wf.seasons_predicted}")
    print(f"OOS rows: {len(wf.predictions):,}")
    print(f"Margin residual std: {wf.margin_residual_std:.2f}")
    print(f"Total residual std:  {wf.total_residual_std:.2f}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])
    print(f"Margin: ML pred vs actual correlation: {margin_corr:.4f}")
    print(f"Total:  ML pred vs actual correlation: {total_corr:.4f}")

    hr(f"ENSEMBLE RESIDUAL STDS — {label}")
    stds = ensemble.compute_residual_stds(oos, nfl_config.ELO_POINTS_PER_MARGIN)
    print(stds)

    hr(f"BACKTEST — {label}")
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"))
    bets = backtest.run_backtest(oos, stds, nfl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"Qualifying bets (edge >= {bt_cfg.min_edge_pct}%, confidence in {bt_cfg.allowed_confidence}, "
          f"priced at {bt_cfg.price_point}): {len(bets):,}")

    if bets.empty:
        raise RuntimeError(f"{label}: no bets cleared thresholds — cannot compute ROI.")

    summary = backtest.summarize(bets)
    roi_pct = float(summary["roi_pct"].iloc[0])
    roi_stderr_pct = float(summary["roi_stderr_pct"].iloc[0])
    n_bets = int(summary["bets"].iloc[0])
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    return {
        "margin_corr": margin_corr, "total_corr": total_corr,
        "roi_pct": roi_pct, "roi_stderr_pct": roi_stderr_pct,
        "n_bets": n_bets,
    }


def main():
    # ------------------------------------------------------------------
    # 0. Hypothesis object — construction fails loudly without real reasoning
    # ------------------------------------------------------------------
    hypothesis = Hypothesis(
        name="scoring_trend_l3_vs_l10",
        reasoning=(
            "Totals correlation (~0.21) is far weaker than margin correlation (~0.36), a "
            "diagnosed weakness in METHODOLOGY.md. Current features only carry a flat L10 "
            "scoring average with no signal on whether a team's scoring pace is trending up "
            "or down recently. Short-window (L3) form diverging from a longer baseline (L10) "
            "is an established momentum/hot-cold signal in sports analytics, distinct from "
            "the flat average, and targets scoring pace specifically — exactly where totals "
            "prediction is weakest. Computed walk-forward-safe from data already loaded."
        ),
        sport="NFL",
    )
    print(f"Hypothesis: {hypothesis.name}\nReasoning: {hypothesis.reasoning}\n")

    # ------------------------------------------------------------------
    # 1. Load (shared by both baseline and variant)
    # ------------------------------------------------------------------
    hr("1. LOAD")
    games = load_games()
    games = attach_weather(games)  # disk-cached; same as build_nfl_pipeline() — keeps weather features live in both runs
    print(f"Rows after load/clean: {len(games):,}")
    print(f"Season range: {games['season'].min()} - {games['season'].max()}")

    # ------------------------------------------------------------------
    # 2. Power ratings (shared)
    # ------------------------------------------------------------------
    hr("2. POWER RATINGS")
    rating_cfg = PowerRatingConfig(
        k_factor=nfl_config.ELO_K_FACTOR, start_rating=nfl_config.ELO_START_RATING,
        home_field_adv=nfl_config.HOME_FIELD_ADV_ELO, season_regression=nfl_config.SEASON_REGRESSION,
        mov_mult_base=nfl_config.MOV_MULT_BASE, mov_mult_divisor=nfl_config.MOV_MULT_DIVISOR,
    )
    print(f"Config: k={rating_cfg.k_factor}, hfa={rating_cfg.home_field_adv}, "
          f"season_regression={rating_cfg.season_regression}, "
          f"mov_base={rating_cfg.mov_mult_base}, mov_divisor={rating_cfg.mov_mult_divisor}, "
          f"elo_points_per_margin={nfl_config.ELO_POINTS_PER_MARGIN}")

    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", neutral_col="is_neutral_venue",
        config=rating_cfg,
    )

    # ------------------------------------------------------------------
    # 3a. BASELINE features (current features.py, unmodified)
    # ------------------------------------------------------------------
    hr("3a. BASELINE FEATURES (features.py unmodified)")
    baseline_feats = nfl_features.build_features(games, rr.history)
    print(f"Feature rows: {len(baseline_feats):,}, ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")

    # ------------------------------------------------------------------
    # 3b. VARIANT features (+ pf_l3/pa_l3/pf_trend/pa_trend)
    # ------------------------------------------------------------------
    hr("3b. VARIANT FEATURES (+ scoring trend L3-vs-L10)")
    variant_feats = build_features_with_trend(games, rr.history)
    print(f"Feature rows: {len(variant_feats):,}, VARIANT_FEATURE_COLS: {len(VARIANT_FEATURE_COLS)} "
          f"({len(TREND_FEATURE_COLS)} new: {TREND_FEATURE_COLS})")

    # sanity check: variant's shared/base columns must be bit-identical to baseline's
    shared_cols = [c for c in ML_FEATURE_COLS if c in baseline_feats.columns and c in variant_feats.columns]
    mism = 0
    for c in shared_cols:
        if not np.allclose(baseline_feats[c].fillna(-999999), variant_feats[c].fillna(-999999)):
            mism += 1
            print(f"  MISMATCH in shared column: {c}")
    print(f"Shared base-column identity check: {len(shared_cols) - mism}/{len(shared_cols)} identical.")

    # ------------------------------------------------------------------
    # 4. Run both pipelines through walk-forward ML -> stds -> backtest
    # ------------------------------------------------------------------
    baseline = run_pipeline(games, rr, baseline_feats, ML_FEATURE_COLS, "BASELINE")
    variant = run_pipeline(games, rr, variant_feats, VARIANT_FEATURE_COLS, "VARIANT (+scoring trend)")

    baseline_metrics = {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}
    variant_metrics = {k: variant[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}

    # ------------------------------------------------------------------
    # 5. Evaluate via core/research.py's disciplined loop (auto-logs to decision_log.jsonl)
    # ------------------------------------------------------------------
    hr("5. HYPOTHESIS EVALUATION")
    print(f"Baseline n_bets={baseline['n_bets']:,}  Variant n_bets={variant['n_bets']:,}")
    result = evaluate_hypothesis(hypothesis, baseline_metrics, variant_metrics)

    hr("RESULT")
    import json
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
