"""
Standalone verification script for the MLB module. Runs loader -> odds join
-> power ratings (with the calibration check) -> features -> walk-forward ML
-> residual std computation -> real backtest, and prints every number needed
to judge whether this dataset supports a credible MLB model. Not wired into
main.py / the live app - backend-only validation pass.

======================================================================
ODDS COVERAGE - READ BEFORE READING THE BACKTEST NUMBERS BELOW
======================================================================
The Retrosheet game logs this module's loader (loader.py) reads
(Datasets/MLB/gl*/*.txt) contain schedule and result data ONLY - no spread,
total, or moneyline column anywhere, for any of the 83,215 games loaded from
that source. An earlier version of this script stopped there and explicitly
declined to run a backtest, because running core.backtest against an
all-None market would misrepresent "cannot test" as "tested and found
nothing."

That is no longer the whole picture. A separate real odds archive
(Datasets/MLB/archive/oddsDataMLB.csv, see odds_loader.py's docstring for
the full verification of what it is) was found and is joined in below. It
covers 2012-2021 ONLY - about 22.8k of the 83.2k games in this module, real
two-sided moneyline/run-line/total prices. So:
  - For the ~72.6% of games from 1990-2011 and 2022-2025, this module still
    has NO market price and NEVER claims one - those games simply carry NaN
    in every odds column and produce zero backtest opportunities, exactly
    as before.
  - For the ~27.4% of games from 2012-2021, a REAL backtest now runs below,
    following the exact same walk-forward pipeline (loader -> features ->
    ratings -> ML -> ensemble -> edge_finder -> backtest) already validated
    for NFL and CFB. See section 6 below for the actual results - reported
    exactly as they came out, whichever direction they point.
  - This is a SINGLE odds snapshot per game (no open vs. close - see
    odds_loader.py), so CLV computed against it is trivially ~0 for every
    bet, the same limitation CFB's backtest has and flags for the same
    reason.
  - A pick made on a game outside 2012-2021 (which today means essentially
    every current/future game, since the archive stops in 2021 and doesn't
    reach the present) still has ZERO backtested track record behind it -
    this module's ML/ratings are trained on the full 1990-2025 history, but
    the backtest below only validates the SLICE of that history where a
    real market price existed to check against.

Run with:  python -m sports.mlb.verify   (from backend/)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, edge_finder, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from sports.mlb import config as mlb_config
from sports.mlb.loader import load_games
from sports.mlb.odds_loader import attach_odds
from sports.mlb.starting_pitcher import attach_starter_quality, LEAGUE_AVG_SP_ER_PER_START
from sports.mlb.features import build_features, ML_FEATURE_COLS


pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    hr("1. LOAD")
    games = load_games()
    print(f"Rows after load/clean (valid scores, ties dropped): {len(games):,}")
    print(f"Season range: {games['season'].min()} - {games['season'].max()}")

    print("\nGames per season range check (min/max date per season - confirms no "
          "postseason/cross-year dates leaked into this game-log source):")
    per_season = games.groupby("season")["date"].agg(["min", "max", "count"])
    print(per_season.to_string())
    cross_year = (per_season["min"].dt.year != per_season.index) | (per_season["max"].dt.year != per_season.index)
    print(f"\nSeasons where min/max date's calendar year != season label: {cross_year.sum()} "
          f"(confirms season_for_date = dt.year is safe for this data)")

    print("\n--- Franchise / relocation mapping ---")
    raw_codes = sorted(set(games["home_team"]) | set(games["away_team"]))
    canonical = sorted(set(games["home_franchise"]) | set(games["away_franchise"]))
    print(f"Distinct raw 3-letter codes in source: {len(raw_codes)}")
    print(f"Resolved to canonical franchises: {len(canonical)}")
    print(f"Mapping applied (raw code -> canonical): {mlb_config.FRANCHISE_CANONICAL}")

    print("\nPer-code first/last game date (this is the actual evidence the mapping "
          "in config.py was built from - a code that stops appearing exactly when "
          "another starts is the relocation signal):")
    long_codes = pd.concat([
        games[["home_team", "date"]].rename(columns={"home_team": "code"}),
        games[["away_team", "date"]].rename(columns={"away_team": "code"}),
    ])
    code_range = long_codes.groupby("code")["date"].agg(["min", "max", "count"]).sort_index()
    print(code_range.to_string())

    print(f"\nTies dropped (true final-score ties, no winner - see loader.py docstring): "
          f"83,228 raw rows - {len(games):,} kept = {83228 - len(games)}")

    # ------------------------------------------------------------------
    # 1b. Odds join
    # ------------------------------------------------------------------
    hr("1b. ODDS JOIN (Datasets/MLB/archive/, 2012-2021 only - see odds_loader.py)")
    games = attach_odds(games)
    has_ml = games["Home Odds Close"].notna() & games["Away Odds Close"].notna()
    has_spread_num = games["Home Line Close"].notna()
    has_spread_full = has_spread_num & games["Home Line Odds Close"].notna() & games["Away Line Odds Close"].notna()
    has_total = games["Total Score Close"].notna()
    print(f"Games with a real two-sided moneyline: {has_ml.sum():,} ({has_ml.mean()*100:.1f}%)")
    print(f"Games with a run-line number:          {has_spread_num.sum():,} ({has_spread_num.mean()*100:.1f}%)")
    print(f"Games with a bettable run line (number + both-sided juice): "
          f"{has_spread_full.sum():,} ({has_spread_full.mean()*100:.1f}%)")
    print(f"Games with a real two-sided total:     {has_total.sum():,} ({has_total.mean()*100:.1f}%)")

    print("\nOdds coverage by season (moneyline / run-line-bettable / total %):")
    cov = games.assign(has_ml=has_ml, has_spread_full=has_spread_full, has_total=has_total)
    cov_by_season = cov.groupby("season").agg(
        n=("game_id", "size"),
        ml_pct=("has_ml", "mean"),
        spread_pct=("has_spread_full", "mean"),
        total_pct=("has_total", "mean"),
    )
    print((cov_by_season * [1, 100, 100, 100]).round(1).to_string())
    print("\nRun-line values are essentially always exactly +/-1.5 runs (99.0% of non-null "
          "rows in the source archive) - see odds_loader.py's docstring. Unlike a football "
          "spread, the pricing signal here lives almost entirely in the two-sided juice, "
          "not in a moving line number.")
    print("SINGLE snapshot per game - no open-vs-close distinction in this source, so CLV "
          "below is trivially ~0 for every bet (same limitation CFB's backtest has).")

    # ------------------------------------------------------------------
    # 1c. Starting pitcher rolling ER/start (adopt_cautiously, see
    # sports/mlb/starting_pitcher.py and decision_log.jsonl)
    # ------------------------------------------------------------------
    hr("1c. STARTING PITCHER ROLLING ER/START (Datasets/MLB/2010seve+2020seve event files)")
    print("NOT innings-normalized ERA - earned runs allowed PER START, rolling over a "
          "pitcher's last 8 starts. See starting_pitcher.py's module docstring for why "
          "this is deliberately scoped down from true ERA.")
    games = attach_starter_quality(games)
    real_sp = (games["home_sp_er_lN"] != LEAGUE_AVG_SP_ER_PER_START) | (games["away_sp_er_lN"] != LEAGUE_AVG_SP_ER_PER_START)
    window_games = has_ml  # the 2012-2021 odds-covered / backtest-eligible slice
    n_window = int(window_games.sum())
    n_window_real = int((window_games & real_sp).sum())
    print(f"Backtest-eligible games (real odds, 2012-2021): {n_window:,}")
    if n_window:
        print(f"...with a real (non-fallback) starter-quality value on both sides: "
              f"{n_window_real:,} ({n_window_real / n_window * 100:.1f}%)")
    else:
        print(f"...with a real (non-fallback) starter-quality value on both sides: "
              f"{n_window_real:,} (n/a - no odds-covered games)")
    print(f"League-average fallback used for out-of-coverage games / a pitcher's true first "
          f"tracked start: {LEAGUE_AVG_SP_ER_PER_START} earned runs allowed/start.")

    # ------------------------------------------------------------------
    # 2. Power ratings
    # ------------------------------------------------------------------
    hr("2. POWER RATINGS")
    rating_cfg = PowerRatingConfig(
        k_factor=mlb_config.ELO_K_FACTOR,
        start_rating=mlb_config.ELO_START_RATING,
        home_field_adv=mlb_config.HOME_FIELD_ADV_ELO,
        season_regression=mlb_config.SEASON_REGRESSION,
        mov_mult_base=mlb_config.MOV_MULT_BASE,
        mov_mult_divisor=mlb_config.MOV_MULT_DIVISOR,
    )
    print(f"Config: k={rating_cfg.k_factor}, hfa={rating_cfg.home_field_adv}, "
          f"season_regression={rating_cfg.season_regression}, "
          f"mov_base={rating_cfg.mov_mult_base}, mov_divisor={rating_cfg.mov_mult_divisor}")

    actual_home_win_rate = games["home_win"].mean()
    print(f"\nActual home win rate, full dataset: {actual_home_win_rate:.4f} "
          f"(the well-documented MLB figure is ~0.54 - this config's home_field_adv "
          f"was calibrated against this exact number, see config.py)")

    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        config=rating_cfg,
    )
    hist = rr.history.join(games.set_index("game_id")[["home_win", "date", "season"]])

    print("\nCalibration check (elo_home_win_prob decile vs actual home-win rate), full history:")
    hist["decile"] = pd.qcut(hist["elo_home_win_prob"], 10, duplicates="drop")
    calib = hist.groupby("decile", observed=True).agg(
        n=("home_win", "size"),
        mean_pred=("elo_home_win_prob", "mean"),
        actual_home_win_rate=("home_win", "mean"),
    )
    print(calib.to_string(float_format=lambda x: f"{x:.4f}"))
    calib_err = (calib["mean_pred"] - calib["actual_home_win_rate"]).abs().mean()
    print(f"\nMean |predicted - actual| across deciles: {calib_err:.4f}")

    # ------------------------------------------------------------------
    # 3. Features
    # ------------------------------------------------------------------
    hr("3. FEATURES")
    feats = build_features(games, rr.history)
    print(f"Feature rows: {len(feats):,}, columns added: "
          f"{len([c for c in feats.columns if c not in games.columns])}")
    print("No ats_pct_l10-equivalent feature is built here - odds only cover 2012-2021 "
          "(27% of history), too sparse a window for a full-history rolling ATS feature "
          "the way NFL's is built. home_covers_close/over_result_close are still computed "
          "by odds_loader.py and available for anyone who wants to add one later.")
    print("home_sp_er_lN/away_sp_er_lN/sp_er_diff_lN (starting pitcher rolling earned runs "
          "allowed per start, NOT innings-normalized ERA - see starting_pitcher.py) ARE "
          "included below - adopt_cautiously, see decision_log.jsonl.")

    # ------------------------------------------------------------------
    # 4. Walk-forward ML
    # ------------------------------------------------------------------
    hr("4. WALK-FORWARD ML")
    wf = walk_forward_predict(feats, ML_FEATURE_COLS, min_train_seasons=3)
    print(f"Seasons predicted out-of-sample: {wf.seasons_predicted}")
    print(f"OOS rows: {len(wf.predictions):,}")
    print(f"Margin residual std: {wf.margin_residual_std:.2f}")
    print(f"Total residual std:  {wf.total_residual_std:.2f}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    print(f"\nMargin: ML pred vs actual correlation:    {corr('predicted_margin', 'actual_margin'):.4f}")
    print(f"Margin: naive (rating_diff/elo_ppm) vs actual correlation: "
          f"{corr('rating_diff_pre', 'actual_margin'):.4f}  (rating_diff_pre, unscaled, for direction only)")
    elo_pred_margin = oos["rating_diff_pre"] / mlb_config.ELO_POINTS_PER_MARGIN
    elo_margin_corr = float(np.corrcoef(elo_pred_margin, oos["actual_margin"])[0, 1])
    print(f"Margin: pure Elo-implied margin vs actual correlation: {elo_margin_corr:.4f}")
    print(f"Total:  ML pred vs actual correlation:    {corr('predicted_total', 'actual_total'):.4f}")
    print(f"Total:  naive pace baseline vs actual correlation: {corr('naive_total', 'actual_total'):.4f}")

    mae_ml_margin = float((oos['actual_margin'] - oos['predicted_margin']).abs().mean())
    mae_elo_margin = float((oos['actual_margin'] - elo_pred_margin).abs().mean())
    mae_naive_margin = float((oos['actual_margin'] - oos['naive_margin']).abs().mean())
    mae_ml_total = float((oos['actual_total'] - oos['predicted_total']).abs().mean())
    mae_naive_total = float((oos['actual_total'] - oos['naive_total']).abs().mean())
    print(f"\nMAE margin: ML={mae_ml_margin:.2f}  Elo-only={mae_elo_margin:.2f}  naive-pace={mae_naive_margin:.2f}")
    print(f"MAE total:  ML={mae_ml_total:.2f}  naive-pace={mae_naive_total:.2f}")

    # ------------------------------------------------------------------
    # 5. Residual stds
    # ------------------------------------------------------------------
    hr("5. ENSEMBLE RESIDUAL STDS")
    stds = ensemble.compute_residual_stds(oos, mlb_config.ELO_POINTS_PER_MARGIN)
    print(stds)

    # ------------------------------------------------------------------
    # 6. Real backtest (2012-2021 odds-covered games only)
    # ------------------------------------------------------------------
    hr("6. BACKTEST (odds-covered games only, 2012-2021)")
    print("NOTE ON CLV: this dataset has only one odds snapshot per game (no true "
          "open vs close). run_backtest prices bets at cfg.price_point and separately "
          "looks up the hardcoded 'Home Odds Close' etc. columns for CLV - since both "
          "point at the SAME assumed price here, CLV is trivially ~0 for every bet. This "
          "backtest can only speak to hit-rate/ROI against one real recorded price point, "
          "not genuine closing-line value - identical caveat to CFB's backtest.")
    print("\nAlso note: only games with odds (2012-2021) can produce a bet here - "
          "the ML/ratings model itself was still trained walk-forward across the FULL "
          "1990-2025 history above; this section just restricts to the slice where a "
          "market price exists to check the model against.")

    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0,
        allowed_confidence=("Medium", "High"),
        price_point="Close",  # only price point that exists in this data
    )
    bets = backtest.run_backtest(oos, stds, mlb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"\nOOS rows with a real moneyline: {(oos['Home Odds Close'].notna() & oos['Away Odds Close'].notna()).sum():,}")
    print(f"Total qualifying bets found (edge >= {bt_cfg.min_edge_pct}%, "
          f"confidence in {bt_cfg.allowed_confidence}): {len(bets):,}")

    if bets.empty:
        print("No bets cleared the edge/confidence thresholds - nothing further to summarize.")
        hr("DONE")
        return

    hr("6a. OVERALL SUMMARY")
    print(backtest.summarize(bets).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("6b. SUMMARY BY MARKET")
    print(backtest.summarize(bets, group_cols=["market"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("6c. SUMMARY BY MARKET x EDGE BUCKET")
    bets["edge_bucket"] = bets["edge_pct"].apply(backtest.edge_bucket)
    print(backtest.summarize(bets, group_cols=["market", "edge_bucket"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("6d. SUMMARY BY SEASON")
    print(backtest.summarize(bets, group_cols=["season"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("7. BOTTOM LINE — READ THIS, NOT JUST THE ROI NUMBER ABOVE")
    print("This backtest covers 2012-2021 only (the odds archive's range) - about 27% of "
          "this module's 1990-2025 history. It is real (real market prices, real walk-"
          "forward-honest out-of-sample predictions, real settlement against actual final "
          "scores) but it is NOT evidence about how this model performs on any game from "
          "1990-2011, 2022-2025, or any future/live game - which today means essentially "
          "every game this module would actually be used to score, since the archive stops "
          "in 2021. A positive or negative number above describes a specific decade in the "
          "past, not a live track record.")

    hr("DONE")


if __name__ == "__main__":
    main()
