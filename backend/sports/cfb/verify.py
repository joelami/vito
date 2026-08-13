"""
Standalone Phase-1 verification script for the CFB module. Runs the exact
same pipeline shape used for NFL (loader -> power ratings -> features ->
walk-forward ML -> residual stds -> backtest -> summarize) and prints every
number needed to judge whether this dataset supports a credible CFB model.
Not wired into main.py / the live app - backend-only validation pass.

Run with:  python -m sports.cfb.verify   (from backend/)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import odds_math, ensemble, edge_finder, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from sports.cfb import config as cfb_config
from sports.cfb.loader import load_games
from sports.cfb.features import build_features, ML_FEATURE_COLS


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
    print(f"Rows after load/clean (completed=1, valid scores): {len(games):,}")
    print(f"Season range: {games['season'].min()} - {games['season'].max()}")
    print(f"Unique home_team strings: {games['home_team'].nunique()}")
    print(f"Unique away_team strings: {games['away_team'].nunique()}")

    has_spread = games["Home Line Close"].notna()
    has_total = games["Total Score Close"].notna()
    has_ml = games["Home Odds Close"].notna() & games["Away Odds Close"].notna()
    print(f"\nGames with a spread:    {has_spread.sum():,} ({has_spread.mean()*100:.1f}%)")
    print(f"Games with a total:     {has_total.sum():,} ({has_total.mean()*100:.1f}%)")
    print(f"Games with moneylines:  {has_ml.sum():,} ({has_ml.mean()*100:.1f}%)")
    print(f"Games with ANY odds:    {(has_spread | has_total | has_ml).sum():,} "
          f"({(has_spread | has_total | has_ml).mean()*100:.1f}%)")
    print(f"Games with NO odds at all: {(~(has_spread | has_total | has_ml)).sum():,}")

    print("\nOdds coverage is NOT evenly spread across seasons - by season (spread/total/ML coverage %):")
    cov = games.assign(has_spread=has_spread, has_total=has_total, has_ml=has_ml)
    cov_by_season = cov.groupby("season").agg(
        n=("game_id", "size"),
        spread_pct=("has_spread", "mean"),
        total_pct=("has_total", "mean"),
        ml_pct=("has_ml", "mean"),
    )
    print((cov_by_season * [1, 100, 100, 100]).round(1).to_string())

    # cross-check season_for_date against the CSV's own season column, on the RAW file
    raw = pd.read_csv(cfb_config.DATA_PATH)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    derived = raw["date"].apply(cfb_config.season_for_date)
    mismatches = (derived != raw["season"]).sum()
    print(f"\nseason_for_date(date) vs CSV season column mismatches: {mismatches} / {len(raw)}")

    # ------------------------------------------------------------------
    # 2. Power ratings
    # ------------------------------------------------------------------
    hr("2. POWER RATINGS")
    rating_cfg = PowerRatingConfig(
        k_factor=cfb_config.ELO_K_FACTOR,
        start_rating=cfb_config.ELO_START_RATING,
        home_field_adv=cfb_config.HOME_FIELD_ADV_ELO,
        season_regression=cfb_config.SEASON_REGRESSION,
        mov_mult_base=cfb_config.MOV_MULT_BASE,
        mov_mult_divisor=cfb_config.MOV_MULT_DIVISOR,
    )
    print(f"Config: k={rating_cfg.k_factor}, hfa={rating_cfg.home_field_adv}, "
          f"season_regression={rating_cfg.season_regression}, "
          f"mov_base={rating_cfg.mov_mult_base}, mov_divisor={rating_cfg.mov_mult_divisor}")

    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", neutral_col="is_neutral_venue",
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
    print(f"ATS-form NaN before fill would have been present for "
          f"{(games['home_covers_close'].isna()).mean()*100:.1f}% of team-games "
          f"(no spread that game) - ats_pct_l10 rolling windows are that much sparser than NFL's.")

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
    elo_pred_margin = oos["rating_diff_pre"] / cfb_config.ELO_POINTS_PER_MARGIN
    elo_margin_corr = float(np.corrcoef(elo_pred_margin, oos["actual_margin"])[0, 1])
    print(f"Margin: pure Elo-implied margin vs actual correlation: {elo_margin_corr:.4f}")
    print(f"Total:  ML pred vs actual correlation:    {corr('predicted_total', 'actual_total'):.4f}")
    print(f"Total:  naive pace baseline vs actual correlation: {corr('naive_total', 'actual_total'):.4f}")

    mae_ml_margin = float((oos['actual_margin'] - oos['predicted_margin']).abs().mean())
    mae_elo_margin = float((oos['actual_margin'] - elo_pred_margin).abs().mean())
    mae_ml_total = float((oos['actual_total'] - oos['predicted_total']).abs().mean())
    mae_naive_total = float((oos['actual_total'] - oos['naive_total']).abs().mean())
    print(f"\nMAE margin: ML={mae_ml_margin:.2f}  Elo-only={mae_elo_margin:.2f}")
    print(f"MAE total:  ML={mae_ml_total:.2f}  naive-pace={mae_naive_total:.2f}")

    # ------------------------------------------------------------------
    # 5. Residual stds & backtest
    # ------------------------------------------------------------------
    hr("5. ENSEMBLE RESIDUAL STDS")
    stds = ensemble.compute_residual_stds(oos, cfb_config.ELO_POINTS_PER_MARGIN)
    print(stds)

    hr("6. BACKTEST")
    print("NOTE ON CLV: this dataset has only one odds snapshot per game (no true "
          "open vs close). run_backtest prices bets at cfg.price_point and separately "
          "looks up the hardcoded 'Home Odds Close' etc. columns for CLV - since both "
          "point at the SAME assumed price here, CLV is trivially ~0 for every bet. This "
          "backtest can only speak to hit-rate/ROI against one assumed price point, not "
          "genuine closing-line value, unlike the NFL backtest.")

    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0,
        allowed_confidence=("Medium", "High"),
        price_point="Close",  # only price point that exists in this data
    )
    bets = backtest.run_backtest(oos, stds, cfb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"\nTotal qualifying bets found (edge >= {bt_cfg.min_edge_pct}%, "
          f"confidence in {bt_cfg.allowed_confidence}): {len(bets):,}")

    if bets.empty:
        print("No bets cleared the edge/confidence thresholds - nothing further to summarize.")
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

    hr("DONE")


if __name__ == "__main__":
    main()
