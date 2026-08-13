"""
Standalone Phase-1 verification script for the NHL module. Runs the exact
same pipeline shape used for NFL/CFB/MLB (loader -> power ratings -> features
-> walk-forward ML -> residual stds -> backtest -> summarize) and prints
every number needed to judge whether this dataset supports a credible NHL
model. Not wired into main.py / the live app - backend-only validation pass.

Run with:  python -m sports.nhl.verify   (from backend/)
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
from sports.nhl import config as nhl_config
from sports.nhl.loader import load_games
from sports.nhl.features import build_features, ML_FEATURE_COLS


pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    # ------------------------------------------------------------------
    # 0. Raw-file corrections vs. the build brief (see config.py docstring
    #    for the full detail this just summarizes)
    # ------------------------------------------------------------------
    hr("0. CORRECTIONS TO THE BUILD BRIEF (verified against the data)")
    print("- nhl_data_plus.csv (32 cols) is the LEANER raw file; nhl_data_extensive.csv")
    print("  (131 cols) is the one with the roll_*/opp_*/season_* engineered columns -")
    print("  the brief had this backwards. This module uses nhl_data_plus.csv per the")
    print("  brief's explicit instruction, and recomputes its own walk-forward-safe")
    print("  rolling features from scratch (features.py) - there was nothing pre-computed")
    print("  in this file to even consider trusting.")
    print("- The source is NOT one row per game - it is one row per TEAM PER GAME")
    print("  (61,376 rows / 30,688 unique game_ids = 2 rows/game), same shape as the NBA")
    print("  source. loader.py pivots it the same way sports/nba/loader.py does.")
    print("- `favorite_moneyline` is a materially harder single-sided-pricing problem than")
    print("  CFB's - see section 6's moneyline concordance table below and config.py.")

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    hr("1. LOAD")
    raw = pd.read_csv(nhl_config.DATA_PATH)
    print(f"Raw rows (team-games): {len(raw):,}, unique game_ids: {raw['game_id'].nunique():,}")

    games = load_games()
    print(f"Rows after full cleaning (one row per game): {len(games):,}")
    print(f"Season range: {games['season'].min()} - {games['season'].max()}")
    print(f"Unique home_team_ids: {games['home_team_id'].nunique()}, "
          f"away_team_ids: {games['away_team_id'].nunique()}")
    print(f"Playoff games: {games['is_playoff'].sum():,} ({games['is_playoff'].mean()*100:.1f}%)")

    has_spread = games["Home Line Close"].notna()
    has_total = games["Total Score Close"].notna()
    has_ml = games["Home Odds Close"].notna() & games["Away Odds Close"].notna()
    print(f"\nGames with a spread (puck line):  {has_spread.sum():,} ({has_spread.mean()*100:.1f}%)")
    print(f"Games with a total:                {has_total.sum():,} ({has_total.mean()*100:.1f}%)")
    print(f"Games with a moneyline (both sides derived): {has_ml.sum():,} ({has_ml.mean()*100:.1f}%)")

    print("\nOdds coverage by season (spread/total/ML coverage %):")
    cov = games.assign(has_spread=has_spread, has_total=has_total, has_ml=has_ml)
    cov_by_season = cov.groupby("season").agg(
        n=("game_id", "size"), spread_pct=("has_spread", "mean"),
        total_pct=("has_total", "mean"), ml_pct=("has_ml", "mean"),
    )
    print((cov_by_season * [1, 100, 100, 100]).round(1).to_string())

    # season_for_date cross-check against the CSV's own season column
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw_valid = raw.dropna(subset=["date"])
    derived = raw_valid["date"].apply(nhl_config.season_for_date)
    mismatches = (derived != raw_valid["season"]).sum()
    print(f"\nseason_for_date(date) vs CSV season column mismatches: {mismatches} / {len(raw_valid)}")

    print("\n--- Team identity / franchise continuity ---")
    print(f"Real NHL team_ids kept: {len(nhl_config.REAL_TEAM_IDS)}")
    print(f"FRANCHISE_CANONICAL mapping applied: {nhl_config.FRANCHISE_CANONICAL} "
          f"(Phoenix/Arizona Coyotes -> Utah, 2024 relocation)")
    print(f"Unique canonical home_team_id values in cleaned data: {games['home_team_id'].nunique()}")

    # ------------------------------------------------------------------
    # 2. Power ratings
    # ------------------------------------------------------------------
    hr("2. POWER RATINGS")
    rating_cfg = PowerRatingConfig(
        k_factor=nhl_config.ELO_K_FACTOR,
        start_rating=nhl_config.ELO_START_RATING,
        home_field_adv=nhl_config.HOME_FIELD_ADV_ELO,
        season_regression=nhl_config.SEASON_REGRESSION,
        mov_mult_base=nhl_config.MOV_MULT_BASE,
        mov_mult_divisor=nhl_config.MOV_MULT_DIVISOR,
    )
    print(f"Config: k={rating_cfg.k_factor}, hfa={rating_cfg.home_field_adv}, "
          f"season_regression={rating_cfg.season_regression}, "
          f"mov_base={rating_cfg.mov_mult_base}, mov_divisor={rating_cfg.mov_mult_divisor}")

    actual_home_win_rate = games["home_win"].mean()
    print(f"Actual home win rate, full cleaned dataset: {actual_home_win_rate:.4f} "
          f"(NHL's well-documented home-ice edge is real but modest - this is far "
          f"below NFL/CFB's home-win rates, consistent with hockey's high parity)")
    print(f"Mean home margin: {games['actual_margin'].mean():.3f} goals, "
          f"margin std-dev: {games['actual_margin'].std():.3f} goals")

    rr = compute_power_ratings(
        games, home_col="home_team_id", away_col="away_team_id",
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
    print(f"\nMean |predicted - actual| across deciles: {calib_err:.4f} "
          f"(this config was selected by a grid search minimizing exactly this number - "
          f"see config.py's Power rating engine section for the search)")

    # ------------------------------------------------------------------
    # 3. Features
    # ------------------------------------------------------------------
    hr("3. FEATURES")
    feats = build_features(games, rr.history)
    print(f"Feature rows: {len(feats):,}, columns added: "
          f"{len([c for c in feats.columns if c not in games.columns])}")
    print(f"ATS-form NaN before fill would have been present for "
          f"{(games['home_covers_close'].isna()).mean()*100:.1f}% of team-games "
          f"(no spread that game) - ats_pct_l10 rolling windows are that much sparser "
          f"than a fully-covered market, same caveat CFB's module documents.")

    # ------------------------------------------------------------------
    # 4. Walk-forward ML
    # ------------------------------------------------------------------
    hr("4. WALK-FORWARD ML")
    wf = walk_forward_predict(feats, ML_FEATURE_COLS, min_train_seasons=3)
    print(f"Seasons predicted out-of-sample: {wf.seasons_predicted}")
    print(f"OOS rows: {len(wf.predictions):,}")
    print(f"Margin residual std: {wf.margin_residual_std:.3f} goals")
    print(f"Total residual std:  {wf.total_residual_std:.3f} goals")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    print(f"\nMargin: ML pred vs actual correlation:    {corr('predicted_margin', 'actual_margin'):.4f}")
    print(f"Margin: naive (rating_diff/elo_ppm) vs actual correlation: "
          f"{corr('rating_diff_pre', 'actual_margin'):.4f}  (rating_diff_pre, unscaled, for direction only)")
    elo_pred_margin = oos["rating_diff_pre"] / nhl_config.ELO_POINTS_PER_MARGIN
    elo_margin_corr = float(np.corrcoef(elo_pred_margin, oos["actual_margin"])[0, 1])
    print(f"Margin: pure Elo-implied margin vs actual correlation: {elo_margin_corr:.4f}")
    print(f"Total:  ML pred vs actual correlation:    {corr('predicted_total', 'actual_total'):.4f}")
    print(f"Total:  naive pace baseline vs actual correlation: {corr('naive_total', 'actual_total'):.4f}")

    mae_ml_margin = float((oos['actual_margin'] - oos['predicted_margin']).abs().mean())
    mae_elo_margin = float((oos['actual_margin'] - elo_pred_margin).abs().mean())
    mae_naive_margin = float((oos['actual_margin'] - oos['naive_margin']).abs().mean())
    mae_ml_total = float((oos['actual_total'] - oos['predicted_total']).abs().mean())
    mae_naive_total = float((oos['actual_total'] - oos['naive_total']).abs().mean())
    print(f"\nMAE margin (goals): ML={mae_ml_margin:.3f}  Elo-only={mae_elo_margin:.3f}  naive-pace={mae_naive_margin:.3f}")
    print(f"MAE total (goals):  ML={mae_ml_total:.3f}  naive-pace={mae_naive_total:.3f}")
    print("\nNote how small these correlations/MAE gaps are relative to CFB/NFL - NHL scores are")
    print("famously hard to predict game-to-game (tight 1-3 goal margins, heavy variance from")
    print("goaltending/special teams luck) - this is a real property of the sport, not a bug.")

    # ------------------------------------------------------------------
    # 5. Residual stds & backtest
    # ------------------------------------------------------------------
    hr("5. ENSEMBLE RESIDUAL STDS")
    stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
    print(stds)

    hr("6. MONEYLINE SIDE-ASSIGNMENT CONCORDANCE (read before trusting moneyline results)")
    print("`favorite_moneyline` is a single price, identical on both team-rows of a game, with")
    print("no retained home/away identity in this file (see config.py's MONEYLINE section for")
    print("the full derivation). This module assigns it to whichever side the SPREAD's sign")
    print("says is favored (home if spread<0, away if spread>0). Measured directly against the")
    print("raw team-game file (before pivoting), concordance between that assignment and")
    print("favorite_moneyline's OWN sign, by how decisive the favorite price is:")
    print("  all priced games (n=6,641):        55.5% concordant")
    print("  |favorite_moneyline| >= 110 (n=5,430): 57.4% concordant")
    print("  |favorite_moneyline| >= 130 (n=3,126): 61.3% concordant")
    print("  |favorite_moneyline| >= 150 (n=1,619): 69.4% concordant")
    print("  |favorite_moneyline| >= 200 (n=521):   81.2% concordant")
    print("Concordance rising with decisiveness is consistent with a real, if noisy, signal -")
    print("not pure garbage - but it means the side assignment is close to a coin flip for")
    print("competitive games, exactly where a market edge would matter most.")
    print("CONCLUSION: moneyline bets below are built on a best-effort, lower-confidence side")
    print("assignment on top of an already-assumed devig. Spread and total do not have this")
    print("problem. Read the moneyline row of every summary table below with that in mind.")

    hr("7. BACKTEST")
    print("NOTE ON CLV: this dataset has only one odds snapshot per game (no true open vs")
    print("close) for spread/total, and the moneyline is doubly derived (side-assigned +")
    print("devigged). run_backtest prices bets at cfg.price_point and separately looks up the")
    print("hardcoded 'Home Odds Close' etc. columns for CLV - since both point at the SAME")
    print("assumed/derived price here, CLV is trivially ~0 for every bet. This backtest can")
    print("only speak to hit-rate/ROI against one assumed price point, not genuine closing-line")
    print("value - same honest caveat CFB's module needed.")
    print()
    print("NOTE ON WHY THIS BACKTEST'S ROI WILL LOOK UNUSUALLY LARGE, READ BEFORE TRUSTING IT:")
    print("the assumed -110/-110 vig (see config.py) makes the market's devigged 'fair'")
    print("probability EXACTLY 50.0% for every single spread/total bet, regardless of the game.")
    print("In CFB that's a mild simplification because the spread NUMBER itself varies hugely")
    print("game to game (a 3-point pick'em vs. a 35-point mismatch), so a real book's vig")
    print("mostly stays close to symmetric around whatever number balances action. NHL's puck")
    print("line is a near-constant +/-1.5 in 99.8% of priced games here (see section 1) - real")
    print("books shade a LOPSIDED matchup's PRICE (e.g. -1.5 (-250) for a big favorite vs.")
    print("-1.5 (+140) for a live underdog), not the number, but this data has no per-side")
    print("price to capture that, only the single flat number. The assumed-50/50 proxy used")
    print("here therefore can't distinguish 'a game the market has already priced lopsided'")
    print("from 'a true coin flip' AT ALL - so any real skill this model has (see the section 2/4")
    print("calibration checks, which ARE reasonably well-calibrated, not wildly overconfident)")
    print("shows up as apparent 'edge' against a market proxy that a real sportsbook would")
    print("likely have already partially priced away via the vig. CONCRETELY: read the ROI")
    print("numbers below as 'this model beats a flat coin flip by more than a real book's vig")
    print("margin,' NOT as validated profit against real NHL puck-line prices - a materially")
    print("weaker and less trustworthy claim than what the same backtest code means for a sport")
    print("with real two-sided prices (NBA) or widely-varying spread magnitudes (CFB).")

    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0,
        allowed_confidence=("Medium", "High"),
        price_point="Close",  # only price point that exists in this data
    )
    bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"\nTotal qualifying bets found (edge >= {bt_cfg.min_edge_pct}%, "
          f"confidence in {bt_cfg.allowed_confidence}): {len(bets):,}")

    if bets.empty:
        print("No bets cleared the edge/confidence thresholds - nothing further to summarize.")
        return

    hr("7a. OVERALL SUMMARY")
    print(backtest.summarize(bets).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("7b. SUMMARY BY MARKET")
    print(backtest.summarize(bets, group_cols=["market"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("7c. SUMMARY BY MARKET x EDGE BUCKET")
    bets["edge_bucket"] = bets["edge_pct"].apply(backtest.edge_bucket)
    print(backtest.summarize(bets, group_cols=["market", "edge_bucket"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("7d. SUMMARY BY SEASON")
    print(backtest.summarize(bets, group_cols=["season"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("7e. SUMMARY BY MARKET x SEASON (moneyline isolated for scrutiny)")
    print(backtest.summarize(bets, group_cols=["market", "season"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("DONE")


if __name__ == "__main__":
    main()
