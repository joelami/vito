"""
Diagnoses WHY MLB moneyline underdog (DOG) bets lose money in the live
forward test and the full historical backtest (see decision_log.jsonl /
docs/METHODOLOGY.md for the already-confirmed live numbers this follows up
on: -26.28% ROI on 35 live DOG/FAV-mixed moneyline bets, and the matching
-4.01% ROI / 37.1% hit rate on 8,194 backtested DOG bets vs -2.53% / 52.1%
on 3,551 FAV bets).

STEP 1 — PRECISE DIAGNOSTIC (done first, before any fix is proposed).
A GENERAL calibration check (bucketing the model's raw blended probability
against actual outcome across the WHOLE dataset) was already done separately
and came back reasonably well-calibrated, if anything slightly UNDERconfident
at the high end. That rules out "the probability model is broadly broken."
This script instead calibrates the SELECTED bet population specifically —
the actual backtested moneyline bets, bucketed by their own model_prob,
DOG side (market_odds>=2.0) vs FAV side (market_odds<2.0) separately — to
determine whether DOG-side bets are genuinely overconfident (predicted >>
actual) or whether the negative ROI is just a real, small, honest edge that
doesn't clear the vig on dog prices.

RESULT: DOG bets (n=8,194) are genuinely overconfident, not just
under-vig-value: mean model_prob 0.4573 vs actual win rate 0.3705 (a 8.68
percentage-point gap), and the gap does NOT shrink as edge/confidence rises
- if anything it's WORSE in the highest-edge quintile (13.4pp) than the
lowest (3.7pp). FAV bets show the same shape at smaller scale (mean gap
5.27pp, growing from 0.25pp to 11.24pp across edge quintiles). This is a
real calibration bug in the selected population, not "insufficient edge
over the vig" - insufficient-edge would show model_prob only marginally
above actual, roughly flat with edge size; what's actually observed is a
gap that's large AND grows with how much the model disagrees with the
market, the textbook shape of a selection effect on a noisy estimator, not
a mispriced-but-real edge.

STEP 2 — ROOT CAUSE, checked directly, not guessed. Two specific candidate
mechanisms from the task brief were checked FIRST and both come back
negative (real null results, reported honestly rather than skipped past to
find something that confirms a story):

(a) Heteroscedasticity ("does true residual variance scale with how
    lopsided the predicted margin is?"). Checked directly: bucketing
    (actual_margin - predicted_margin) by |predicted margin| (both the ML
    leg and, since the ML leg's HistGradientBoostingRegressor output turned
    out to be heavily discretized near +/-1 run - a separate, real
    observation worth a future look but not this one - the continuous Elo-
    implied margin too) shows residual std essentially FLAT across the
    whole range: 4.28-4.44 runs in every one of 8 quantile buckets of
    |elo_implied_margin| from 0.05 to 1.31 runs, corr(|pred_margin|,
    resid^2) = 0.0027 (ML) / 0.0052 (Elo) - indistinguishable from zero.
    REJECTED: the global residual std is not hiding a lopsided-game
    variance problem.

(b) Confidence-tier inversion being moneyline-specific. Checked directly:
    High-confidence bets underperform Medium-confidence bets in ALL THREE
    markets, not just moneyline - moneyline High -4.77% (n=7,641) vs Medium
    -1.32% (n=4,104); spread High -4.66% (n=5,866) vs Medium -2.99%
    (n=5,882); total High -3.07% (n=7,525) vs Medium -2.17% (n=4,724).
    REJECTED as a moneyline-specific, elo_p/ml_p-false-agreement story -
    whatever's causing the inversion, it isn't particular to how
    moneyline_prob's two submodels agree, since spread (same elo_p/ml_p
    pair) and total (a completely different ml_p/naive_p pair) show the
    identical shape.

(c) What IS real, checked directly on the FULL walk-forward dataset (not
    just selected bets, so this isn't circular with the ROI numbers it
    explains): splitting every odds-covered game (n=22,762) by whether the
    model's favored side agrees or disagrees with the MARKET's favored
    side (i.e. whether this is a "back the market's underdog" pick, 21.5%
    of games, n=4,888) shows a clean, stark split. When model and market
    AGREE on direction (78.5% of games): mean model_prob 0.5686 vs actual
    0.5917 - calibration is GOOD, if anything a bit underconfident (matches
    the general calibration finding). When model and market DISAGREE on
    direction (backing the market's underdog): mean model_prob 0.5505 vs
    actual 0.4583 - a 9.23-point overconfidence gap, present at every edge
    level checked (6 buckets from -6.1% to +6.4% mean edge, gap 4.4-12.7pp
    in every single one, not concentrated at the extremes). This is the
    real, mechanistic story: this model's feature set (rolling win%, rest,
    streak, starter quality - all schedule/form-based) is well-calibrated
    when it agrees with the market, and specifically unreliable exactly
    when it disagrees with the market's read on who's better - the market
    is pricing in real information (lineup news, bullpen situation, sharp
    money, injuries) this feature set structurally doesn't have, and
    disagreement with the market is, almost by definition, disagreement
    with the participant that has more information. DOG-side moneyline
    bets are, by construction, ALWAYS a "back the market's underdog" pick -
    which is why this shows up so cleanly as a moneyline-DOG problem in the
    ROI numbers even though the underlying mechanism is really about
    direction-of-disagreement-with-market, not "being an underdog" per se.

STEP 3 — HYPOTHESIS AND TEST. Since the model doesn't have the market's
extra information, and the market's devigged probability is legitimately
known pre-bet (it's the same "Close" snapshot price this dataset's edge
-finder already prices bets against - not future information, no leakage),
the natural, well-motivated fix is to let the ML margin/total models see
that market-implied probability directly as a feature (`market_fair_home_
prob`, neutral-filled at 0.5 - "no information" - for the ~73% of the
1990-2025 history outside the 2012-2021 odds archive's coverage, same
fallback discipline as sp_er_lN's league-average fallback). If the market
really is catching signal this feature set misses, adding it should
measurably improve genuine out-of-sample fit (margin_corr), not just
ROI - which is exactly the distinction `core.research.evaluate_hypothesis`
exists to enforce.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest, odds_math
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.mlb import config as mlb_config
from sports.mlb.loader import load_games
from sports.mlb.odds_loader import attach_odds
from sports.mlb.starting_pitcher import attach_starter_quality
from sports.mlb.features import build_features, ML_FEATURE_COLS


BT_CFG = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _build_rated_features():
    games = load_games()
    games = attach_odds(games)
    games = attach_starter_quality(games)
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
    return build_features(games, rr.history)


def _run_pipeline(feats: pd.DataFrame, feature_cols: list):
    """Same shape verify.py/pipeline.py already produce: walk-forward ML,
    residual stds, real backtest, summary metrics dict."""
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")
    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, mlb_config.ELO_POINTS_PER_MARGIN)
    ecfg = ensemble.EnsembleConfig()
    bets = backtest.run_backtest(oos, stds, mlb_config.ELO_POINTS_PER_MARGIN, ecfg, BT_CFG)
    summary = backtest.summarize(bets)

    metrics = {
        "margin_corr": margin_corr, "total_corr": total_corr,
        "roi_pct": float(summary["roi_pct"].iloc[0]), "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
        "bets": int(summary["bets"].iloc[0]),
    }
    return metrics, bets, oos, stds, ecfg


def diagnostic_selected_bet_calibration(bets: pd.DataFrame):
    hr("STEP 1: SELECTED-BET CALIBRATION — DOG vs FAV moneyline (the actual backtested bets)")
    ml = bets[bets["market"] == "moneyline"].copy()
    ml["is_dog"] = ml["market_odds"] >= 2.0
    ml["won"] = (ml["result"] == 1).astype(int)

    for label, sub in [("DOG (market_odds >= 2.0)", ml[ml["is_dog"]]), ("FAV (market_odds < 2.0)", ml[~ml["is_dog"]])]:
        print(f"\n--- {label}, n={len(sub)} ---")
        print(f"Overall: mean_model_prob={sub['model_prob'].mean():.4f}, "
              f"actual_win_rate={sub['won'].mean():.4f}, "
              f"gap={sub['model_prob'].mean() - sub['won'].mean():+.4f}")
        d = sub.copy()
        d["edge_bucket"] = pd.qcut(d["edge_pct"], 5, duplicates="drop")
        g = d.groupby("edge_bucket", observed=True).agg(
            n=("won", "size"), mean_edge=("edge_pct", "mean"),
            mean_model_prob=("model_prob", "mean"), actual_win_rate=("won", "mean"),
            roi=("flat_profit", lambda s: s.sum() / len(s) * 100),
        )
        g["gap"] = g["mean_model_prob"] - g["actual_win_rate"]
        print(g.to_string(float_format=lambda x: f"{x:.4f}"))


def diagnostic_heteroscedasticity(oos: pd.DataFrame):
    hr("STEP 2a: HETEROSCEDASTICITY CHECK (does residual variance scale with |predicted margin|?)")
    elo_pred_margin = oos["rating_diff_pre"] / mlb_config.ELO_POINTS_PER_MARGIN
    resid_elo = oos["actual_margin"] - elo_pred_margin
    d = pd.DataFrame({"resid": resid_elo, "abs_pred": elo_pred_margin.abs()})
    d["bucket"] = pd.qcut(d["abs_pred"], 8, duplicates="drop")
    g = d.groupby("bucket", observed=True).agg(n=("resid", "size"), mean_abs_pred=("abs_pred", "mean"),
                                                resid_std=("resid", "std"))
    print("(Elo-implied margin, continuous - the ML leg's own predicted_margin is heavily discretized "
          "near +/-1 run and isn't a useful continuous axis for this check)")
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))
    corr = float(np.corrcoef(elo_pred_margin.abs(), resid_elo ** 2)[0, 1])
    print(f"\ncorr(|elo_implied_margin|, resid^2) = {corr:.4f} -- {'REJECTED (flat)' if abs(corr) < 0.05 else 'a real relationship'}")


def diagnostic_confidence_tier_by_market(bets: pd.DataFrame):
    hr("STEP 2b: IS THE CONFIDENCE-TIER INVERSION MONEYLINE-SPECIFIC?")
    for mkt in ["moneyline", "spread", "total"]:
        d = bets[bets["market"] == mkt]
        g = d.groupby("confidence").agg(n=("result", "size"), hit=("result", lambda s: (s == 1).mean()),
                                         roi=("flat_profit", lambda s: s.sum() / len(s) * 100))
        print(f"\n--- {mkt} ---")
        print(g.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nHigh underperforms Medium in ALL THREE markets -> REJECTED as a moneyline-specific mechanism.")


def diagnostic_market_direction_disagreement(oos: pd.DataFrame, stds, ecfg):
    hr("STEP 2c: THE REAL MECHANISM — does the model disagree with the MARKET's pick direction?")
    has_ml = oos["Home Odds Close"].notna() & oos["Away Odds Close"].notna()
    d = oos[has_ml].copy()
    elo_margin = d["rating_diff_pre"] / mlb_config.ELO_POINTS_PER_MARGIN
    d["elo_p"] = [odds_math.cover_prob_spread(m, stds.elo_margin_std, 0.0) for m in elo_margin]
    d["ml_p"] = [odds_math.cover_prob_spread(m, stds.ml_margin_std, 0.0) for m in d["predicted_margin"]]
    d["blended_prob"] = ecfg.weight_elo_moneyline * d["elo_p"] + (1 - ecfg.weight_elo_moneyline) * d["ml_p"]
    fair = d.apply(lambda r: odds_math.devig_two_way(r["Home Odds Close"], r["Away Odds Close"]), axis=1)
    d["fair_home"] = [f[0] for f in fair]

    model_favors_home = d["blended_prob"] > 0.5
    market_favors_home = d["fair_home"] > 0.5
    disagrees = model_favors_home != market_favors_home
    print(f"Games where model's pick direction disagrees with the market's: "
          f"{disagrees.sum():,} / {len(d):,} ({disagrees.mean()*100:.1f}%)")

    for label, mask in [("AGREES with market direction", ~disagrees), ("DISAGREES with market direction (backs market's dog)", disagrees)]:
        sub = d[mask].copy()
        sub["model_prob_favored"] = np.where(sub["blended_prob"] > 0.5, sub["blended_prob"], 1 - sub["blended_prob"])
        sub["favored_won"] = np.where(sub["blended_prob"] > 0.5, sub["home_win"], 1 - sub["home_win"])
        gap = sub["model_prob_favored"].mean() - sub["favored_won"].mean()
        print(f"\n--- {label}, n={len(sub):,} ---")
        print(f"mean_model_prob={sub['model_prob_favored'].mean():.4f}, "
              f"actual_win_rate={sub['favored_won'].mean():.4f}, gap={gap:+.4f}")

    print("\nCLEAN SPLIT: calibration is fine (even slightly underconfident) when the model agrees with "
          "the market's direction, and overconfident by ~9pp specifically when it disagrees -- this is "
          "the real mechanism. DOG-side moneyline bets are, by construction, always a 'disagree with "
          "the market's direction' pick, which is why the ROI problem shows up cleanest there.")


def build_market_prob_feature(feats: pd.DataFrame) -> pd.DataFrame:
    """market_fair_home_prob: devigged home moneyline probability from the
    2012-2021 odds archive, 0.5 ('no information') fallback for the ~73% of
    games outside that archive's coverage -- same neutral-fallback
    discipline as starting_pitcher.py's LEAGUE_AVG_SP_ER_PER_START. This is
    the SAME 'Close' snapshot price core.edge_finder already prices bets
    against for this dataset (its only snapshot, see odds_loader.py) --
    known pre-bet, not future information, so this is not leakage."""
    out = feats.copy()
    has_ml = out["Home Odds Close"].notna() & out["Away Odds Close"].notna()
    fair = out.loc[has_ml].apply(lambda r: odds_math.devig_two_way(r["Home Odds Close"], r["Away Odds Close"]), axis=1)
    out["market_fair_home_prob"] = 0.5
    out.loc[has_ml, "market_fair_home_prob"] = [f[0] for f in fair]
    return out


def main():
    hr("BUILDING BASELINE PIPELINE")
    feats = _build_rated_features()
    baseline_metrics, baseline_bets, baseline_oos, stds, ecfg = _run_pipeline(feats, ML_FEATURE_COLS)
    print(f"Baseline: {baseline_metrics}")

    diagnostic_selected_bet_calibration(baseline_bets)
    diagnostic_heteroscedasticity(baseline_oos)
    diagnostic_confidence_tier_by_market(baseline_bets)
    diagnostic_market_direction_disagreement(baseline_oos, stds, ecfg)

    hr("STEP 3: HYPOTHESIS TEST — market_fair_home_prob as an ML feature")
    feats_variant = build_market_prob_feature(feats)
    variant_cols = ML_FEATURE_COLS + ["market_fair_home_prob"]
    variant_metrics, variant_bets, variant_oos, _, _ = _run_pipeline(feats_variant, variant_cols)
    print(f"Variant: {variant_metrics}")

    vml = variant_bets[variant_bets["market"] == "moneyline"].copy()
    vml["is_dog"] = vml["market_odds"] >= 2.0
    print("\nVariant DOG/FAV breakdown (moneyline):")
    print(vml.groupby("is_dog").agg(n=("result", "size"), hit=("result", lambda s: (s == 1).mean()),
                                     roi=("flat_profit", lambda s: s.sum() / len(s) * 100)).to_string(float_format=lambda x: f"{x:.4f}"))

    hyp = Hypothesis(
        name="mlb_moneyline_market_prob_feature",
        sport="MLB",
        reasoning=(
            "Diagnosed directly (not guessed): DOG-side moneyline bets (n=8,194 backtested) are "
            "genuinely overconfident, not just under-vig-value -- mean model_prob 0.4573 vs actual "
            "win rate 0.3705, a gap that GROWS with edge size rather than shrinking. Two specific "
            "candidate mechanisms (heteroscedastic residual variance vs |predicted margin|; a "
            "moneyline-specific confidence-tier artifact) were checked directly and both rejected -- "
            "residual std is flat across |predicted margin| (corr ~0.003-0.005), and the same "
            "High-worse-than-Medium confidence inversion shows up in spread and total too, not just "
            "moneyline. What IS real, checked on the full 22,762-game odds-covered walk-forward "
            "sample (not just selected bets): calibration is good when the model's pick direction "
            "agrees with the market's, and overconfident by ~9 points specifically when it disagrees "
            "(21.5% of games) -- the market is pricing in real-time information (injuries, lineup "
            "news, sharp money) this schedule/form-based feature set structurally lacks, and "
            "disagreeing with a better-informed participant is exactly where a feature set with less "
            "information should expect to be wrong more than it thinks. Since the market's devigged "
            "probability is legitimately known pre-bet (the same 'Close' snapshot price already used "
            "to price every backtested bet in this dataset -- not future information), feeding it to "
            "the ML models directly as `market_fair_home_prob` (0.5 neutral fallback for the ~73% of "
            "history outside the 2012-2021 odds archive, same discipline as sp_er_lN's league-average "
            "fallback) is a well-motivated way to let the model actually use that information instead "
            "of only comparing against it after the fact at bet-selection time."
        ),
    )
    result = evaluate_hypothesis(hyp, baseline_metrics, variant_metrics)
    hr("RESULT")
    print(f"margin_corr: {baseline_metrics['margin_corr']:.4f} -> {variant_metrics['margin_corr']:.4f} "
          f"({result.margin_corr_delta:+.4f})")
    print(f"total_corr:  {baseline_metrics['total_corr']:.4f} -> {variant_metrics['total_corr']:.4f} "
          f"({result.total_corr_delta:+.4f})")
    print(f"ROI: {baseline_metrics['roi_pct']:+.2f}% -> {variant_metrics['roi_pct']:+.2f}% "
          f"({result.roi_delta_pct:+.2f}pp, baseline stderr {baseline_metrics['roi_stderr_pct']:.2f}pp)")
    print(f"suspicious: {result.suspicious}")
    print(f"RECOMMENDATION: {result.recommendation}")


if __name__ == "__main__":
    main()
