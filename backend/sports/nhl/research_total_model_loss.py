"""
THROWAWAY research script -- tests switching NHL's TOTAL model (only) from
this project's project-wide default loss (MAE/"absolute_error") to
squared-error, via this project's standard Hypothesis framework.

MOTIVATION: direct answer to the app owner's "take a look at this and make
any necessary changes to make the algo better" (2026-11-07). Found while
investigating a live confidence-audit flag (NHL total confidence_tier()
proven backwards with real data, see decision_log.jsonl) that NHL's total
MODEL itself carries a real, substantial bias: mean predicted_total 5.29
vs mean actual_total 5.80 (bias -0.49 goals/game) under the project-wide
default MAE loss. MAE targets the conditional MEDIAN; NHL's total
distribution is right-skewed (skew ~0.38) the same direction and
comparable magnitude to NFL/CFB's own total skew, so this isn't obviously
an NHL-only phenomenon -- it had to be checked per-sport, not assumed.

NFL was checked side-by-side (see core/ml_models.py's _make_model()
docstring for the exact numbers) and showed the OPPOSITE result:
squared-error REDUCED NFL's total correlation despite also shrinking its
own smaller bias -- confirming MAE was and remains the right choice for
NFL specifically, not evidence this is a universal fix. NHL's own margin
model (skew ~-0.03, already symmetric) showed no comparable bias problem,
so this script only tests the TOTAL market.

Includes a REQUIRED split-half stability check, not just the full-sample
evaluate_hypothesis() result -- same discipline as every other feature
test in this project (see sports/nfl/research_qb_features.py's own
docstring for exactly why a full-sample-only "adopt" is not trustworthy
on its own).

Run:  python -m sports.nhl.research_total_model_loss   (from backend/)
"""

import json
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest, live_results
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nhl import config as nhl_config
from sports.nhl.loader import load_games
from sports.nhl.features import build_features, ML_FEATURE_COLS
from sports.nhl.moneypuck_xg_features import build_trailing_xg_features


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def run_pipeline(feats, feature_cols, total_loss) -> dict:
    wf = walk_forward_predict(feats, feature_cols, total_loss=total_loss)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])
    total_bias = float((oos["predicted_total"] - oos["actual_total"]).mean())

    stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, price_point="Close")
    bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg, sport="NHL")
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr, "total_bias": total_bias,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    summary = backtest.summarize(bets)
    return {"margin_corr": margin_corr, "total_corr": total_corr, "total_bias": total_bias,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "bets": int(summary["bets"].iloc[0])}


def main():
    hr("BUILDING NHL FEATURES (production ML_FEATURE_COLS incl. trailing xG)")
    games = load_games()
    games_for_ratings = live_results.with_live_results(games, "NHL")
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_team_id", away_col="away_team_id",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        neutral_col="is_neutral_venue" if "is_neutral_venue" in games.columns else None,
        config=PowerRatingConfig(
            k_factor=nhl_config.ELO_K_FACTOR, start_rating=nhl_config.ELO_START_RATING,
            home_field_adv=nhl_config.HOME_FIELD_ADV_ELO, season_regression=nhl_config.SEASON_REGRESSION,
            mov_mult_base=nhl_config.MOV_MULT_BASE, mov_mult_divisor=nhl_config.MOV_MULT_DIVISOR,
        ),
    )
    feats = build_features(games, rr.history)
    feats = build_trailing_xg_features(feats)
    print(f"Feature rows: {len(feats):,}")
    print(f"actual_total skew: {feats['actual_total'].skew():.4f}, "
          f"mean={feats['actual_total'].mean():.3f}, median={feats['actual_total'].median():.1f}")

    hr("BASELINE RUN (current production default: MAE/absolute_error total loss)")
    baseline = run_pipeline(feats, ML_FEATURE_COLS, total_loss="absolute_error")
    print(baseline)

    hr("HYPOTHESIS: switch TOTAL model (only) to squared-error loss")
    variant = run_pipeline(feats, ML_FEATURE_COLS, total_loss="squared_error")
    print(variant)

    hyp = Hypothesis(
        name="nhl_total_model_squared_error_loss",
        reasoning=(
            "NHL's total model carries a real, substantial bias under this project's project-wide default "
            "MAE loss: mean predicted_total meaningfully below mean actual_total, consistent with MAE "
            "targeting the conditional median of a right-skewed target (actual_total skew ~0.38) rather "
            "than its mean. Squared-error targets the conditional mean directly, which should reduce this "
            "specific bias -- tested on the TOTAL market only (NHL's margin target is already symmetric, "
            "no comparable bias to fix); NFL's own total model was checked side-by-side with the identical "
            "method and showed the opposite result, confirming this is a genuine per-sport difference, not "
            "a case for a project-wide default change."
        ),
        sport="NHL",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT (full sample)")
    print(f"total bias: {baseline['total_bias']:+.4f} -> {variant['total_bias']:+.4f}")
    print(json.dumps(result.to_dict(), indent=2))

    hr("SPLIT-HALF STABILITY CHECK -- required before trusting the full-sample result above")
    seasons = sorted(feats["season"].unique())
    mid = seasons[len(seasons) // 2]
    stable = True
    for label, sub in [("first half", feats[feats["season"] < mid]),
                        ("second half", feats[feats["season"] >= mid])]:
        b = run_pipeline(sub, ML_FEATURE_COLS, total_loss="absolute_error")
        v = run_pipeline(sub, ML_FEATURE_COLS, total_loss="squared_error")
        td = v["total_corr"] - b["total_corr"]
        print(f"{label}: total_corr {b['total_corr']:.4f}->{v['total_corr']:.4f} ({td:+.4f}), "
              f"bias {b['total_bias']:+.4f}->{v['total_bias']:+.4f}")
        if td < -0.005:
            stable = False
    print(f"\n{'STABLE' if stable else 'NOT STABLE'} across both halves -- "
          f"{'trust' if stable else 'do NOT trust'} the full-sample recommendation above.")
    hr("DONE")


if __name__ == "__main__":
    main()
