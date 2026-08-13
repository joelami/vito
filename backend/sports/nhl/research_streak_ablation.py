"""
THROWAWAY research script — NOT part of the production pipeline. Prompted
directly by the app owner: "do hot streaks really matter?" `home_streak`/
`away_streak` (a signed, uncapped consecutive win/loss count, pre-game) has
been part of NHL's baseline feature set since this module was first built -
it was never itself run through core/research.py's disciplined hypothesis
test, unlike every feature ADOPTED since. This script asks the real
question the app owner's phrasing implies: does the discrete streak count
carry any predictive power BEYOND the recent-form features already sitting
next to it in ML_FEATURE_COLS (win_pct_l10, pf_l10, pa_l10, ats_pct_l10)?

This is precisely the "hot hand" question sports-analytics literature has
argued about since Gilovich/Vallone/Tversky (1985) first found the hot hand
in basketball shooting to be largely a cognitive illusion - and follow-up
work across sports has repeatedly found that once you control for a
player/team's underlying recent quality, a discrete "on a streak" flag adds
little or nothing on top. Whether that holds for THIS model, on THIS data,
is worth checking directly rather than assuming either way (assuming the
literature's finding transfers un-tested would be exactly the kind of
un-examined claim this project's discipline exists to avoid).

Tested as an ABLATION rather than an addition, the mirror image of every
other script in this project: baseline = current production ML_FEATURE_COLS
(streak included, as it always has been), variant = the SAME feature set
with home_streak/away_streak REMOVED. If fit doesn't move, streak adds
nothing beyond the recent-form features already present (supporting "hot
streaks don't matter beyond recent form" for this model) - if fit degrades
when removed, streak carries real incremental signal (supporting a genuine,
if modest, hot-hand effect this model would be worse without).

Run with:  python -m sports.nhl.research_streak_ablation   (from backend/)
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
from sports.nhl import config as nhl_config
from sports.nhl.loader import load_games
from sports.nhl.features import build_features, ML_FEATURE_COLS


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_pipeline(feats: pd.DataFrame, feature_cols: list) -> dict:
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")
    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])
    stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "n_bets": 0}
    summary = backtest.summarize(bets)
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "n_bets": int(summary["bets"].iloc[0])}


def main():
    hr("BUILDING BASE PIPELINE (loader -> power ratings -> features, current production state)")
    games = load_games()
    rating_cfg = PowerRatingConfig(
        k_factor=nhl_config.ELO_K_FACTOR, start_rating=nhl_config.ELO_START_RATING,
        home_field_adv=nhl_config.HOME_FIELD_ADV_ELO, season_regression=nhl_config.SEASON_REGRESSION,
        mov_mult_base=nhl_config.MOV_MULT_BASE, mov_mult_divisor=nhl_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_team_id", away_col="away_team_id",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", config=rating_cfg,
    )
    feats = build_features(games, rr.history)
    print(f"Feature rows: {len(feats):,}, current production ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")

    print(f"\nDescribe home_streak (signed, uncapped consecutive win/loss count, pre-game):")
    print(feats["home_streak"].describe())

    hr("BASELINE RUN (current production ML_FEATURE_COLS, streak INCLUDED)")
    baseline = run_pipeline(feats, ML_FEATURE_COLS)
    print(baseline)

    hr("VARIANT RUN (streak REMOVED - ablation)")
    feature_cols_no_streak = [c for c in ML_FEATURE_COLS if c not in ("home_streak", "away_streak")]
    print(f"Removed: home_streak, away_streak. Remaining feature count: {len(feature_cols_no_streak)}")
    variant = run_pipeline(feats, feature_cols_no_streak)
    print(variant)

    import json

    h = Hypothesis(
        name="nhl_streak_ablation_hot_hand",
        reasoning=(
            "App owner's direct question: do hot streaks really matter? home_streak/away_streak (signed, "
            "uncapped consecutive win/loss count) has been in this model's baseline feature set since it was "
            "first built, but was never itself run through core/research.py's disciplined test - unlike every "
            "feature adopted since. This is the real 'hot hand' question sports analytics has debated since "
            "Gilovich/Vallone/Tversky (1985): does a discrete streak carry predictive power BEYOND a team's "
            "recent-form features (win_pct_l10, pf_l10, pa_l10, ats_pct_l10) already sitting next to it in "
            "ML_FEATURE_COLS, or is it redundant with them? Tested as an ablation (baseline has streak, "
            "variant removes it) rather than an addition - if removing it doesn't move fit, streak adds "
            "nothing on top of recent form for this model; if fit degrades, it carries real incremental signal."
        ),
        sport="NHL",
    )
    # NOTE: baseline/variant are intentionally reversed in framing relative to
    # every other hypothesis in this project (variant = feature REMOVED, not
    # added) - evaluate_hypothesis() itself is symmetric (just measures
    # variant-minus-baseline deltas), so this is a legitimate use: a POSITIVE
    # margin/total delta here would mean the model got BETTER WITHOUT streak
    # (streak was net noise), a NEGATIVE delta means the model got WORSE
    # without it (streak carries real signal, keep it).
    result = evaluate_hypothesis(h, baseline, variant)
    hr("HYPOTHESIS RESULT (evaluate_hypothesis().to_dict())")
    print(json.dumps(result.to_dict(), indent=2))
    print("\nINTERPRETATION GUIDE (since this is an ablation, not an addition):")
    print("  - fit_degraded / recommendation='reject' => removing streak made the model WORSE => KEEP streak, real signal")
    print("  - fit_improved / recommendation='adopt' => removing streak made the model BETTER => streak was net noise, consider dropping it")
    print("  - adopt_cautiously / null => streak is redundant with recent-form features, doesn't matter either way")

    hr("DONE")


if __name__ == "__main__":
    main()
