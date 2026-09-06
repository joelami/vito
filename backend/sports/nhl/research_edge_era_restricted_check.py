"""
THROWAWAY follow-up to research_edge_goalie_high_danger_save_pct.py and
research_edge_star_skating_speed.py -- NOT part of the production pipeline.

MOTIVATION (app owner pushback, 2026-09-05, and a fair one): both scripts'
"adopt_cautiously"/flat verdicts were measured across the FULL 2004-2026
walk-forward backtest, where the Edge-derived feature only carries a real,
non-fallback value for ~17-24% of rows (every season whose PRIOR season is
Edge-covered, i.e. 2023-2026) -- the other ~76-83% of the backtest
population sees the feature collapse to a constant league-average fallback,
which is a genuine no-op for that row. Averaging ROI/correlation deltas
across a population that's 3-4x more "no-op" rows than real ones dilutes
any real signal the feature might carry FAR more than the era of hockey
itself does -- the app owner's point that "5 seasons should be enough,
especially given how much the game has changed in 20 years" is really an
argument that a recent-only sample is a perfectly legitimate one for a
live-forward model, not a defect. Restricting the FINAL comparison to only
the rows where the feature is real (not the whole 22-season backtest) is
the honest way to test that.

WHAT THIS DOES: reuses the exact same feature-building + walk-forward
pipeline as the two parent scripts unchanged (same training data, same
expanding-window walk-forward -- the model still needs the long run-up),
but filters the OUT-OF-SAMPLE COMPARISON population down to season>=2023
(the seasons where home_hd_save_pct / home_star_speed_pctile can be real,
not fallback) before computing correlation/backtest deltas. This is a
strictly cleaner test of "does this feature help in the era it actually
exists," not a different feature or a different model.
"""

import json

import numpy as np

from core import ensemble, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nhl import config as nhl_config
from sports.nhl.features import build_features, ML_FEATURE_COLS
from sports.nhl.loader import load_games


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def run_pipeline_restricted(feats, feature_cols, min_season: int) -> dict:
    """Same as the parent scripts' run_pipeline(), except the comparison
    population (correlation + backtest) is filtered to season>=min_season
    AFTER walk-forward prediction -- training still uses the full history,
    only the evaluation window is restricted."""
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")
    oos = oos[oos["season"] >= min_season]
    if oos.empty:
        return {"margin_corr": float("nan"), "total_corr": float("nan"),
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0, "rows": 0}

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    total_bets = bets[bets["market"] == "total"] if not bets.empty else bets
    if total_bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0, "rows": len(oos)}
    summary = backtest.summarize(total_bets)
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "bets": int(summary["bets"].iloc[0]), "rows": len(oos)}


def main():
    hr("BUILDING BASE PIPELINE (unchanged, full history for training/rating run-up)")
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
    feats_base = build_features(games, rr.history)

    from sports.nhl.research_starting_goalie_save_pct import add_starter_save_pct_feature
    feats_with_recency = add_starter_save_pct_feature(feats_base, games)
    baseline_cols = ML_FEATURE_COLS + ["home_starter_save_pct_l10", "away_starter_save_pct_l10"]

    MIN_SEASON = 2023  # first season whose PRIOR season (2022) is Edge-covered

    hr(f"HIGH-DANGER SAVE% -- restricted to season>={MIN_SEASON} (Edge-real rows only)")
    from sports.nhl.research_edge_goalie_high_danger_save_pct import build_hd_save_pct_feature
    feats_hd, coverage_hd = build_hd_save_pct_feature(feats_with_recency, games)
    baseline_hd = run_pipeline_restricted(feats_hd, baseline_cols, MIN_SEASON)
    variant_hd = run_pipeline_restricted(feats_hd, baseline_cols + ["home_hd_save_pct", "away_hd_save_pct"], MIN_SEASON)
    print("baseline:", baseline_hd)
    print("variant :", variant_hd)
    hyp_hd = Hypothesis(
        name="nhl_edge_goalie_hd_save_pct_ERA_RESTRICTED",
        reasoning=(f"Same feature as research_edge_goalie_high_danger_save_pct.py, but the "
                   f"comparison population is restricted to season>={MIN_SEASON} (where the "
                   f"feature is real, not fallback) instead of the full 2004-2026 backtest -- "
                   f"testing whether the flat full-history result was signal dilution from "
                   f"~76-83% no-op rows, not evidence the feature itself is inert."),
        sport="NHL",
    )
    result_hd = evaluate_hypothesis(hyp_hd, baseline_hd, variant_hd)
    print(json.dumps(result_hd.to_dict(), indent=2))

    hr(f"STAR SKATING SPEED -- restricted to season>={MIN_SEASON} (Edge-real rows only)")
    from sports.nhl.research_edge_star_skating_speed import build_star_speed_feature
    from sports.nhl.research_player_matchup import load_player_games, identify_stars
    mp_games = load_player_games()
    stars = identify_stars(mp_games)
    feats_speed, coverage_speed = build_star_speed_feature(feats_base, games, stars)
    baseline_speed = run_pipeline_restricted(feats_speed, ML_FEATURE_COLS, MIN_SEASON)
    variant_speed = run_pipeline_restricted(feats_speed, ML_FEATURE_COLS + ["home_star_speed_pctile", "away_star_speed_pctile"], MIN_SEASON)
    print("baseline:", baseline_speed)
    print("variant :", variant_speed)
    hyp_speed = Hypothesis(
        name="nhl_edge_star_speed_ERA_RESTRICTED",
        reasoning=(f"Same feature as research_edge_star_skating_speed.py, restricted to "
                   f"season>={MIN_SEASON} for the same reason as the high-danger save% "
                   f"restricted check above."),
        sport="NHL",
    )
    result_speed = evaluate_hypothesis(hyp_speed, baseline_speed, variant_speed)
    print(json.dumps(result_speed.to_dict(), indent=2))

    hr("DONE")


if __name__ == "__main__":
    main()
