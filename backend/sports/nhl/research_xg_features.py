"""
THROWAWAY research script -- tests the trailing expected-goals (xG)
features (sports/nhl/moneypuck_xg_features.build_trailing_xg_features)
against NHL's current production ML_FEATURE_COLS, via this project's
standard Hypothesis framework.

MOTIVATION: direct successor to the already-adopted shot-differential/
special-teams features (see decision_log.jsonl, "nhl_trailing_shot_diff_
and_pp_rate"/"nhl_trailing_pk_pct") -- modern hockey analytics treats raw
shot-attempt differential (Corsi/Fenwick) as a first-generation possession
proxy and xG (shots weighted by real historical conversion rate given
location/type/situation/danger) as its more precise descendant, since it
separates shot VOLUME from shot QUALITY. See docs/model_improvement_
candidates_2026-09-15.md's NHL section for the full external motivation --
this was flagged there as the single strongest candidate in that whole
document (data already on disk, join already solved, no new dependency).

Includes a REQUIRED split-half stability check, not just the full-sample
evaluate_hypothesis() result -- see sports/nfl/research_qb_features.py's
own docstring/decision_log.jsonl entry for exactly why a full-sample-only
"adopt" is not trustworthy on its own (a real full-sample "adopt" from that
session did not survive this exact check and was correctly rejected).

Run:  python -m sports.nhl.research_xg_features   (from backend/)
"""

import json
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nhl import config as nhl_config
from sports.nhl.loader import load_games
from sports.nhl.features import build_features, ML_FEATURE_COLS
from sports.nhl.moneypuck_xg_features import build_trailing_xg_features, TRAILING_COLS


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def run_pipeline(feats, feature_cols) -> dict:
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg, sport="NHL")
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    summary = backtest.summarize(bets)
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "bets": int(summary["bets"].iloc[0])}


def build_all_feats():
    games = load_games()
    rr = compute_power_ratings(
        games, home_col="home_team_id", away_col="away_team_id",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        config=PowerRatingConfig(
            k_factor=nhl_config.ELO_K_FACTOR, start_rating=nhl_config.ELO_START_RATING,
            home_field_adv=nhl_config.HOME_FIELD_ADV_ELO, season_regression=nhl_config.SEASON_REGRESSION,
            mov_mult_base=nhl_config.MOV_MULT_BASE, mov_mult_divisor=nhl_config.MOV_MULT_DIVISOR,
        ),
    )
    feats = build_features(games, rr.history)
    feats = build_trailing_xg_features(feats)  # adds home_/away_xg_for_l10/xg_against_l10 columns
    return feats


def main():
    hr("BUILDING FEATURES (current production ML_FEATURE_COLS + trailing xG columns, both present)")
    feats = build_all_feats()
    xg_cols = [f"{side}_{c}_l10" for side in ("home", "away") for c in TRAILING_COLS]
    from sports.nhl.moneypuck_xg_features import load_team_game_xg
    real_xg_games = load_team_game_xg(load_games())["game_id"].nunique()
    print(f"Feature rows: {len(feats):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}, xG cols: {len(xg_cols)}")
    print(f"Real (non-fallback) MoneyPuck xG coverage: {real_xg_games:,}/{len(feats):,} games "
          f"({real_xg_games / len(feats) * 100:.1f}%) -- the rest fall back to the league average, "
          f"reported here for honesty, not used to filter anything.")

    hr("BASELINE RUN (current production ML_FEATURE_COLS -- no xG)")
    baseline = run_pipeline(feats, ML_FEATURE_COLS)
    print(baseline)

    hr("HYPOTHESIS: adding trailing xG-for/xG-against (home/away)")
    variant_cols = ML_FEATURE_COLS + xg_cols
    variant = run_pipeline(feats, variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="nhl_moneypuck_trailing_xg_features",
        reasoning=(
            "xG is the direct successor to the Corsi/Fenwick shot-differential family this project "
            "already adopted (nhl_trailing_shot_diff_and_pp_rate, real margin_corr gain) -- modern hockey "
            "analytics (MoneyPuck, Evolving-Hockey, Natural Stat Trick) treats raw shot attempts as a "
            "first-generation possession proxy and xG (shot attempts weighted by real historical "
            "conversion rate given location/type/situation) as the more precise descendant, specifically "
            "because it separates shot VOLUME from shot QUALITY -- two teams with identical shot "
            "differentials can have very different real scoring-chance quality. The exact data needed "
            "(OnIce_F_xGoals/OnIce_A_xGoals, per player per game, situation-tagged) is already downloaded "
            "to Datasets/NHL/2008_to_2024 copy 2.csv / 2025 copy.csv, and the game-id join (MoneyPuck's "
            "gameId -> this project's game_id, via home/away team + date with a 1-day merge_asof "
            "tolerance) is already built and verified in sports/nhl/research_player_matchup.py. Tested as "
            "trailing (last-10-game, walk-forward-safe) team-level additions, same shape as every other "
            "trailing feature here."
        ),
        sport="NHL",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT (full sample)")
    print(json.dumps(result.to_dict(), indent=2))

    hr("SPLIT-HALF STABILITY CHECK -- required before trusting the full-sample result above")
    seasons = sorted(feats["season"].unique())
    mid = seasons[len(seasons) // 2]
    stable = True
    for label, sub in [("first half", feats[feats["season"] < mid]),
                        ("second half", feats[feats["season"] >= mid])]:
        b = run_pipeline(sub, ML_FEATURE_COLS)
        v = run_pipeline(sub, variant_cols)
        md, td = v["margin_corr"] - b["margin_corr"], v["total_corr"] - b["total_corr"]
        print(f"{label}: margin {b['margin_corr']:.4f}->{v['margin_corr']:.4f} ({md:+.4f}), "
              f"total {b['total_corr']:.4f}->{v['total_corr']:.4f} ({td:+.4f})")
        if md < -0.005 or td < -0.005:
            stable = False
    print(f"\n{'STABLE' if stable else 'NOT STABLE'} across both halves (neither corr degraded past the noise floor "
          f"in either half) -- {'trust' if stable else 'do NOT trust'} the full-sample recommendation above.")
    hr("DONE")


if __name__ == "__main__":
    main()
