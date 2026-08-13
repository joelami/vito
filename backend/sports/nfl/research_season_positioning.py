"""
THROWAWAY research script — NOT wired into main.py/pipeline.py/harness.py.
Tests one hypothesis via core/research.py's disciplined evaluate_hypothesis()
loop, following the exact shape of sports/cfb/research_week_trends.py (the
season_week_adj precedent) and this sport's own prior research scripts
(research_scoring_trend.py, research_over_tendency.py): loader -> power
ratings -> features (baseline vs variant) -> walk-forward ML -> residual
stds -> backtest -> evaluate_hypothesis().

Also runs a quick, free (non-hypothesis-test) diagnostic first: permutation
feature importance on the CURRENT production feature set, to directly answer
the task's question "is is_divisional actually being used effectively" with
real evidence rather than a guess, before deciding whether a divisional
refinement is even worth formally testing.

pipeline.py read ONLY as reference (weather attach, config wiring); not
imported or modified.

Run with:  python -m sports.nfl.research_season_positioning   (from backend/)
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
from core.ml_models import walk_forward_predict, compute_feature_importance
from core.research import Hypothesis, evaluate_hypothesis
from sports.nfl import config as nfl_config
from sports.nfl.loader import load_games
from sports.nfl import features as nfl_features
from sports.nfl.features import ML_FEATURE_COLS, naive_score_features, pythagorean_win_pct
from sports.nfl.weather import attach_weather

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Variant feature builder: same as features.build_features, but adds two new
# per-team binary flags, home_playing_out_string / away_playing_out_string.
#
# Rationale: NFL betting literature well documents that teams with nothing
# left to play for late in the season — either mathematically/practically
# eliminated, or already locked into their playoff seed — play with less
# urgency (rest starters, less aggressive game-planning) than teams still
# fighting for a spot. This is distinct from anything currently in
# ML_FEATURE_COLS: is_playoff flags playoff GAMES themselves, but nothing
# captures regular-season "stakes" as the season winds down.
#
# This is deliberately a PROXY, not exact playoff math. Real elimination
# requires full divisional/conference standings, tiebreakers, and remaining
# schedule — out of scope and easy to get subtly wrong. Instead: each team's
# own within-season win percentage (cumulative, walk-forward-safe: current
# game always excluded) by a given point in the season correlates strongly
# with contention status without needing to simulate the whole standings
# picture. A team at .200 in week 15 is a plausible "tanking" candidate; a
# team at .850 in week 15 has likely clinched or is playing for seeding only.
#
# "Week of season" has no real column in this dataset (loader.py confirmed:
# the raw sheet has Date/Team/Score/odds columns only, nothing week-shaped) —
# approximated from the calendar instead: days since that season's first
# game, divided by 7. This tracks the real week number closely since NFL
# games run on a strict weekly cadence, and is a calendar-lateness signal
# either way, which is what the hypothesis actually needs.
# ---------------------------------------------------------------------------
LATE_SEASON_WEEK_THRESHOLD = 15   # regular-season stretch run (research_week_trends.py's
                                   # CFB precedent used a comparable "late phase" cut)
TANKING_WIN_PCT = 0.25            # roughly a 4-13 team through 17 games — plausible non-contender
CLINCHED_WIN_PCT = 0.85           # roughly 14-3 or better — likely locked into seeding


def build_features_with_positioning(games: pd.DataFrame, rating_history: pd.DataFrame) -> pd.DataFrame:
    log = nfl_features._team_game_log(games)
    log = nfl_features._rolling_form(log)  # adds ats_pct_l10, win_pct_l10, pf_l10, pa_l10, rest_days, streak

    grp_season = log.groupby(["team", "season"], group_keys=False)
    # cumulative win pct WITHIN this season, current game excluded (shift(1) before expanding)
    log["season_win_pct_to_date"] = grp_season["won"].apply(lambda s: s.shift(1).expanding().mean())
    # first game of a team's season has no prior record this season -> neutral 0.5, same
    # convention as every other "no history yet" fill in features.py
    log["season_win_pct_to_date"] = log["season_win_pct_to_date"].fillna(0.5)

    season_start = log.groupby("season")["date"].transform("min")
    log["week_of_season"] = ((log["date"] - season_start).dt.days // 7) + 1

    late = log["week_of_season"] >= LATE_SEASON_WEEK_THRESHOLD
    no_stakes = (log["season_win_pct_to_date"] <= TANKING_WIN_PCT) | (log["season_win_pct_to_date"] >= CLINCHED_WIN_PCT)
    log["playing_out_string"] = (late & no_stakes).astype(int)

    base_cols = ["ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak", "playing_out_string"]
    home_feats = log[log["is_home"]][["game_id"] + base_cols].rename(columns={
        "ats_pct_l10": "home_ats_pct_l10", "win_pct_l10": "home_win_pct_l10",
        "pf_l10": "home_pf_l10", "pa_l10": "home_pa_l10",
        "rest_days": "home_rest_days", "streak": "home_streak",
        "playing_out_string": "home_playing_out_string",
    })
    away_feats = log[~log["is_home"]][["game_id"] + base_cols].rename(columns={
        "ats_pct_l10": "away_ats_pct_l10", "win_pct_l10": "away_win_pct_l10",
        "pf_l10": "away_pf_l10", "pa_l10": "away_pa_l10",
        "rest_days": "away_rest_days", "streak": "away_streak",
        "playing_out_string": "away_playing_out_string",
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
    out["home_playing_out_string"] = out["home_playing_out_string"].fillna(0).astype(int)
    out["away_playing_out_string"] = out["away_playing_out_string"].fillna(0).astype(int)
    # playoff games are a different structure entirely (every team present still has
    # something to play for, by definition) — zero the flag out there rather than let
    # the calendar-week approximation (which runs past 18 into playoff dates) misfire
    out.loc[out["is_playoff"], "home_playing_out_string"] = 0
    out.loc[out["is_playoff"], "away_playing_out_string"] = 0

    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]

    out["naive_total"], out["naive_margin"] = naive_score_features(
        out["home_pf_l10"], out["home_pa_l10"], out["away_pf_l10"], out["away_pa_l10"]
    )
    out["home_pyth_pct"] = pythagorean_win_pct(out["home_pf_l10"], out["home_pa_l10"])
    out["away_pyth_pct"] = pythagorean_win_pct(out["away_pf_l10"], out["away_pa_l10"])
    out["pyth_pct_diff"] = out["home_pyth_pct"] - out["away_pyth_pct"]
    return out


POSITIONING_FEATURE_COLS = ["home_playing_out_string", "away_playing_out_string"]
VARIANT_FEATURE_COLS = ML_FEATURE_COLS + POSITIONING_FEATURE_COLS


def run_pipeline(feats: pd.DataFrame, feature_cols: list, label: str) -> dict:
    hr(f"WALK-FORWARD ML — {label}")
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    print(f"OOS rows: {len(wf.predictions):,}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")
    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])
    print(f"Margin corr: {margin_corr:.4f}   Total corr: {total_corr:.4f}")

    stds = ensemble.compute_residual_stds(oos, nfl_config.ELO_POINTS_PER_MARGIN)

    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"))
    bets = backtest.run_backtest(oos, stds, nfl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"Qualifying bets: {len(bets):,}")
    if bets.empty:
        raise RuntimeError(f"{label}: no bets cleared thresholds.")

    summary = backtest.summarize(bets)
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    return {
        "margin_corr": margin_corr, "total_corr": total_corr,
        "roi_pct": float(summary["roi_pct"].iloc[0]),
        "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
        "n_bets": int(summary["bets"].iloc[0]),
    }


def main():
    hr("0. LOAD + POWER RATINGS (shared)")
    games = load_games()
    games = attach_weather(games)
    print(f"Rows: {len(games):,}, seasons {games['season'].min()}-{games['season'].max()}")

    rating_cfg = PowerRatingConfig(
        k_factor=nfl_config.ELO_K_FACTOR, start_rating=nfl_config.ELO_START_RATING,
        home_field_adv=nfl_config.HOME_FIELD_ADV_ELO, season_regression=nfl_config.SEASON_REGRESSION,
        mov_mult_base=nfl_config.MOV_MULT_BASE, mov_mult_divisor=nfl_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", neutral_col="is_neutral_venue",
        config=rating_cfg,
    )
    baseline_feats = nfl_features.build_features(games, rr.history)

    # ------------------------------------------------------------------
    # DIAGNOSTIC (free, not a hypothesis test): is is_divisional actually
    # pulling any weight in the CURRENT production feature set?
    # ------------------------------------------------------------------
    hr("DIAGNOSTIC: permutation feature importance, current production features")
    importances = compute_feature_importance(baseline_feats, ML_FEATURE_COLS)
    for target in ("margin", "total"):
        print(f"\n-- {target} model, ranked --")
        for row in importances[target]:
            print(f"  {row['feature']:<22s} {row['importance_mean']:+.5f}  (std {row['importance_std']:.5f})")
        div_row = next(r for r in importances[target] if r["feature"] == "is_divisional")
        rank = importances[target].index(div_row) + 1
        print(f"  -> is_divisional: rank {rank}/{len(ML_FEATURE_COLS)}, importance {div_row['importance_mean']:+.5f}")

    # ------------------------------------------------------------------
    # HYPOTHESIS: playing_out_string (season positioning / motivation proxy)
    # ------------------------------------------------------------------
    hypothesis = Hypothesis(
        name="late_season_playing_out_string",
        reasoning=(
            "NFL betting analytics well documents that teams with nothing left to play for "
            "late in the season (mathematically/practically eliminated, or already locked into "
            "their playoff seed) play with reduced urgency (rested starters, conservative "
            "gameplanning) relative to teams still fighting for a spot -- a real, externally "
            "motivated situational effect distinct from team strength/form, which the current "
            "feature set has no representation of at all (is_playoff flags playoff GAMES, not "
            "regular-season stakes). Built as a walk-forward-safe proxy from data already "
            "loaded: each team's own cumulative in-season win pct to date (current game always "
            "excluded) combined with an approximate week-of-season derived from the calendar, "
            "flagging teams in the week>=15 stretch run sitting at <=0.25 or >=0.85 season win "
            "pct as 'playing out the string'. Not exact playoff elimination math (that needs "
            "full standings/tiebreakers/remaining schedule, out of scope and error-prone) but a "
            "reasonable, honestly-labeled proxy for contention status."
        ),
        sport="NFL",
    )
    hr("HYPOTHESIS")
    print(f"Hypothesis: {hypothesis.name}\nReasoning: {hypothesis.reasoning}\n")

    hr("VARIANT FEATURES (+ home/away_playing_out_string)")
    variant_feats = build_features_with_positioning(games, rr.history)
    late_rows = variant_feats[(variant_feats["home_playing_out_string"] == 1) |
                               (variant_feats["away_playing_out_string"] == 1)]
    print(f"VARIANT_FEATURE_COLS: {len(VARIANT_FEATURE_COLS)} ({len(POSITIONING_FEATURE_COLS)} new: {POSITIONING_FEATURE_COLS})")
    print(f"Games with at least one team flagged 'playing out the string': {len(late_rows):,} / {len(variant_feats):,} "
          f"({len(late_rows) / len(variant_feats) * 100:.1f}%)")

    shared_cols = [c for c in ML_FEATURE_COLS if c in baseline_feats.columns and c in variant_feats.columns]
    mism = sum(
        1 for c in shared_cols
        if not np.allclose(baseline_feats[c].fillna(-999999), variant_feats[c].fillna(-999999))
    )
    print(f"Shared base-column identity check: {len(shared_cols) - mism}/{len(shared_cols)} identical.")

    baseline = run_pipeline(baseline_feats, ML_FEATURE_COLS, "BASELINE")
    variant = run_pipeline(variant_feats, VARIANT_FEATURE_COLS, "VARIANT (+playing_out_string)")

    baseline_metrics = {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}
    variant_metrics = {k: variant[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}

    hr("HYPOTHESIS EVALUATION")
    print(f"Baseline n_bets={baseline['n_bets']:,}  Variant n_bets={variant['n_bets']:,}")
    result = evaluate_hypothesis(hypothesis, baseline_metrics, variant_metrics)

    hr("RESULT")
    import json
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
