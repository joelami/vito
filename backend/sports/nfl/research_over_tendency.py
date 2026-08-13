"""
THROWAWAY research script #2 — NOT wired into main.py/pipeline.py/harness.py.
Second, distinct hypothesis test via core/research.py's disciplined
evaluate_hypothesis() loop, run after research_scoring_trend.py (which came
back "inconclusive"). Mirrors that script's shape (loader -> power ratings
-> features -> walk-forward ML -> residual stds -> backtest -> summarize).

Hypothesis: `over_result_close` (the game-level over/under outcome vs the
closing total) is loaded by loader.py but never turned into a rolling
per-team feature — unlike its exact spread analog, `home_covers_close`,
which DOES become a rolling feature (`ats_pct_l10`) already proven useful
enough to be in ML_FEATURE_COLS. There is no equivalent `over_pct_l10`. A
team's trailing tendency for its games to go over/under the closing total
is real, market-relative information (opponent pace forced, situational
tendencies already priced by the market) distinct from raw pf_l10/pa_l10
scoring level or the L3-vs-L10 trend already tested (inconclusive) — and it
targets the same diagnosed totals weakness. Zero new data source: this is
already-loaded data, walk-forward-safe via the same shift(1)-before-rolling
discipline as ats_pct_l10.

pipeline.py read ONLY as reference; not imported or modified.

Run with:  python -m sports.nfl.research_over_tendency   (from backend/)
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


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Variant feature builder: same as features.build_features, but adds
# home_over_pct_l10 / away_over_pct_l10 — each team's trailing rate (walk-
# forward, shift(1)-before-rolling) of its games going OVER the closing
# total. Unlike ATS cover (home-perspective, needs a sign flip for the away
# row), over/under is a game-level outcome shared identically by both teams
# in that game, so no sign flip is needed — both rows just inherit the same
# game-level result.
# ---------------------------------------------------------------------------
ROLL_WINDOW = nfl_features.ROLL_WINDOW  # 10, same window as every other L10 feature


def _team_game_log_with_over(games: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "date", "season", "home_covers_close", "over_result_close",
            "home_win", "home_score", "away_score"]

    home = games[cols + ["home_franchise", "away_franchise"]].rename(columns={
        "home_franchise": "team", "away_franchise": "opponent", "home_covers_close": "covered",
        "home_win": "won", "home_score": "points_for", "away_score": "points_against",
    })
    home["is_home"] = True

    away = games[cols + ["away_franchise", "home_franchise"]].rename(columns={
        "away_franchise": "team", "home_franchise": "opponent", "home_covers_close": "covered",
        "home_win": "won", "home_score": "points_against", "away_score": "points_for",
    })
    away["covered"] = away["covered"].apply(lambda v: -v if pd.notna(v) else v)
    away["won"] = 1 - away["won"]
    away["is_home"] = False

    log = pd.concat([home, away], ignore_index=True)
    log["covered"] = log["covered"].map({1: 1.0, 0: 0.5, -1: 0.0})
    # over_result_close is a GAME-level outcome (not home-perspective) — both
    # teams' rows inherit it identically, no sign flip.
    log["over"] = log["over_result_close"].map({1: 1.0, 0: 0.5, -1: 0.0})
    return log.sort_values(["team", "date"], kind="stable")


def build_features_with_over(games: pd.DataFrame, rating_history: pd.DataFrame) -> pd.DataFrame:
    log = _team_game_log_with_over(games)
    grp = log.groupby("team", group_keys=False)

    log["ats_pct_l10"] = grp["covered"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    log["win_pct_l10"] = grp["won"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    log["pf_l10"] = grp["points_for"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    log["pa_l10"] = grp["points_against"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    log["over_pct_l10"] = grp["over"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    log["rest_days"] = grp["date"].apply(lambda s: s.diff().dt.days)

    def _streak(sub: pd.DataFrame) -> pd.Series:
        prior = sub["won"].shift(1)
        streaks, cur = [], 0
        for w in prior:
            if pd.isna(w):
                streaks.append(0)
                cur = 0
                continue
            cur = (cur + 1 if cur >= 0 else 1) if w == 1 else (cur - 1 if cur <= 0 else -1)
            streaks.append(cur)
        return pd.Series(streaks, index=sub.index)

    log["streak"] = grp.apply(_streak).reset_index(level=0, drop=True)

    base_cols = ["ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak", "over_pct_l10"]
    home_feats = log[log["is_home"]][["game_id"] + base_cols].rename(columns={
        "ats_pct_l10": "home_ats_pct_l10", "win_pct_l10": "home_win_pct_l10",
        "pf_l10": "home_pf_l10", "pa_l10": "home_pa_l10",
        "rest_days": "home_rest_days", "streak": "home_streak",
        "over_pct_l10": "home_over_pct_l10",
    })
    away_feats = log[~log["is_home"]][["game_id"] + base_cols].rename(columns={
        "ats_pct_l10": "away_ats_pct_l10", "win_pct_l10": "away_win_pct_l10",
        "pf_l10": "away_pf_l10", "pa_l10": "away_pa_l10",
        "rest_days": "away_rest_days", "streak": "away_streak",
        "over_pct_l10": "away_over_pct_l10",
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
    # no over/under history yet (or no total-line coverage that far back, e.g.
    # 2006-2013) -> neutral 0.5 fill, same convention as ats_pct_l10
    out["home_over_pct_l10"] = out["home_over_pct_l10"].fillna(0.5)
    out["away_over_pct_l10"] = out["away_over_pct_l10"].fillna(0.5)

    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]

    out["naive_total"], out["naive_margin"] = naive_score_features(
        out["home_pf_l10"], out["home_pa_l10"], out["away_pf_l10"], out["away_pa_l10"]
    )
    out["home_pyth_pct"] = pythagorean_win_pct(out["home_pf_l10"], out["home_pa_l10"])
    out["away_pyth_pct"] = pythagorean_win_pct(out["away_pf_l10"], out["away_pa_l10"])
    out["pyth_pct_diff"] = out["home_pyth_pct"] - out["away_pyth_pct"]
    return out


OVER_FEATURE_COLS = ["home_over_pct_l10", "away_over_pct_l10"]
VARIANT_FEATURE_COLS = ML_FEATURE_COLS + OVER_FEATURE_COLS


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
    hypothesis = Hypothesis(
        name="over_under_tendency_l10",
        reasoning=(
            "loader.py already computes over_result_close (a game's over/under outcome vs "
            "the closing total) into every row, but features.py only ever turns its exact "
            "spread analog (home_covers_close) into a rolling per-team feature (ats_pct_l10, "
            "already in ML_FEATURE_COLS) - there is no equivalent over_pct_l10. A team's "
            "trailing over/under tendency is real market-relative information (opponent pace "
            "forced, situational tendencies the market already priced) distinct from raw "
            "pf_l10/pa_l10 level or the L3-vs-L10 trend already tested inconclusive, and "
            "targets the same diagnosed totals weakness. Zero new data source, same "
            "walk-forward-safe shift(1)-before-rolling discipline as ats_pct_l10."
        ),
        sport="NFL",
    )
    print(f"Hypothesis: {hypothesis.name}\nReasoning: {hypothesis.reasoning}\n")

    hr("1. LOAD")
    games = load_games()
    games = attach_weather(games)
    print(f"Rows: {len(games):,}, seasons {games['season'].min()}-{games['season'].max()}")
    has_total = games["over_result_close"].notna()
    print(f"Games with a usable over_result_close: {has_total.sum():,} ({has_total.mean()*100:.1f}%)")

    hr("2. POWER RATINGS")
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

    hr("3a. BASELINE FEATURES (features.py unmodified)")
    baseline_feats = nfl_features.build_features(games, rr.history)
    print(f"ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")

    hr("3b. VARIANT FEATURES (+ home/away_over_pct_l10)")
    variant_feats = build_features_with_over(games, rr.history)
    print(f"VARIANT_FEATURE_COLS: {len(VARIANT_FEATURE_COLS)} ({len(OVER_FEATURE_COLS)} new: {OVER_FEATURE_COLS})")

    shared_cols = [c for c in ML_FEATURE_COLS if c in baseline_feats.columns and c in variant_feats.columns]
    mism = sum(
        1 for c in shared_cols
        if not np.allclose(baseline_feats[c].fillna(-999999), variant_feats[c].fillna(-999999))
    )
    print(f"Shared base-column identity check: {len(shared_cols) - mism}/{len(shared_cols)} identical.")

    baseline = run_pipeline(baseline_feats, ML_FEATURE_COLS, "BASELINE")
    variant = run_pipeline(variant_feats, VARIANT_FEATURE_COLS, "VARIANT (+over/under tendency)")

    baseline_metrics = {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}
    variant_metrics = {k: variant[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}

    hr("5. HYPOTHESIS EVALUATION")
    print(f"Baseline n_bets={baseline['n_bets']:,}  Variant n_bets={variant['n_bets']:,}")
    result = evaluate_hypothesis(hypothesis, baseline_metrics, variant_metrics)

    hr("RESULT")
    import json
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
