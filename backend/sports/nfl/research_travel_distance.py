"""
THROWAWAY research script — NOT wired into main.py/pipeline.py/harness.py.
Tests one hypothesis via core/research.py's disciplined evaluate_hypothesis()
loop, following the exact shape of research_scoring_trend.py/
research_season_positioning.py: loader -> power ratings -> features
(baseline vs variant) -> walk-forward ML -> residual stds -> backtest ->
evaluate_hypothesis().

Hypothesis: does the AWAY team's time-zone travel distance for this specific
matchup (a proxy for cross-country jet lag / circadian disruption, distinct
from simple rest-day count) improve NFL margin/total prediction?

pipeline.py's build_nfl_pipeline() was read ONLY as a reference for how NFL
wires loader/features/ratings/config together; not imported, not modified.

Run with:  python -m sports.nfl.research_travel_distance   (from backend/)

OUTCOME (2026-08-25, logged in decision_log.jsonl): recommendation
"adopt_cautiously". Baseline reproduced the documented current NFL numbers
exactly (margin_corr 0.362405, total_corr 0.207985, 6,151 bets, ROI -0.296%
+-1.537pp) before the variant was ever run. Statistical fit barely moved
(margin_corr 0.362405->0.362175, -0.00023; total_corr 0.207985->0.208763,
+0.00078 — both well inside the 0.005 noise floor), while backtest ROI
dropped from -0.296% to -1.561% (-1.27pp) — a real dip, but inside its own
+-1.54pp standard error once Bonferroni-adjusted for this project's
accumulated test count, so per core/research.py's rules this is not
"reject" (fit didn't degrade past the floor) and not a real "adopt" (fit
didn't improve either) — a clean, harmless null, the same profile as NFL's
own late_season_playing_out_string and MLB's travel_fatigue tests.
NOT ADOPTED into ML_FEATURE_COLS: per this project's standing practice
(NFL playing_out_string, NBA star-venue-split, NHL player-matchup, MLB
travel-fatigue — every prior "adopt_cautiously" null in this codebase was
judged the same way), an adopt_cautiously label is a floor, not a mandate,
and a feature that measurably moves nothing on the fit axis (while ROI
moved negative, even if within noise) isn't worth the added pipeline
surface (a new 32-team lookup table, a new column, ongoing maintenance) for
zero offsetting benefit — especially wired straight into the pipeline two
weeks before a live season starts. features.py, config.py, and loader.py
were left exactly as found. This file and the decision_log.jsonl entry
(`Hypothesis test: away_team_travel_distance`) are the permanent record.
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
from sports.nfl.features import ML_FEATURE_COLS
from sports.nfl.weather import attach_weather

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Each of the 32 current NFL franchises' home city, expressed as the real
# IANA timezone for that city — a real, verifiable, low-ambiguity fact (not
# fabricated), matching the exact style/rigor precedent set by
# sports/mlb/features.py's TEAM_TIMEZONE dict: keyed by this sport's own
# canonical franchise identifier (config.FRANCHISE_CANONICAL / the same
# strings config.DIVISIONS already uses), one real city per team.
#
# Two teams get a specific, deliberate call worth documenting rather than
# glossing over:
#   - Indianapolis Colts: America/Indiana/Indianapolis, which has observed
#     Eastern Time (with DST) statewide since 2006 — the correct IANA zone
#     for the entire span this dataset covers, not a coincidence with
#     America/New_York.
#   - Arizona Cardinals: America/Phoenix, which does NOT observe DST. For
#     most of the year that's genuinely Mountain time, but during U.S. DST
#     (roughly mid-March-early November, which overlaps the back half of
#     the NFL preseason and the front half of the regular season) Phoenix's
#     clock is identical to Pacific, not Mountain. This feature deliberately
#     does NOT model that seasonal wrinkle — it's simplified to Arizona's
#     nominal geographic zone (Mountain) year-round, per the task's own
#     "simpler proxy: which of 4 US time zones" framing. Documented here so
#     it's a known, deliberate simplification rather than a silent error.
# ---------------------------------------------------------------------------
TEAM_TIMEZONE = {
    "Buffalo Bills": "America/New_York", "Miami Dolphins": "America/New_York",
    "New England Patriots": "America/New_York", "New York Jets": "America/New_York",
    "Baltimore Ravens": "America/New_York", "Cincinnati Bengals": "America/New_York",
    "Cleveland Browns": "America/New_York", "Pittsburgh Steelers": "America/New_York",
    "Houston Texans": "America/Chicago", "Indianapolis Colts": "America/Indiana/Indianapolis",
    "Jacksonville Jaguars": "America/New_York", "Tennessee Titans": "America/Chicago",
    "Denver Broncos": "America/Denver", "Kansas City Chiefs": "America/Chicago",
    "Raiders": "America/Los_Angeles", "Chargers": "America/Los_Angeles",
    "Dallas Cowboys": "America/Chicago", "New York Giants": "America/New_York",
    "Philadelphia Eagles": "America/New_York", "Washington": "America/New_York",
    "Chicago Bears": "America/Chicago", "Detroit Lions": "America/New_York",
    "Green Bay Packers": "America/Chicago", "Minnesota Vikings": "America/Chicago",
    "Atlanta Falcons": "America/New_York", "Carolina Panthers": "America/New_York",
    "New Orleans Saints": "America/Chicago", "Tampa Bay Buccaneers": "America/New_York",
    "Arizona Cardinals": "America/Phoenix",  # nominal Mountain zone (no DST) — see docstring above
    "San Francisco 49ers": "America/Los_Angeles", "Seattle Seahawks": "America/Los_Angeles",
    "Rams": "America/Los_Angeles",
}

# IANA zone name -> ordinal rank among the 4 continental US zones this
# league's 32 teams actually span (Eastern < Central < Mountain < Pacific).
# The RANK is what matters for "zones crossed," not the raw UTC offset,
# which is why Indianapolis (Eastern-with-DST) and Phoenix (Mountain,
# no-DST) both map cleanly onto a single rank each despite each having a
# real-world DST quirk documented above.
ZONE_RANK = {
    "America/New_York": 0, "America/Indiana/Indianapolis": 0,
    "America/Chicago": 1,
    "America/Denver": 2, "America/Phoenix": 2,
    "America/Los_Angeles": 3,
}


def zones_crossed(games: pd.DataFrame) -> pd.Series:
    """
    away_zones_crossed: the absolute number of the 4 US time zones separating
    the away team's home city from the home team's home city for THIS
    specific matchup — 0 if same zone, up to 3 for a coast-to-coast (Pacific
    <-> Eastern) trip. This is a pure, static function of which two teams
    are playing (both known well in advance of kickoff) — it does not depend
    on any team's rolling history, prior game location, or the outcome of
    this or any other game, so it carries zero walk-forward leakage risk by
    construction, unlike every rolling feature in features.py which needs an
    explicit shift(1) to be safe.

    Neutral-venue games (78 of 5,431 — Super Bowl/international) are zeroed
    out rather than computed: per METHODOLOGY.md, the "home team" label on a
    neutral-site game is a scheduling artifact, not a real home crowd/city,
    and this dataset has no actual venue-city column to locate the real game
    site for these rows. Guessing at a travel distance from a label known to
    be fictional would be worse than admitting the data doesn't support a
    real answer here — same judgment call `power_ratings.py` already makes
    by zeroing home-field advantage for these exact rows.
    """
    home_zone = games["home_franchise"].map(TEAM_TIMEZONE).map(ZONE_RANK)
    away_zone = games["away_franchise"].map(TEAM_TIMEZONE).map(ZONE_RANK)
    crossed = (home_zone - away_zone).abs()
    crossed = crossed.where(~games["is_neutral_venue"].fillna(False), 0)
    return crossed.fillna(0).astype(int)


TRAVEL_FEATURE_COLS = ["away_zones_crossed"]
VARIANT_FEATURE_COLS = ML_FEATURE_COLS + TRAVEL_FEATURE_COLS


# ---------------------------------------------------------------------------
# Variant feature builder: reuses features.build_features() verbatim (no
# reimplementation of _team_game_log/_rolling_form needed at all, unlike the
# scoring-trend/season-positioning research scripts) and merges on the one
# new static column by game_id — since away_zones_crossed depends on nothing
# but the two teams' identities for this game, there is no rolling logic to
# duplicate, and the shared columns are trivially bit-identical to baseline
# by construction (same DataFrame, one extra column).
# ---------------------------------------------------------------------------
def build_features_with_travel(games: pd.DataFrame, rating_history: pd.DataFrame) -> pd.DataFrame:
    feats = nfl_features.build_features(games, rating_history)
    travel = games[["game_id", "home_franchise", "away_franchise", "is_neutral_venue"]].copy()
    travel["away_zones_crossed"] = zones_crossed(travel)
    feats = feats.merge(travel[["game_id", "away_zones_crossed"]], on="game_id", how="left")
    feats["away_zones_crossed"] = feats["away_zones_crossed"].fillna(0).astype(int)
    return feats


def run_pipeline(feats: pd.DataFrame, feature_cols: list, label: str) -> dict:
    hr(f"WALK-FORWARD ML — {label}")
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    print(f"Seasons predicted out-of-sample: {wf.seasons_predicted}")
    print(f"OOS rows: {len(wf.predictions):,}")
    print(f"Margin residual std: {wf.margin_residual_std:.2f}")
    print(f"Total residual std:  {wf.total_residual_std:.2f}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])
    print(f"Margin: ML pred vs actual correlation: {margin_corr:.4f}")
    print(f"Total:  ML pred vs actual correlation: {total_corr:.4f}")

    hr(f"ENSEMBLE RESIDUAL STDS — {label}")
    stds = ensemble.compute_residual_stds(oos, nfl_config.ELO_POINTS_PER_MARGIN)
    print(stds)

    hr(f"BACKTEST — {label}")
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"))
    bets = backtest.run_backtest(oos, stds, nfl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"Qualifying bets (edge >= {bt_cfg.min_edge_pct}%, confidence in {bt_cfg.allowed_confidence}, "
          f"priced at {bt_cfg.price_point}): {len(bets):,}")

    if bets.empty:
        raise RuntimeError(f"{label}: no bets cleared thresholds — cannot compute ROI.")

    summary = backtest.summarize(bets)
    roi_pct = float(summary["roi_pct"].iloc[0])
    roi_stderr_pct = float(summary["roi_stderr_pct"].iloc[0])
    n_bets = int(summary["bets"].iloc[0])
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    return {
        "margin_corr": margin_corr, "total_corr": total_corr,
        "roi_pct": roi_pct, "roi_stderr_pct": roi_stderr_pct,
        "n_bets": n_bets,
    }


def main():
    # ------------------------------------------------------------------
    # 0. Hypothesis object — construction fails loudly without real reasoning
    # ------------------------------------------------------------------
    hypothesis = Hypothesis(
        name="away_team_travel_distance",
        reasoning=(
            "NFL betting analytics well documents that long-distance travel for the away "
            "team -- especially cross-country trips spanning 2+ time zones -- is a real "
            "situational factor distinct from team strength or rest days alone: a team can "
            "have a completely normal rest-day count and still face genuine jet lag / "
            "circadian disruption from a coast-to-coast flight, which is a different "
            "physiological effect than simply having fewer days between games. The current "
            "feature set already has rest_diff/home_rest_days/away_rest_days (time between "
            "games) but nothing capturing travel DISTANCE or time-zone change specifically -- "
            "confirmed genuinely untested by checking decision_log.jsonl's 17 prior NFL "
            "entries (pythagorean_feature, over_under_tendency_l10, scoring_trend_l3_vs_l10, "
            "late_season_playing_out_string, and the deliberate fake_threshold_tweak guardrail "
            "self-test are the only prior NFL hypotheses -- none touch travel distance or time "
            "zones). Built as a static, walk-forward-safe-by-construction feature (depends "
            "only on which two teams are playing this specific game, not on any rolling "
            "history or outcome): away_zones_crossed, the absolute number of US time zones "
            "(0-3) separating the away team's real home city from the home team's real home "
            "city, using each of the 32 teams' actual home-city IANA timezone -- the same "
            "real, verifiable, low-ambiguity-fact standard as MLB's existing TEAM_TIMEZONE "
            "table in sports/mlb/features.py."
        ),
        sport="NFL",
    )
    print(f"Hypothesis: {hypothesis.name}\nReasoning: {hypothesis.reasoning}\n")

    # ------------------------------------------------------------------
    # 1. Load (shared by both baseline and variant)
    # ------------------------------------------------------------------
    hr("1. LOAD")
    games = load_games()
    games = attach_weather(games)  # disk-cached; same as build_nfl_pipeline()
    print(f"Rows after load/clean: {len(games):,}")
    print(f"Season range: {games['season'].min()} - {games['season'].max()}")

    # sanity: every team appearing in the data must resolve to a real timezone
    teams = sorted(set(games["home_franchise"]) | set(games["away_franchise"]))
    unmapped = [t for t in teams if t not in TEAM_TIMEZONE]
    print(f"Distinct franchises: {len(teams)}, unmapped in TEAM_TIMEZONE: {unmapped}")
    assert not unmapped, f"TEAM_TIMEZONE is missing real teams: {unmapped}"

    # ------------------------------------------------------------------
    # 2. Power ratings (shared)
    # ------------------------------------------------------------------
    hr("2. POWER RATINGS")
    rating_cfg = PowerRatingConfig(
        k_factor=nfl_config.ELO_K_FACTOR, start_rating=nfl_config.ELO_START_RATING,
        home_field_adv=nfl_config.HOME_FIELD_ADV_ELO, season_regression=nfl_config.SEASON_REGRESSION,
        mov_mult_base=nfl_config.MOV_MULT_BASE, mov_mult_divisor=nfl_config.MOV_MULT_DIVISOR,
    )
    print(f"Config: k={rating_cfg.k_factor}, hfa={rating_cfg.home_field_adv}, "
          f"season_regression={rating_cfg.season_regression}, "
          f"mov_base={rating_cfg.mov_mult_base}, mov_divisor={rating_cfg.mov_mult_divisor}, "
          f"elo_points_per_margin={nfl_config.ELO_POINTS_PER_MARGIN}")

    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", neutral_col="is_neutral_venue",
        config=rating_cfg,
    )

    # ------------------------------------------------------------------
    # 3a. BASELINE features (current features.py, unmodified)
    # ------------------------------------------------------------------
    hr("3a. BASELINE FEATURES (features.py unmodified)")
    baseline_feats = nfl_features.build_features(games, rr.history)
    print(f"Feature rows: {len(baseline_feats):,}, ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")

    # ------------------------------------------------------------------
    # 3b. VARIANT features (+ away_zones_crossed)
    # ------------------------------------------------------------------
    hr("3b. VARIANT FEATURES (+ away_zones_crossed)")
    variant_feats = build_features_with_travel(games, rr.history)
    print(f"Feature rows: {len(variant_feats):,}, VARIANT_FEATURE_COLS: {len(VARIANT_FEATURE_COLS)} "
          f"({len(TRAVEL_FEATURE_COLS)} new: {TRAVEL_FEATURE_COLS})")
    print(variant_feats["away_zones_crossed"].value_counts().sort_index())
    non_zero = (variant_feats["away_zones_crossed"] > 0).mean() * 100
    print(f"Games with any away-team time-zone travel: {non_zero:.1f}%")

    # sanity check: variant's shared/base columns must be bit-identical to baseline's
    shared_cols = [c for c in ML_FEATURE_COLS if c in baseline_feats.columns and c in variant_feats.columns]
    mism = 0
    for c in shared_cols:
        if not np.allclose(baseline_feats[c].fillna(-999999), variant_feats[c].fillna(-999999)):
            mism += 1
            print(f"  MISMATCH in shared column: {c}")
    print(f"Shared base-column identity check: {len(shared_cols) - mism}/{len(shared_cols)} identical.")

    # ------------------------------------------------------------------
    # 4. Run both pipelines through walk-forward ML -> stds -> backtest
    # ------------------------------------------------------------------
    baseline = run_pipeline(baseline_feats, ML_FEATURE_COLS, "BASELINE")
    variant = run_pipeline(variant_feats, VARIANT_FEATURE_COLS, "VARIANT (+away_zones_crossed)")

    baseline_metrics = {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}
    variant_metrics = {k: variant[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}

    # ------------------------------------------------------------------
    # 5. Evaluate via core/research.py's disciplined loop (auto-logs to decision_log.jsonl)
    # ------------------------------------------------------------------
    hr("5. HYPOTHESIS EVALUATION")
    print(f"Baseline n_bets={baseline['n_bets']:,}  Variant n_bets={variant['n_bets']:,}")
    result = evaluate_hypothesis(hypothesis, baseline_metrics, variant_metrics)

    hr("RESULT")
    import json
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
