"""
THROWAWAY research script — NOT wired into main.py/pipeline.py/harness.py.

Follow-up to the ALREADY-REJECTED (adopt_cautiously/null) league-wide
`away_zones_crossed` test in research_travel_distance.py. That test pooled
every team crossing every number of zones into one static feature and found
nothing (margin_corr -0.0002, total_corr +0.0008, both inside the 0.005
noise floor). The app owner's follow-up question: could a league-wide
average be hiding a REAL effect concentrated in one specific, small group,
rather than proving no group has one?

This tests exactly ONE pre-specified subgroup, chosen for a real, external,
non-data-driven reason decided BEFORE looking at any results — NOT a sweep
over teams/thresholds/windows looking for the best-looking one. That would
be precisely the multiple-comparisons trap this project's own methodology
(core/research.py's docstring, the fake_threshold_tweak guardrail test,
METHODOLOGY.md's multiple-comparisons discussion) exists to prevent: check
enough subgroups and something looks "significant" by pure chance even with
zero real effect anywhere. Testing all 32 teams, or a handful hand-picked
after eyeballing the data, is exactly what's NOT done here.

The one group tested: Pacific-timezone-home NFL franchises (Raiders,
Chargers, San Francisco 49ers, Seattle Seahawks, Rams) — the specific group
NFL analytics literature repeatedly names for a body-clock-mismatch
road-disadvantage claim (Pacific team's internal clock reads ~10am at a
1pm ET kickoff), a distinct claim from "long travel is bad for anyone,"
and the exact group already called out in research_travel_distance.py's own
TEAM_TIMEZONE table as sharing America/Los_Angeles. Reuses that script's
TEAM_TIMEZONE/ZONE_RANK dicts and run_pipeline()/hr() helpers verbatim
(imported, not re-derived) per the same "one real city per team" standard.

A real, permanent, separately-confirmed data limitation worth stating even
though it doesn't block this specific test: sports.nfl.loader.load_games()'s
`date` column has NO time-of-day component at all (every one of 5,431 rows
is midnight, 00:00:00) — confirmed directly. A genuine "kickoff time /
circadian mismatch" hypothesis (e.g. distinguishing an actual 10am-body-
clock 1pm-ET kickoff from a more forgiving 4:05pm ET kickoff) CANNOT be
built from this dataset; there is no real kickoff-time column to build it
from, and this script does not fabricate one. What CAN be tested here is
the coarser, still-externally-motivated claim: does simply being a Pacific-
origin team on a real cross-timezone road trip (regardless of kickoff clock
time, which isn't in the data) show ANY fit improvement at all.

Run with:  python -m sports.nfl.research_pacific_travel_effect   (from backend/)

OUTCOME (2026-09-03, logged in decision_log.jsonl): recommendation
"adopt_cautiously" — a clean null, not a discovery. Baseline reproduced the
documented current NFL numbers exactly (margin_corr 0.362405, total_corr
0.207985, 6,151 bets, ROI -0.296% +-1.537pp) before the variant was ever
run. The single pre-specified feature (away_pacific_road_trip: 1 only when
the away team is one of the 5 Pacific franchises AND the home team is NOT
also Pacific AND the game isn't a neutral-site game — 654 of 5,431 games,
12.0%) moved statistical fit slightly NEGATIVE, but still well inside the
0.005 noise floor (margin_corr 0.362405->0.359792, -0.00261; total_corr
0.207985->0.205118, -0.00287 — both under the floor, so not "fit_degraded"
by evaluate_hypothesis()'s own -0.005 disqualifying threshold, but clearly
not an improvement either), while backtest ROI moved from -0.296% to
-0.653% (-0.357pp) against a +-1.531pp stderr — comfortably inside noise
even before the Bonferroni multiplier (~3.11 at 27 accumulated tests, i.e.
the bar was ~4.77pp and the actual move was 0.357pp). Season-based
split-half (2009-2016, n=2,136 games/255 treated vs. 2017-2025, n=2,494
games/305 treated) showed both correlations dipping in BOTH halves rather
than flipping sign (early: margin -0.0051/total -0.0039; late: margin
-0.0004/total -0.0021) — consistently non-positive, not a real effect that
happens to average out to zero, but a small, all-noise dip that happens to
point the same (uninteresting) direction in a small early sample and
nearly vanishes in the larger, more reliable later sample. That pattern
(shrinking toward zero as sample size grows) is itself evidence for noise,
not against it. Per this project's standing practice (identical judgment
already applied to the league-wide travel test, NFL playing_out_string,
NBA star-venue-split, NHL player-matchup, MLB travel-fatigue), a harmless
null that moves nothing meaningful on the fit axis is not worth the added
pipeline surface (a new indicator column, a live-scoring code path, ongoing
maintenance) for zero offsetting benefit. NOT ADOPTED into ML_FEATURE_COLS.
features.py, config.py, matchup.py, and loader.py were left exactly as
found. Isolating the exact group the literature names did NOT surface a
real effect that the league-wide pooled test was masking — a legitimate,
complete answer to "is it hiding in a subgroup," not an inconclusive one.
This file and the decision_log.jsonl entry (`Hypothesis test:
nfl_pacific_team_travel_effect`) are the permanent record.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core.research import Hypothesis, evaluate_hypothesis
from sports.nfl import config as nfl_config
from sports.nfl.loader import load_games
from sports.nfl import features as nfl_features
from sports.nfl.features import ML_FEATURE_COLS
from sports.nfl.weather import attach_weather
from sports.nfl.research_travel_distance import TEAM_TIMEZONE, ZONE_RANK, run_pipeline, hr
from core.power_ratings import compute_power_ratings, PowerRatingConfig

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


# ---------------------------------------------------------------------------
# The ONE pre-specified subgroup: NFL analytics literature's specific,
# repeatedly-named "West Coast road disadvantage" group — teams whose real
# home city is Pacific time. Verified below (assert) against
# research_travel_distance.py's own TEAM_TIMEZONE dict rather than
# re-typed/assumed, so this can't silently drift from that script's facts.
# ---------------------------------------------------------------------------
PACIFIC_TEAMS = {"Raiders", "Chargers", "San Francisco 49ers", "Seattle Seahawks", "Rams"}
PACIFIC_ZONE = "America/Los_Angeles"


def pacific_road_trip(games: pd.DataFrame) -> pd.Series:
    """
    away_pacific_road_trip: 1 only when ALL of —
      (a) the away team is one of the 5 Pacific-timezone franchises,
      (b) the home team is NOT also Pacific (i.e. this is a REAL
          cross-timezone trip for this specific team, not a Pacific-vs-
          Pacific divisional/conference game that involves zero body-clock
          disruption), and
      (c) the game isn't a neutral-site game (same judgment call as
          research_travel_distance.py's zones_crossed: a neutral-site
          "home team" label is a scheduling artifact, not a real city, and
          this dataset has no actual venue-city column for these rows).
    0 otherwise — including every home game and every Pacific-vs-Pacific
    road game for these 5 teams.

    Pure function of which two teams are playing (both known well before
    kickoff) — zero walk-forward leakage risk by construction, same as
    away_zones_crossed.
    """
    is_pacific_away = games["away_franchise"].isin(PACIFIC_TEAMS)
    home_zone = games["home_franchise"].map(TEAM_TIMEZONE)
    real_road_trip = is_pacific_away & (home_zone != PACIFIC_ZONE)
    real_road_trip = real_road_trip.where(~games["is_neutral_venue"].fillna(False), False)
    return real_road_trip.astype(int)


PACIFIC_FEATURE_COLS = ["away_pacific_road_trip"]
VARIANT_FEATURE_COLS = ML_FEATURE_COLS + PACIFIC_FEATURE_COLS


def build_features_with_pacific_travel(games: pd.DataFrame, rating_history: pd.DataFrame) -> pd.DataFrame:
    feats = nfl_features.build_features(games, rating_history)
    travel = games[["game_id", "home_franchise", "away_franchise", "is_neutral_venue"]].copy()
    travel["away_pacific_road_trip"] = pacific_road_trip(travel)
    feats = feats.merge(travel[["game_id", "away_pacific_road_trip"]], on="game_id", how="left")
    feats["away_pacific_road_trip"] = feats["away_pacific_road_trip"].fillna(0).astype(int)
    return feats


def season_split_half_check(oos_baseline: pd.DataFrame, oos_variant: pd.DataFrame) -> None:
    """
    Stability check per the task's process requirements: does whatever the
    full-sample delta shows hold up as a CONSISTENT direction across two
    halves of the walk-forward OOS seasons, or does it flip sign / vanish —
    the signature of noise rather than a real, sample-spanning effect?
    Report-only; does not feed evaluate_hypothesis() (which already governs
    adopt/reject on the full sample) — an extra honesty check before
    trusting any full-sample number as "real."
    """
    seasons = sorted(oos_variant["season"].dropna().unique())
    mid = seasons[len(seasons) // 2]
    halves = {
        f"early ({seasons[0]}-{mid - 1})": (oos_baseline[oos_baseline["season"] < mid],
                                             oos_variant[oos_variant["season"] < mid]),
        f"late ({mid}-{seasons[-1]})": (oos_baseline[oos_baseline["season"] >= mid],
                                        oos_variant[oos_variant["season"] >= mid]),
    }
    hr("SEASON SPLIT-HALF STABILITY CHECK")
    for label, (base_half, var_half) in halves.items():
        n_treated = int(var_half["away_pacific_road_trip"].sum())
        base_m = float(np.corrcoef(base_half["predicted_margin"], base_half["actual_margin"])[0, 1])
        var_m = float(np.corrcoef(var_half["predicted_margin"], var_half["actual_margin"])[0, 1])
        base_t = float(np.corrcoef(base_half["predicted_total"], base_half["actual_total"])[0, 1])
        var_t = float(np.corrcoef(var_half["predicted_total"], var_half["actual_total"])[0, 1])
        print(f"{label}: n_games={len(var_half):,}, n_pacific_road_trips={n_treated:,}")
        print(f"  margin_corr {base_m:.4f} -> {var_m:.4f} ({var_m - base_m:+.4f})")
        print(f"  total_corr  {base_t:.4f} -> {var_t:.4f} ({var_t - base_t:+.4f})")


def main():
    # ------------------------------------------------------------------
    # 0. Hypothesis object
    # ------------------------------------------------------------------
    hypothesis = Hypothesis(
        name="nfl_pacific_team_travel_effect",
        reasoning=(
            "The already-completed league-wide away_zones_crossed test (research_travel_distance.py, "
            "rejected/adopt_cautiously null: margin_corr -0.0002, total_corr +0.0008, both inside the "
            "0.005 noise floor) pooled every team crossing every number of zones into one static feature "
            "and found nothing on average. But NFL analytics literature specifically and repeatedly names "
            "Pacific-timezone (West Coast) teams as the group most discussed for travel-related road "
            "disadvantage -- the claim is that a Pacific-based team traveling east for an early-afternoon "
            "Eastern kickoff faces a body-clock mismatch, a distinct, narrower claim from 'long travel is "
            "bad for anyone.' If the real effect (if any) is concentrated specifically in this one named "
            "group rather than spread evenly across every team that crosses any number of zones, a "
            "league-wide pooled coefficient could show no average effect while still masking a real, "
            "isolated effect for this specific small group -- exactly the follow-up question the app "
            "owner asked after the league-wide null. Tested as ONE single pre-specified subgroup (not a "
            "sweep over teams/thresholds chosen after looking at results, which would be the exact "
            "multiple-comparisons trap this project's methodology exists to prevent): a static, "
            "walk-forward-safe-by-construction indicator (away_pacific_road_trip) that is 1 only when the "
            "away team is one of the 5 real Pacific-timezone franchises (Raiders, Chargers, San Francisco "
            "49ers, Seattle Seahawks, Rams -- confirmed against research_travel_distance.py's own "
            "TEAM_TIMEZONE dict) AND the home team is genuinely outside the Pacific zone (a real "
            "cross-timezone trip for this specific team, not a same-zone divisional game), 654 of 5,431 "
            "games (12.0%)."
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

    teams = sorted(set(games["home_franchise"]) | set(games["away_franchise"]))
    unmapped = [t for t in teams if t not in TEAM_TIMEZONE]
    assert not unmapped, f"TEAM_TIMEZONE is missing real teams: {unmapped}"
    unmapped_pacific = [t for t in PACIFIC_TEAMS if t not in teams]
    assert not unmapped_pacific, f"PACIFIC_TEAMS names don't match dataset franchise strings: {unmapped_pacific}"
    non_pacific_zones = {TEAM_TIMEZONE[t] for t in PACIFIC_TEAMS}
    assert non_pacific_zones == {PACIFIC_ZONE}, f"PACIFIC_TEAMS don't all map to {PACIFIC_ZONE}: {non_pacific_zones}"
    print(f"Confirmed: all 5 Pacific teams ({sorted(PACIFIC_TEAMS)}) map to {PACIFIC_ZONE}.")

    all_pacific_road_games = games["away_franchise"].isin(PACIFIC_TEAMS).sum()
    print(f"All road games for these 5 teams combined (any destination, incl. neutral-site): "
          f"{all_pacific_road_games:,}")

    # ------------------------------------------------------------------
    # 2. Power ratings (shared)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3a. BASELINE features (current features.py, unmodified)
    # ------------------------------------------------------------------
    hr("3a. BASELINE FEATURES (features.py unmodified)")
    baseline_feats = nfl_features.build_features(games, rr.history)
    print(f"Feature rows: {len(baseline_feats):,}, ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")

    # ------------------------------------------------------------------
    # 3b. VARIANT features (+ away_pacific_road_trip)
    # ------------------------------------------------------------------
    hr("3b. VARIANT FEATURES (+ away_pacific_road_trip)")
    variant_feats = build_features_with_pacific_travel(games, rr.history)
    print(f"Feature rows: {len(variant_feats):,}, VARIANT_FEATURE_COLS: {len(VARIANT_FEATURE_COLS)} "
          f"({len(PACIFIC_FEATURE_COLS)} new: {PACIFIC_FEATURE_COLS})")
    n_treated = int(variant_feats["away_pacific_road_trip"].sum())
    print(f"Games with away_pacific_road_trip=1: {n_treated:,} "
          f"({n_treated / len(variant_feats) * 100:.1f}% of all games)")

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
    variant = run_pipeline(variant_feats, VARIANT_FEATURE_COLS, "VARIANT (+away_pacific_road_trip)")

    print("\nBaseline documented-number check (per docs/METHODOLOGY.md): expect margin_corr~0.362405, "
          f"total_corr~0.207985 -> got margin_corr={baseline['margin_corr']:.6f}, "
          f"total_corr={baseline['total_corr']:.6f}")

    baseline_metrics = {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}
    variant_metrics = {k: variant[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}

    # ------------------------------------------------------------------
    # 5. Season-based split-half stability check (required before trusting
    #    any full-sample number) — recompute OOS frames directly since
    #    run_pipeline() doesn't return them.
    # ------------------------------------------------------------------
    from core.ml_models import walk_forward_predict

    wf_baseline = walk_forward_predict(baseline_feats, ML_FEATURE_COLS, min_train_seasons=3)
    oos_baseline = baseline_feats.set_index("game_id").join(wf_baseline.predictions, how="inner")
    wf_variant = walk_forward_predict(variant_feats, VARIANT_FEATURE_COLS, min_train_seasons=3)
    oos_variant = variant_feats.set_index("game_id").join(wf_variant.predictions, how="inner")
    season_split_half_check(oos_baseline, oos_variant)

    # ------------------------------------------------------------------
    # 6. Evaluate via core/research.py's disciplined loop (auto-logs to decision_log.jsonl)
    # ------------------------------------------------------------------
    hr("6. HYPOTHESIS EVALUATION")
    print(f"Baseline n_bets={baseline['n_bets']:,}  Variant n_bets={variant['n_bets']:,}")
    result = evaluate_hypothesis(hypothesis, baseline_metrics, variant_metrics)

    hr("RESULT")
    import json
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
