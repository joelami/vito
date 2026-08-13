"""
RESEARCH SCRIPT (throwaway, not part of the production module shape) --
tests ONE new, well-motivated hypothesis via core.research.evaluate_hypothesis,
following the exact discipline sports/cfb/research_week_trends.py used and
this project's own core/research.py enforces. Distinct from the two MLB
hypotheses already tested and adopt_cautiously'd (starting_pitcher_rolling_
er_per_start, bullpen_fatigue_relief_appearances_l3d -- see
decision_log.jsonl and sports/mlb/research_bullpen_fatigue.py): those both
measure PITCHER-side fatigue/quality. This one is about the TEAM as a whole
travelling with little or no recovery time before first pitch -- a
different, well-documented baseball phenomenon ("getaway day" scheduling,
where a team's last game of a homestand/road trip is deliberately scheduled
so the team can fly out immediately after) that shows up in broadcast
commentary and beat-writer coverage as a real fatigue factor, distinct from
both the already-tested pitcher signals and from the `rest_days` feature
already in the model (rest_days measures TIME since the last game; it says
nothing about whether that time included a change of city).

HYPOTHESIS, stated precisely up front: a team that (a) played its most
recent game at a DIFFERENT park than today's game (i.e. it changed cities
since its last game) AND (b) had <=1 day of rest before today is playing
under genuine, currently-uncaptured travel fatigue -- short-notice venue
change with minimal recovery. This is a strictly walk-forward-safe,
backward-looking feature (same shift(1)-before-lookup discipline every other
rolling feature in this project follows: only the team's own PAST games are
consulted, never anything about today's or a future game), built entirely
from `loader.py`'s existing `park_id` and `date` columns -- no new data
source needed.

SCOPING, stated honestly: this is a proxy, not a GPS/flight-plan model.
`park_id` changing between two consecutive games is a coarse signal -- a
team could conceivably play a getaway travel day within the SAME metro area
(no current MLB city has two parks, so this doesn't actually happen in this
data, but it's worth naming the limitation class) or travel a short regional
hop (e.g. NYA->BOS) vs a genuine cross-country flight (e.g. SEA->BOS) with
identical flag value. Distance-weighting would need each park's lat/long,
which this dataset doesn't carry -- out of scope for this pass, named
explicitly rather than faked with a guessed distance table.

Run with:  python -m sports.mlb.research_travel_fatigue   (from backend/)
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
from sports.mlb import config as mlb_config
from sports.mlb.loader import load_games
from sports.mlb.odds_loader import attach_odds
from sports.mlb.starting_pitcher import attach_starter_quality
from sports.mlb.features import build_features, ML_FEATURE_COLS


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Build the travel-fatigue flag directly from loader.py's park_id/date columns.
# Same long-format-team-log + shift(1) pattern as features.py's _team_game_log
# / _rolling_form, generalized to a park-change lookup instead of a rolling
# numeric mean.
# ---------------------------------------------------------------------------
def attach_travel_fatigue(games: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "date", "game_number", "park_id"]
    home = games[cols + ["home_franchise"]].rename(columns={"home_franchise": "team"})
    away = games[cols + ["away_franchise"]].rename(columns={"away_franchise": "team"})
    log = pd.concat([home, away], ignore_index=True)
    log = log.sort_values(["team", "date", "game_number"], kind="stable").reset_index(drop=True)

    grp = log.groupby("team", group_keys=False)
    log["prev_park_id"] = grp["park_id"].shift(1)
    log["prev_date"] = grp["date"].shift(1)
    log["days_since_prev"] = (log["date"] - log["prev_date"]).dt.days

    # First-ever game in the data for a team has no prior park to compare
    # against -> no travel signal either way, fill 0 (same "no history yet
    # means no fatigue" convention research_bullpen_fatigue.py uses for its
    # own first-appearance fallback).
    changed_park = (log["prev_park_id"].notna()) & (log["park_id"] != log["prev_park_id"])
    short_rest = log["days_since_prev"] <= 1
    log["just_traveled"] = changed_park.astype(int)
    log["short_rest_travel"] = (changed_park & short_rest).fillna(False).astype(int)

    home_f = log.merge(games[["game_id", "home_franchise"]], on="game_id")
    home_f = home_f[home_f["team"] == home_f["home_franchise"]][
        ["game_id", "just_traveled", "short_rest_travel"]
    ].rename(columns={"just_traveled": "home_just_traveled", "short_rest_travel": "home_short_rest_travel"})

    away_f = log.merge(games[["game_id", "away_franchise"]], on="game_id")
    away_f = away_f[away_f["team"] == away_f["away_franchise"]][
        ["game_id", "just_traveled", "short_rest_travel"]
    ].rename(columns={"just_traveled": "away_just_traveled", "short_rest_travel": "away_short_rest_travel"})

    out = games.merge(home_f, on="game_id", how="left").merge(away_f, on="game_id", how="left")
    for c in ["home_just_traveled", "home_short_rest_travel", "away_just_traveled", "away_short_rest_travel"]:
        out[c] = out[c].fillna(0).astype(int)
    return out


# ---------------------------------------------------------------------------
# Shared pipeline runner, identical shape to verify.py / other MLB research scripts
# ---------------------------------------------------------------------------
def run_pipeline(feature_cols, feats):
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    margin_corr = corr("predicted_margin", "actual_margin")
    total_corr = corr("predicted_total", "actual_total")

    stds = ensemble.compute_residual_stds(oos, mlb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, mlb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)

    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "n_bets": 0}, oos

    summary = backtest.summarize(bets)
    row = summary.iloc[0]
    return {
        "margin_corr": margin_corr, "total_corr": total_corr,
        "roi_pct": float(row["roi_pct"]), "roi_stderr_pct": float(row["roi_stderr_pct"]),
        "n_bets": int(row["bets"]),
    }, oos


def main():
    hr("LOAD + ODDS + STARTER QUALITY + POWER RATINGS + FEATURES (current production pipeline)")
    games = load_games()
    games = attach_odds(games)
    games = attach_starter_quality(games)  # already-adopted feature, part of the live baseline

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
    feats = build_features(games, rr.history)
    feats = attach_travel_fatigue(feats)
    print(f"Feature rows: {len(feats):,}")

    hr("EXPLORATION: HOW OFTEN DOES THE FLAG FIRE, AND WHAT DOES A RAW SPLIT LOOK LIKE?")
    print(feats[["home_just_traveled", "home_short_rest_travel",
                  "away_just_traveled", "away_short_rest_travel"]].mean().rename("rate").to_string())
    print(
        "\n'just_traveled' (park changed since the team's last game, regardless of rest) fires on "
        "roughly half of away-team rows, as expected -- most series openers ARE a park change for "
        "the visiting team. 'short_rest_travel' (park changed AND <=1 day rest) is the actual "
        "hypothesis under test -- a much rarer, more specific condition: a genuine short-notice "
        "travel day, not just 'first game of a new series with normal days off in between.'"
    )

    # Diagnostic: does actual margin differ, on average, for the travel-fatigued side, before
    # any model is involved at all -- same "look at the raw signal first" step CFB's
    # research_week_trends.py used before formalizing a hypothesis.
    home_fatigued = feats[feats["home_short_rest_travel"] == 1]
    away_fatigued = feats[feats["away_short_rest_travel"] == 1]
    neither = feats[(feats["home_short_rest_travel"] == 0) & (feats["away_short_rest_travel"] == 0)]
    print(f"\nn home-team-fatigued rows: {len(home_fatigued):,}  mean actual_margin (home persp.): "
          f"{home_fatigued['actual_margin'].mean():+.3f}")
    print(f"n away-team-fatigued rows: {len(away_fatigued):,}  mean actual_margin (home persp.): "
          f"{away_fatigued['actual_margin'].mean():+.3f}")
    print(f"n neither-fatigued rows:   {len(neither):,}  mean actual_margin (home persp.): "
          f"{neither['actual_margin'].mean():+.3f}")
    print(
        "(If travel fatigue is real and this proxy captures it: home-team-fatigued rows should skew "
        "toward a lower/negative home margin relative to baseline, away-team-fatigued rows should "
        "skew toward a higher/positive home margin relative to baseline.)"
    )

    hr("BASELINE: current live ML_FEATURE_COLS")
    baseline, _ = run_pipeline(ML_FEATURE_COLS, feats)
    print(baseline)

    hr("VARIANT: + travel_fatigue_diff (away_short_rest_travel - home_short_rest_travel)")
    feats["travel_fatigue_diff"] = feats["away_short_rest_travel"] - feats["home_short_rest_travel"]
    variant_cols = ML_FEATURE_COLS + ["travel_fatigue_diff"]
    variant, _ = run_pipeline(variant_cols, feats)
    print(variant)

    hyp = Hypothesis(
        name="travel_fatigue_short_rest_park_change",
        reasoning=(
            "Baseball teams routinely schedule 'getaway day' games -- typically an early day game "
            "on the last day of a homestand or road series -- specifically so the team can fly "
            "immediately afterward to the next city; broadcasters and beat writers treat short-"
            "notice travel with minimal recovery time as a real, qualitatively distinct fatigue "
            "factor from simple rest-day count. The model's existing `rest_days`/`rest_diff` "
            "features measure TIME since a team's last game but are blind to whether that time "
            "involved changing cities at all -- a team with 1 day of rest after a cross-country "
            "flight and a team with 1 day of rest after an off-day at home get an identical "
            "rest_days value today, even though the two situations are not equivalent. This is "
            "also a different signal from both already-adopted MLB features: starting_pitcher_"
            "rolling_er_per_start reflects the starter's own recent run prevention, and "
            "bullpen_fatigue_relief_appearances_l3d reflects the relief corps' recent workload -- "
            "neither touches whether the TEAM AS A WHOLE just traveled. Built entirely from "
            "loader.py's existing park_id/date columns (no new data source): a team's most recent "
            "game being at a DIFFERENT park than today's, combined with <=1 day of rest since that "
            "game, is a walk-forward-safe (backward-looking only, same shift(1) discipline as every "
            "other rolling feature in this project) proxy for genuine short-notice travel fatigue."
        ),
        sport="MLB",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT: travel_fatigue_short_rest_park_change")
    import json
    print(json.dumps(result.to_dict(), indent=2, default=str))

    hr("DONE")


if __name__ == "__main__":
    main()
