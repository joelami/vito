"""
THROWAWAY research script -- NOT wired into main.py/pipeline.py/harness.py.
Tests one hypothesis (in two candidate feature representations, per this
project's "genuinely torn -> test both once" allowance) via core/research.py's
disciplined evaluate_hypothesis() loop, following the exact shape of
sports/nfl/research_travel_distance.py / sports/nba/research_star_venue_split.py:
loader -> power ratings -> baseline features vs variant features -> walk-
forward ML -> residual stds -> backtest -> evaluate_hypothesis().

Hypothesis: NBA_SCHEDULE_DENSITY_FATIGUE
----------------------------------------
NBA analytics widely documents that CUMULATIVE schedule density -- not just
"played yesterday" -- predicts real performance decline. features.py already
has home_b2b/away_b2b (rest_days <= 1: a genuine, already-adopted, already-in-
ML_FEATURE_COLS signal for "played literally yesterday"). This hypothesis
tests something DISTINCT from that: a team can have a full day of rest before
TONIGHT's specific game (rest_days == 2, home_b2b/away_b2b == 0) while still
being in the middle of a brutal stretch -- e.g. this is their 4th game in 6
nights. That is a real, externally-documented fatigue driver (the NBA's own
injury-and-illness reports and multiple public analytics writeups treat
"3-in-4" and "4-in-6" as named, tracked situations distinct from a simple
back-to-back flag) that nothing in ML_FEATURE_COLS currently captures --
confirmed directly by grepping sports/nba/ for "density"/"games_in"/"3.*4"/
"4.*6" before writing a line of this script: zero matches outside this file.

Built from sports/nba/features.py's own `_team_game_log` (imported, not
reimplemented) -- the same per-team chronological log home_b2b/away_b2b and
every rolling-form column already use -- with a rolling WINDOW COUNT (how
many of a team's own prior games fall in the last N calendar days) in place
of a rolling mean. Walk-forward-safe by construction: for game i, only that
team's own STRICTLY PRIOR games (index < i in that team's sorted date order)
are counted, exactly mirroring the shift(1)-before-rolling discipline every
other feature in features.py uses -- there is no shift(1) to add here because
the counting window itself is defined as "strictly before this game's date,"
so game i can never count itself.

Two candidate representations, per the task's explicit "if genuinely torn,
test both once" allowance (not an unlimited search) -- genuinely unclear in
advance whether a continuous count or the exact named "3-in-4"/"4-in-6"
boolean thresholds used in real NBA reporting would fit better, so both are
tested ONCE each, independently evaluated, with no further iteration on
either based on which looked better:

  A) RAW COUNT: home/away_games_in_prev_4_days, home/away_games_in_prev_6_days,
     plus _diff versions (home minus away), following this project's existing
     naming convention (rest_diff, pyth_pct_diff, etc.).
  B) BOOLEAN THRESHOLD: home/away_3_in_4 (games_in_prev_4_days >= 2, i.e. this
     game would be the team's 3rd in 4 nights), home/away_4_in_6
     (games_in_prev_6_days >= 3, i.e. this game would be the 4th in 6 nights),
     plus _diff versions.

Run with:  python -m sports.nba.research_schedule_density   (from backend/)

=============================================================================
OUTCOME (2026-09-03, logged in decision_log.jsonl) -- REAL numbers, exactly
as printed by this script's own run, no rounding in either direction's favor:
=============================================================================
Baseline reproduced the documented current NBA numbers exactly before either
variant was run: margin_corr 0.405157, total_corr 0.632622, 22,354 qualifying
bets, ROI -2.702% +-0.861pp -- bit-for-bit the same baseline
research_star_venue_split.py's independent re-verification recorded.

Sanity check on the new columns before any modeling: of games flagged
home_3_in_4==1 (this game is the team's 3rd in 4 nights), only 26.7% are
ALSO flagged home_b2b==1 -- confirming up front that this is measuring a
genuinely different (though related) situation than the existing
back-to-back flag, not a relabeling of it.

CANDIDATE A (raw counts -- games_in_prev_4_days/6_days + diffs): margin_corr
0.405157->0.405553 (+0.00040, still inside the 0.005 noise floor), total_corr
0.632622->0.632516 (-0.00011, likewise inside the floor), ROI -2.702%->
-2.212% (+0.49pp, inside its own +-0.86pp stderr -- not the "ROI moved but
fit didn't" suspicious pattern, since fit didn't degrade and the move stayed
within the stated noise band). evaluate_hypothesis(): "adopt_cautiously" --
a clean, harmless null.

CANDIDATE B (boolean 3-in-4/4-in-6 thresholds + diffs): margin_corr
0.405157->0.405142 (-0.00001, far inside the noise floor), total_corr
0.632622->0.632838 (+0.00022, likewise), ROI -2.702%->-2.792% (-0.09pp,
trivially within its own +-0.86pp stderr). evaluate_hypothesis():
"adopt_cautiously" -- also a clean, harmless null.

Both candidates land in the same place: real, external motivation: yes;
statistically distinguishable improvement: no (both margin/total deltas are
well below the 0.005 noise floor on both axes); ROI: moved a bit in each
direction but stayed inside its own standard error both times, and never in
the "ROI up with no fit basis" shape this project's suspicious-pattern check
exists to catch. This is a genuine, honest null -- schedule DENSITY beyond a
simple back-to-back flag does not measurably improve this model's grip on
NBA margin or total, at least not in a form the ML leg can extract on top of
rating_diff_pre, rest_days, home_b2b/away_b2b, and the rolling form columns
already in ML_FEATURE_COLS. A plausible reason, stated honestly rather than
explained away: even though the sanity check above shows the new flags are
NOT simply relabeling home_b2b (only 26.7% overlap), rest_days/home_b2b and
the rolling form columns (win_pct_l10, pf_l10/pa_l10, which already reflect
a team playing worse during a dense stretch, whatever the specific cause)
may already be soaking up most of the same downstream effect a schedule-
density count would otherwise explain -- so the incremental information
left over, once those are already in the model, is small.

NOT ADOPTED into ML_FEATURE_COLS for either candidate -- per this project's
standing practice, an adopt_cautiously label is a floor, not a mandate. The
closest precedent isn't the expensive-external-join cases (NBA star-venue-
split's 118MB file, NHL player-matchup's 2.6GB file) -- this feature is
CHEAP, built entirely from `_team_game_log`'s columns that are already loaded
for home_b2b/rest_days/rolling form. The closer precedent is NHL's trailing
save_pct_l10: also cheap, also a real, well-motivated, non-suspicious clean
null, and also correctly left out of production for the same reason --
"doesn't hurt and was cheap to test" is not the same bar as "measurably
helps," and a feature that demonstrably moves nothing on the fit axis isn't
worth two new permanent engineered columns and an extra per-team date-window
computation on every pipeline build, even at low cost, for zero offsetting
benefit. sports/nba/features.py, config.py, loader.py, and core/matchup.py
are all left exactly as found. This script and the two decision_log.jsonl
entries (`Hypothesis test: nba_schedule_density_fatigue_rawcount`,
`Hypothesis test: nba_schedule_density_fatigue_boolean`) are the permanent
record.
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
from sports.nba import config as nba_config
from sports.nba.loader import load_games
from sports.nba.features import build_features, ML_FEATURE_COLS, _team_game_log


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Core windowed-count computation. Given one team's own dates (already sorted
# ascending, as _team_game_log guarantees), returns, for each game, the count
# of that SAME team's own games whose date falls strictly within the
# `window_days` calendar days immediately before that game's date (never
# including the game itself). Vectorized via searchsorted rather than a
# per-row Python loop, exploiting that the input is already sorted: for game
# i at day-ordinal d[i], every prior game j<i already has d[j] < d[i]
# (dates are strictly increasing per team -- an NBA team never plays twice on
# the same calendar date), so "how many prior games fall within window_days"
# reduces to "how many of d[0..i-1] are >= d[i]-window_days", found via one
# searchsorted call per team rather than per row.
# ---------------------------------------------------------------------------
def _games_in_window(dates: pd.Series, window_days: int) -> pd.Series:
    d = dates.values.astype("datetime64[D]").astype(np.int64)
    idx = np.arange(len(d))
    pos = np.searchsorted(d, d - window_days, side="left")
    return pd.Series(idx - pos, index=dates.index)


def attach_schedule_density(games: pd.DataFrame) -> pd.DataFrame:
    """
    Returns `games` with 8 new walk-forward-safe columns attached:
    home/away_games_in_prev_4_days, home/away_games_in_prev_6_days (raw
    counts, Candidate A), home/away_3_in_4, home/away_4_in_6 (boolean
    thresholds, Candidate B), plus the four corresponding _diff columns.
    Built on top of features.py's own `_team_game_log` -- the identical
    per-team chronological log home_b2b/away_b2b and every rolling-form
    column already use -- so this is the same base data, not a
    reimplementation or a second source of truth.
    """
    log = _team_game_log(games).sort_values(["team", "date"], kind="stable")
    grp = log.groupby("team", group_keys=False)
    log["g4"] = grp["date"].apply(lambda s: _games_in_window(s, 4))
    log["g6"] = grp["date"].apply(lambda s: _games_in_window(s, 6))

    home = log[log["is_home"]][["game_id", "g4", "g6"]].rename(
        columns={"g4": "home_games_in_prev_4_days", "g6": "home_games_in_prev_6_days"}
    )
    away = log[~log["is_home"]][["game_id", "g4", "g6"]].rename(
        columns={"g4": "away_games_in_prev_4_days", "g6": "away_games_in_prev_6_days"}
    )
    out = games.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")

    # a team's very first game(s) in the data have no real window history yet
    # -- 0 prior games is the honest, non-fabricated value here (unlike
    # rest_days' 3-day neutral fill, there is no "neutral" count; 0 is simply
    # correct for a team with no games yet in that window).
    count_cols = ["home_games_in_prev_4_days", "away_games_in_prev_4_days",
                  "home_games_in_prev_6_days", "away_games_in_prev_6_days"]
    for c in count_cols:
        out[c] = out[c].fillna(0).astype(int)

    out["games_in_prev_4_days_diff"] = out["home_games_in_prev_4_days"] - out["away_games_in_prev_4_days"]
    out["games_in_prev_6_days_diff"] = out["home_games_in_prev_6_days"] - out["away_games_in_prev_6_days"]

    # Boolean thresholds named after the exact situations real NBA reporting
    # uses: "games_in_prev_4_days >= 2" means this game is the team's 3rd
    # game within a 4-night window (today + 2 prior = 3rd); "games_in_prev_6_
    # days >= 3" means this is the team's 4th game within a 6-night window.
    out["home_3_in_4"] = (out["home_games_in_prev_4_days"] >= 2).astype(int)
    out["away_3_in_4"] = (out["away_games_in_prev_4_days"] >= 2).astype(int)
    out["home_4_in_6"] = (out["home_games_in_prev_6_days"] >= 3).astype(int)
    out["away_4_in_6"] = (out["away_games_in_prev_6_days"] >= 3).astype(int)
    out["threein4_diff"] = out["home_3_in_4"] - out["away_3_in_4"]
    out["fourin6_diff"] = out["home_4_in_6"] - out["away_4_in_6"]

    return out


RAWCOUNT_COLS = [
    "home_games_in_prev_4_days", "away_games_in_prev_4_days",
    "home_games_in_prev_6_days", "away_games_in_prev_6_days",
    "games_in_prev_4_days_diff", "games_in_prev_6_days_diff",
]
BOOLEAN_COLS = [
    "home_3_in_4", "away_3_in_4", "home_4_in_6", "away_4_in_6",
    "threein4_diff", "fourin6_diff",
]


def run_pipeline(feats: pd.DataFrame, feature_cols: list, label: str) -> dict:
    hr(f"WALK-FORWARD ML -- {label}")
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    print(f"Seasons predicted out-of-sample: {wf.seasons_predicted}")
    print(f"OOS rows: {len(wf.predictions):,}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    margin_corr = corr("predicted_margin", "actual_margin")
    total_corr = corr("predicted_total", "actual_total")
    print(f"Margin: ML pred vs actual correlation: {margin_corr:.6f}")
    print(f"Total:  ML pred vs actual correlation: {total_corr:.6f}")

    stds = ensemble.compute_residual_stds(oos, nba_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, nba_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    if bets.empty:
        raise RuntimeError(f"{label}: no bets cleared thresholds -- cannot compute ROI.")
    s = backtest.summarize(bets)
    roi_pct = float(s["roi_pct"].iloc[0])
    roi_stderr_pct = float(s["roi_stderr_pct"].iloc[0])
    n_bets = int(s["bets"].iloc[0])
    print(f"Qualifying bets: {n_bets:,}, ROI {roi_pct:+.3f}% +-{roi_stderr_pct:.3f}pp")

    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": roi_pct, "roi_stderr_pct": roi_stderr_pct, "n_bets": n_bets}


def main():
    hr("0. LOAD + POWER RATINGS (shared by baseline and both variants)")
    games = load_games()
    print(f"Rows after load/clean: {len(games):,}")
    print(f"Season range: {games['season'].min()} - {games['season'].max()}")

    rating_cfg = PowerRatingConfig(
        k_factor=nba_config.ELO_K_FACTOR, start_rating=nba_config.ELO_START_RATING,
        home_field_adv=nba_config.HOME_FIELD_ADV_ELO, season_regression=nba_config.SEASON_REGRESSION,
        mov_mult_base=nba_config.MOV_MULT_BASE, mov_mult_divisor=nba_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", config=rating_cfg,
    )

    hr("1. ATTACH SCHEDULE-DENSITY COLUMNS (raw counts + boolean thresholds)")
    games_sd = attach_schedule_density(games)
    for c in RAWCOUNT_COLS[:4]:
        print(f"{c}: value_counts\n{games_sd[c].value_counts().sort_index().head(10)}\n")
    print(f"3-in-4 rate: home {games_sd['home_3_in_4'].mean()*100:.1f}%, away {games_sd['away_3_in_4'].mean()*100:.1f}%")
    print(f"4-in-6 rate: home {games_sd['home_4_in_6'].mean()*100:.1f}%, away {games_sd['away_4_in_6'].mean()*100:.1f}%")
    # sanity: a 3-in-4 game should virtually always also be flagged b2b OR the
    # prior game was a b2b -- not required to be identical to home_b2b (that
    # would mean this feature adds nothing structurally), just correlated.
    tmp_feats_for_sanity = build_features(games_sd, rr.history)
    overlap = (tmp_feats_for_sanity["home_3_in_4"] == 1) & (tmp_feats_for_sanity["home_b2b"] == 1)
    print(f"Of home_3_in_4==1 games, home_b2b==1 also true for "
          f"{overlap.sum()}/{(tmp_feats_for_sanity['home_3_in_4']==1).sum()} "
          f"({overlap.mean()/max((tmp_feats_for_sanity['home_3_in_4']==1).mean(),1e-9)*100:.1f}% of them)"
          if (tmp_feats_for_sanity['home_3_in_4']==1).sum() else "no home_3_in_4==1 games found")

    # ------------------------------------------------------------------
    # 2. BASELINE features (features.py unmodified) -- verify reproduction
    #    of the documented current NBA numbers BEFORE trusting any variant.
    # ------------------------------------------------------------------
    hr("2. BASELINE FEATURES (features.py unmodified)")
    baseline_feats = build_features(games_sd, rr.history)  # extra density cols ride along unused
    print(f"Feature rows: {len(baseline_feats):,}, ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")
    baseline = run_pipeline(baseline_feats, ML_FEATURE_COLS, "BASELINE")
    print(f"\n[Baseline reproduction check] margin_corr={baseline['margin_corr']:.6f} "
          f"(expect ~0.405157), total_corr={baseline['total_corr']:.6f} (expect ~0.632622), "
          f"n_bets={baseline['n_bets']} (expect 22354), ROI={baseline['roi_pct']:+.3f}% "
          f"(expect ~-2.702%). If these don't match, STOP -- do not trust any variant below.")

    # ------------------------------------------------------------------
    # 3. CANDIDATE A: raw counts
    # ------------------------------------------------------------------
    hr("3. CANDIDATE A: RAW WINDOWED COUNTS")
    variant_a_cols = ML_FEATURE_COLS + RAWCOUNT_COLS
    variant_a = run_pipeline(baseline_feats, variant_a_cols, "CANDIDATE A (raw counts)")

    hypothesis_a = Hypothesis(
        name="nba_schedule_density_fatigue_rawcount",
        reasoning=(
            "NBA analytics widely documents that CUMULATIVE schedule density -- not just "
            "'played yesterday' (already captured by the adopted home_b2b/away_b2b flag) -- "
            "predicts real performance decline: a team playing its 3rd game in 4 nights or "
            "4th game in 6 nights carries real fatigue even on a night where it technically "
            "had a day of rest before THIS specific game. This is a distinct signal from a "
            "simple back-to-back flag by construction: a team can show rest_days==2 and "
            "home_b2b==0 for tonight's game while still being in the middle of a dense recent "
            "stretch. Confirmed genuinely untested by grepping sports/nba/ for "
            "'density'/'games_in'/'3.*4'/'4.*6' before writing any code -- no existing feature "
            "computes a rolling window COUNT anywhere in this codebase, only rolling MEANS "
            "(ats_pct_l10, win_pct_l10, pf_l10, pa_l10) and the single-day b2b flag. Built as "
            "raw counts (games_in_prev_4_days, games_in_prev_6_days per team, plus _diff "
            "versions) as the more information-preserving of two candidate representations, "
            "tested here as Candidate A alongside a boolean-threshold Candidate B per this "
            "project's 'genuinely torn -> test both once' allowance."
        ),
        sport="NBA",
    )
    result_a = evaluate_hypothesis(hypothesis_a, {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")},
                                    {k: variant_a[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")})
    import json
    print("\nCandidate A result.to_dict():")
    print(json.dumps(result_a.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # 4. CANDIDATE B: boolean 3-in-4 / 4-in-6 thresholds
    # ------------------------------------------------------------------
    hr("4. CANDIDATE B: BOOLEAN 3-IN-4 / 4-IN-6 THRESHOLDS")
    variant_b_cols = ML_FEATURE_COLS + BOOLEAN_COLS
    variant_b = run_pipeline(baseline_feats, variant_b_cols, "CANDIDATE B (boolean thresholds)")

    hypothesis_b = Hypothesis(
        name="nba_schedule_density_fatigue_boolean",
        reasoning=(
            "Same externally-motivated schedule-density-fatigue reasoning as Candidate A "
            "(nba_schedule_density_fatigue_rawcount, evaluated moments before this in the same "
            "script run) -- tested here in the alternative representation used by real NBA "
            "reporting: named boolean thresholds ('3rd game in 4 nights', 'games_in_prev_4_days "
            ">= 2'; '4th game in 6 nights', 'games_in_prev_6_days >= 3') rather than a raw "
            "count, on the theory that a tree-based model might split more cleanly on the exact "
            "situation analysts and beat writers actually name than on a raw count whose "
            "marginal effect per additional game may not be linear. Tested once, independently "
            "evaluated, with no further iteration on either candidate based on which looks "
            "better -- per this project's explicit 'genuinely torn -> test both once, not an "
            "unlimited search' allowance."
        ),
        sport="NBA",
    )
    result_b = evaluate_hypothesis(hypothesis_b, {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")},
                                    {k: variant_b[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")})
    print("\nCandidate B result.to_dict():")
    print(json.dumps(result_b.to_dict(), indent=2))

    hr("DONE")
    print(f"Candidate A recommendation: {result_a.recommendation}")
    print(f"Candidate B recommendation: {result_b.recommendation}")


if __name__ == "__main__":
    main()
