"""
RESEARCH SCRIPT (throwaway, not part of the production module shape) —
answers the app owner's two-part question ("maybe rankings don't matter,
what about week-over-week trends?") for CFB specifically. See
core/research.py for the evaluation discipline this follows exactly, and
docs/METHODOLOGY.md's NCAAF section for the baseline this compares against.

PART A — AP/CFP ranking data. Checked directly against the raw CSV's own
column list (see STEP 0 below): `Datasets/College Football/cfb-games.csv`
has NO ranking column of any kind (no AP poll rank, no CFP rank, no coaches'
poll, nothing rank-shaped) anywhere in its 30 columns, and neither do the
three box-score files sitting alongside it. Per the task's explicit
instruction, no external ranking dataset was substituted or fabricated as a
proxy. FINDING: this half of the task cannot be tested on this data at all —
not "inconclusive," genuinely "the input doesn't exist." Nothing to log via
evaluate_hypothesis(), since there's no baseline-vs-variant comparison to
run without inventing data.

PART B — week-of-season / schedule trends. A real, testable question: the
raw `week` column IS in the data (games CSV, kept by loader.py). STEP 1
below diagnoses the CURRENT model's own out-of-sample fit split by phase of
season, using the pipeline that's already in production (ML_FEATURE_COLS as
it stands today, both prior CFB hypothesis-test adoptions included). STEP 2
turns the real pattern found there into one well-motivated hypothesis and
runs it through evaluate_hypothesis(), same standard as every other CFB
research script.

Run with:  python -m sports.cfb.research_week_trends   (from backend/)
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
from sports.cfb import config as cfb_config
from sports.cfb.loader import load_games
from sports.cfb.features import build_features, ML_FEATURE_COLS, season_week_adjusted


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# PART A: does a ranking column exist anywhere in this dataset at all?
# ---------------------------------------------------------------------------
def check_for_ranking_data():
    hr("PART A: DOES THIS DATASET CONTAIN AP/CFP RANKING DATA?")
    raw = pd.read_csv(cfb_config.DATA_PATH)
    cols = list(raw.columns)
    print(f"cfb-games.csv columns ({len(cols)}): {cols}")
    rank_like = [c for c in cols if any(k in c.lower() for k in ["rank", "ap_", "cfp", "poll", "seed"])]
    print(f"\nColumns matching rank/poll/seed-shaped names: {rank_like or 'NONE'}")

    ds_dir = cfb_config.DATA_PATH.parent
    for f in sorted(ds_dir.iterdir()):
        if f.suffix != ".csv":
            continue
        try:
            fcols = list(pd.read_csv(f, nrows=1).columns)
        except Exception as e:
            print(f"{f.name}: could not read ({e})")
            continue
        f_rank_like = [c for c in fcols if any(k in c.lower() for k in ["rank", "ap_", "cfp", "poll", "seed"])]
        print(f"{f.name}: {len(fcols)} columns, rank-like: {f_rank_like or 'NONE'}")

    print(
        "\nCONCLUSION: no AP poll rank, CFP rank, coaches' poll, or seed column exists anywhere "
        "in Datasets/College Football/ (games file or any of the three box-score files). Per "
        "the task's explicit instruction, no external ranking source was fetched or fabricated "
        "as a proxy. Hypothesis A (is_ranked / rank_diff vs Elo) is NOT TESTABLE on this data — "
        "reported as a plain fact, not run through evaluate_hypothesis() (there is no variant "
        "feature to build, so there is nothing to compare against baseline)."
    )


# ---------------------------------------------------------------------------
# Shared pipeline runner - identical shape to verify.py / other CFB research scripts
# ---------------------------------------------------------------------------
def run_pipeline(feature_cols: list, feats: pd.DataFrame, label: str) -> dict:
    hr(f"PIPELINE RUN: {label}")
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    print(f"OOS rows: {len(wf.predictions):,}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    margin_corr = corr("predicted_margin", "actual_margin")
    total_corr = corr("predicted_total", "actual_total")
    print(f"Margin corr: {margin_corr:.4f}   Total corr: {total_corr:.4f}")

    stds = ensemble.compute_residual_stds(oos, cfb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, cfb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"Qualifying bets: {len(bets):,}")

    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr, "roi_pct": 0.0,
                "roi_stderr_pct": float("nan")}, oos

    summary = backtest.summarize(bets)
    roi_pct = float(summary["roi_pct"].iloc[0])
    roi_stderr_pct = float(summary["roi_stderr_pct"].iloc[0])
    print(f"ROI: {roi_pct:+.2f}%  (stderr {roi_stderr_pct:.2f}pp, n={len(bets)})")

    return {"margin_corr": margin_corr, "total_corr": total_corr, "roi_pct": roi_pct,
            "roi_stderr_pct": roi_stderr_pct}, oos


# ---------------------------------------------------------------------------
# PART B, STEP 1: diagnostic - split the CURRENT model's own OOS predictions
# by phase of season, using the real `week` column already in the data.
# ---------------------------------------------------------------------------
def diagnose_by_week(oos: pd.DataFrame):
    hr("PART B / STEP 1: DIAGNOSTIC - FIT BY PHASE OF SEASON (current production model)")

    def corr(a, b, sub):
        if len(sub) < 5:
            return float("nan")
        return float(np.corrcoef(sub[a], sub[b])[0, 1])

    non_bowl = oos[~oos["is_bowl"]]
    print(f"{'phase':<20}{'n':>8}{'margin_corr':>14}{'total_corr':>13}")
    for lo, hi, label in [(1, 3, "wk1-3 (early)"), (4, 9, "wk4-9 (mid)"), (10, 16, "wk10-16 (late reg.)")]:
        sub = non_bowl[(non_bowl["week"] >= lo) & (non_bowl["week"] <= hi)]
        print(f"{label:<20}{len(sub):>8}{corr('predicted_margin','actual_margin',sub):>14.4f}"
              f"{corr('predicted_total','actual_total',sub):>13.4f}")
    bowl = oos[oos["is_bowl"]]
    print(f"{'bowl/postseason':<20}{len(bowl):>8}{corr('predicted_margin','actual_margin',bowl):>14.4f}"
          f"{corr('predicted_total','actual_total',bowl):>13.4f}")

    print(
        "\nNote on `week` itself: this dataset labels EVERY postseason/bowl game week=1 "
        "(season_type-based reset, not a real in-season week number - verified directly: 599/599 "
        "is_bowl==True rows have week==1). That's why bowls are broken out as their own bucket "
        "above rather than folded into 'wk1-3' where the raw column would wrongly place them."
    )

    print(
        "\nAlso checked and NOT pursued further: a 'games played this season so far' feature "
        "derived by counting each team's rows within this dataset. Rejected as unreliable - this "
        "CSV holds only ~150-200 games per season leaguewide (a real FBS season is closer to "
        "~780 games across ~130 teams), i.e. it's a SPARSE SAMPLE of games, not a complete "
        "schedule capture. A count of 'games seen so far in this dataset' would mostly measure "
        "how sparsely this specific team got sampled, not how far into its real season it "
        "actually is - not a trustworthy signal, so not built into a feature or a hypothesis."
    )

    print(
        "\nAlso checked: fit by rest situation. Games where either team had <7 days rest "
        f"(n={len(oos[(oos['home_rest_days']<7)|(oos['away_rest_days']<7)])}) vs both >=7 days "
        f"(n={len(oos[(oos['home_rest_days']>=7)&(oos['away_rest_days']>=7)])}) - short-rest margin_corr "
        f"{corr('predicted_margin','actual_margin', oos[(oos['home_rest_days']<7)|(oos['away_rest_days']<7)]):.4f} "
        f"vs normal-rest {corr('predicted_margin','actual_margin', oos[(oos['home_rest_days']>=7)&(oos['away_rest_days']>=7)]):.4f}. "
        "Short-rest games in this dataset are almost entirely Tue/Wed/Thu MACtion-style games (n is "
        "small, ~150 of 3,281), and the apparent fit *improvement* there is not a reliable pattern to "
        "chase on this little data (well within the kind of small-sample swing this project's own "
        "noise-floor discipline exists to catch) - not built into its own hypothesis."
    )


# ---------------------------------------------------------------------------
# PART B, STEP 2: the one real, well-motivated hypothesis this diagnosis
# supports - an explicit, calendar-correct "how far into the season is this"
# signal, since `week` itself is unusable as-is (bowls all mislabeled week=1)
# and nothing resembling it was in ML_FEATURE_COLS before this test.
#
# OUTCOME (2026-08-11, logged in decision_log.jsonl): "adopt". margin_corr
# +0.0190 (well above the noise floor), total_corr +0.0010 (essentially
# flat), ROI -3.05% -> -2.38% (+0.67pp, within its own standard error, and
# not the direction that would ever look "suspicious"). This is now
# PERMANENT: `season_week_adjusted()` lives in sports/cfb/features.py,
# `build_features()` computes it as `season_week_adj`, and it's in the live
# ML_FEATURE_COLS - see that module's docstring for the full writeup. This
# script therefore builds its BASELINE with that one column excluded (i.e.
# the pre-adoption feature set) so the comparison below reproduces the real
# before/after that was actually run, rather than comparing production
# against itself.
# ---------------------------------------------------------------------------
PRE_ADOPTION_FEATURE_COLS = [c for c in ML_FEATURE_COLS if c != "season_week_adj"]


def main():
    check_for_ranking_data()

    hr("LOAD + POWER RATINGS + FEATURES (current production pipeline)")
    games = load_games()
    rating_cfg = PowerRatingConfig(
        k_factor=cfb_config.ELO_K_FACTOR, start_rating=cfb_config.ELO_START_RATING,
        home_field_adv=cfb_config.HOME_FIELD_ADV_ELO, season_regression=cfb_config.SEASON_REGRESSION,
        mov_mult_base=cfb_config.MOV_MULT_BASE, mov_mult_divisor=cfb_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", neutral_col="is_neutral_venue",
        config=rating_cfg,
    )
    # build_features() now computes season_week_adj as a permanent column
    # (post-adoption) - see module docstring. Sanity-check it matches this
    # script's own understanding of the transform before using it as ground
    # truth for anything below.
    feats = build_features(games, rr.history)
    print(f"Feature rows: {len(feats):,}")
    assert (feats["season_week_adj"] == season_week_adjusted(feats)).all(), \
        "build_features()'s season_week_adj drifted from features.season_week_adjusted()"

    baseline, oos = run_pipeline(
        PRE_ADOPTION_FEATURE_COLS, feats, "BASELINE (pre-adoption feature set, season_week_adj excluded)"
    )
    diagnose_by_week(oos)

    hr("PART B / STEP 2: HYPOTHESIS - season_week_adj (calendar-correct season-phase signal)")
    variant, _ = run_pipeline(ML_FEATURE_COLS, feats, "VARIANT (= current production ML_FEATURE_COLS, includes season_week_adj)")

    hyp = Hypothesis(
        name="cfb_season_week_adj",
        reasoning=(
            "Diagnostic split of the current production model's own out-of-sample predictions by "
            "phase of season (this script's Part B / Step 1) shows real dispersion in fit that the "
            "model has almost no direct way to see today: margin_corr runs ~0.38 in weeks 1-3, drops "
            "to ~0.27 for weeks 4-16, and falls further to ~0.13 in bowl/postseason games, despite "
            "is_bowl and raw rest_days already being available features - a depth-3 tree still "
            "benefits from an explicit signal rather than reconstructing season-phase indirectly "
            "from several separate schedule columns. The raw `week` column itself is unusable as-is "
            "for this (every postseason game is mislabeled week=1 in this dataset - verified: 599/599 "
            "is_bowl rows have week==1) and nothing resembling a calendar-correct season-progress "
            "signal is in ML_FEATURE_COLS today. season_week_adj fixes the bowl mislabeling by placing "
            "bowls after the real regular season (that season's max regular week + 1) so the signal "
            "reflects the actual football calendar, then tests whether making that explicit helps the "
            "model fit the season-phase pattern already observed in its own residuals."
        ),
        sport="CFB",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT: cfb_season_week_adj")
    import json
    print(json.dumps(result.to_dict(), indent=2, default=str))

    hr("DONE")


if __name__ == "__main__":
    main()
