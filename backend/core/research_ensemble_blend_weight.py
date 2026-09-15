"""
THROWAWAY research script -- re-tests core/ensemble.py's EnsembleConfig
blend weights (weight_elo_moneyline/spread, weight_ml_total), all flat
0.5 since this project's earliest sessions, now that NFL and CFB have
real play-level trailing features (EPA/success rate, adopted 2026-09-14)
the ML side of the blend didn't have when 0.5 was first chosen. Direct
answer to the app owner's "ground-up model design" question from this
same session: "is the 50/50 blend still right now that ML has richer
inputs" is a real, testable question, not a guess.

METHODOLOGY, and why this is NOT a bare ROI grid-search: sweeping a
weight to maximize backtest ROI directly is exactly the "threshold
tweaking" overfitting pattern this project's own framework already
guards against (see decision_log.jsonl's "fake_threshold_tweak" -- a
deliberately-constructed case where ROI improved with literally zero
change in the model's own predictions, flagged suspicious and rejected).
A blend weight changes ONLY the final probability, never the underlying
margin_corr/total_corr fit metrics evaluate_hypothesis() normally checks
-- so that guardrail can't be applied here directly, and something else
has to stand in for it.

BRIER SCORE is that something else: mean squared error between the
blended probability and the real binary outcome (home win / home covers
/ over hits), computed across the FULL out-of-sample population, not a
bet-selection-filtered subset -- a standard forecasting-calibration
metric with no selection step to overfit, the direct probability analog
of margin_corr/total_corr. A weight is only recommended if it improves
Brier score AND that improvement holds up independently on BOTH halves
of a season-based split -- the same split-half stability bar every other
finding in decision_log.jsonl is held to -- specifically so a weight
that merely fits noise in the full sample gets caught before being
trusted.

Run: python -m core.research_ensemble_blend_weight <SPORT> (e.g. NFL, CFB)
"""

import sys

import numpy as np
import pandas as pd

from core import ensemble


CANDIDATE_WEIGHTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def _season_halves(oos: pd.DataFrame):
    seasons = sorted(oos["season"].unique())
    mid = seasons[len(seasons) // 2]
    return oos[oos["season"] < mid], oos[oos["season"] >= mid]


def sweep_moneyline(oos: pd.DataFrame, stds, elo_points_per_margin: float) -> pd.DataFrame:
    outcomes = (oos["actual_margin"] > 0).astype(float).values
    rows = []
    for w in CANDIDATE_WEIGHTS:
        cfg = ensemble.EnsembleConfig(weight_elo_moneyline=w)
        probs = np.array([ensemble.moneyline_prob(r, stds, elo_points_per_margin, cfg)["blended_prob"]
                           for _, r in oos.iterrows()])
        rows.append({"weight_elo": w, "brier": brier_score(probs, outcomes), "n": len(oos)})
    return pd.DataFrame(rows)


def sweep_spread(oos: pd.DataFrame, stds, elo_points_per_margin: float) -> pd.DataFrame:
    sub = oos.dropna(subset=["Home Line Close"])
    outcomes = (sub["actual_margin"] > sub["Home Line Close"]).astype(float).values
    rows = []
    for w in CANDIDATE_WEIGHTS:
        cfg = ensemble.EnsembleConfig(weight_elo_spread=w)
        probs = np.array([ensemble.spread_cover_prob(r, r["Home Line Close"], stds, elo_points_per_margin, cfg)["blended_prob"]
                           for _, r in sub.iterrows()])
        rows.append({"weight_elo": w, "brier": brier_score(probs, outcomes), "n": len(sub)})
    return pd.DataFrame(rows)


def sweep_total(oos: pd.DataFrame, stds) -> pd.DataFrame:
    sub = oos.dropna(subset=["Total Score Close"])
    outcomes = (sub["actual_total"] > sub["Total Score Close"]).astype(float).values
    rows = []
    for w in CANDIDATE_WEIGHTS:
        cfg = ensemble.EnsembleConfig(weight_ml_total=w)
        probs = np.array([ensemble.total_over_prob(r, r["Total Score Close"], stds, cfg)["blended_prob"]
                           for _, r in sub.iterrows()])
        rows.append({"weight_ml": w, "brier": brier_score(probs, outcomes), "n": len(sub)})
    return pd.DataFrame(rows)


def evaluate_market(name: str, sweep_fn, weight_col: str, current_weight: float = 0.5):
    hr(f"{name}: full-sample sweep")
    full = sweep_fn()
    print(full.to_string(index=False))
    best_row = full.loc[full["brier"].idxmin()]
    best_weight = float(best_row[weight_col])
    current_brier = float(full.loc[full[weight_col] == current_weight, "brier"].iloc[0])
    improvement = current_brier - float(best_row["brier"])
    print(f"\ncurrent weight={current_weight} brier={current_brier:.5f} | "
          f"best weight={best_weight} brier={best_row['brier']:.5f} | improvement={improvement:.5f}")
    return best_weight, improvement


def main():
    sport = sys.argv[1].upper() if len(sys.argv) > 1 else "NFL"
    hr(f"BUILDING {sport} PIPELINE")
    if sport == "NFL":
        import pipeline
        p = pipeline.build_nfl_pipeline(persist_backtest=False)
    else:
        import pipeline
        p = pipeline.build_pipeline(sport.lower(), persist_backtest=False)

    oos = p["oos_df"]
    stds = p["stds"]
    import importlib
    elo_points_per_margin = importlib.import_module(f"sports.{sport.lower()}.config").ELO_POINTS_PER_MARGIN
    print(f"OOS rows: {len(oos):,}, seasons: {sorted(oos['season'].unique())}")

    first_half, second_half = _season_halves(oos)
    print(f"split-half sizes: first={len(first_half)}, second={len(second_half)}")

    for market, weight_col, sweep_fn, current in [
        ("MONEYLINE", "weight_elo", lambda o=oos: sweep_moneyline(o, stds, elo_points_per_margin), 0.5),
        ("SPREAD", "weight_elo", lambda o=oos: sweep_spread(o, stds, elo_points_per_margin), 0.5),
        ("TOTAL", "weight_ml", lambda o=oos: sweep_total(o, stds), 0.5),
    ]:
        best_weight, improvement = evaluate_market(market, sweep_fn, weight_col, current)

        # Stability check: does the SAME weight also improve Brier score
        # independently on both halves? A weight that only helps in the
        # full-sample aggregate (but not both halves) is fit to this one
        # historical sample's noise, not a generalizable finding -- same
        # split-half bar every other decision_log.jsonl finding is held to.
        if market == "MONEYLINE":
            fh, sh = sweep_moneyline(first_half, stds, elo_points_per_margin), sweep_moneyline(second_half, stds, elo_points_per_margin)
        elif market == "SPREAD":
            fh, sh = sweep_spread(first_half, stds, elo_points_per_margin), sweep_spread(second_half, stds, elo_points_per_margin)
        else:
            fh, sh = sweep_total(first_half, stds), sweep_total(second_half, stds)

        wc = "weight_elo" if market != "TOTAL" else "weight_ml"
        fh_best_brier = float(fh.loc[fh[wc] == best_weight, "brier"].iloc[0])
        fh_cur_brier = float(fh.loc[fh[wc] == current, "brier"].iloc[0])
        sh_best_brier = float(sh.loc[sh[wc] == best_weight, "brier"].iloc[0])
        sh_cur_brier = float(sh.loc[sh[wc] == current, "brier"].iloc[0])
        stable = (fh_best_brier <= fh_cur_brier) and (sh_best_brier <= sh_cur_brier)
        print(f"{market} split-half check: best_weight={best_weight} vs current={current} -- "
              f"first-half brier {fh_best_brier:.5f} (best) vs {fh_cur_brier:.5f} (current), "
              f"second-half brier {sh_best_brier:.5f} (best) vs {sh_cur_brier:.5f} (current) "
              f"-- {'STABLE, recommend adopting' if stable and improvement > 0.0002 else 'NOT adopted (unstable or negligible)'}")

    hr("DONE")


if __name__ == "__main__":
    main()
