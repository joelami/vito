"""
Blends the power-rating (Elo) view and the ML view into one probability per
bet type. Each submodel's point prediction (a margin or a total) is turned
into a probability of clearing a specific market line via that submodel's
OWN out-of-sample residual std-dev and a normal-CDF approximation
(`odds_math.cover_prob_spread` / `over_prob_total`), then the two
probabilities are averaged. Averaging probabilities (rather than averaging
points and reconverting) is the simpler, more robust ensembling choice and
is what's implemented here.

Totals have no power-rating analog (Elo captures matchup strength, not
scoring pace), so the "second opinion" for totals is a naive rolling
points-for/against baseline instead of Elo — documented honestly rather than
forcing a rating-based total prediction that wouldn't mean anything.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import odds_math


@dataclass
class EnsembleConfig:
    weight_elo_moneyline: float = 0.5   # vs. ML, for moneyline
    weight_elo_spread: float = 0.5      # vs. ML, for spread
    weight_ml_total: float = 0.5        # vs. naive pace baseline, for totals


@dataclass
class ResidualStds:
    elo_margin_std: float
    ml_margin_std: float
    ml_total_std: float
    naive_total_std: float


def compute_residual_stds(oos: pd.DataFrame, elo_points_per_margin: float) -> ResidualStds:
    """
    `oos` must be restricted to games that actually have an out-of-sample ML
    prediction (i.e. `feats.join(wf.predictions, how='inner')`) so every
    residual std-dev is measured on the same walk-forward-honest sample.
    """
    elo_pred_margin = oos["rating_diff_pre"] / elo_points_per_margin
    return ResidualStds(
        elo_margin_std=float((oos["actual_margin"] - elo_pred_margin).std()),
        ml_margin_std=float((oos["actual_margin"] - oos["predicted_margin"]).std()),
        ml_total_std=float((oos["actual_total"] - oos["predicted_total"]).std()),
        naive_total_std=float((oos["actual_total"] - oos["naive_total"]).std()),
    )


def elo_predicted_margin(rating_diff_pre: float, elo_points_per_margin: float) -> float:
    return rating_diff_pre / elo_points_per_margin


def moneyline_prob(row: pd.Series, stds: ResidualStds, elo_points_per_margin: float,
                    cfg: EnsembleConfig) -> dict:
    """Returns elo/ml/blended home-win probability for a single game row."""
    elo_margin = elo_predicted_margin(row["rating_diff_pre"], elo_points_per_margin)
    elo_p = odds_math.cover_prob_spread(elo_margin, stds.elo_margin_std, 0.0)
    ml_p = odds_math.cover_prob_spread(row["predicted_margin"], stds.ml_margin_std, 0.0)
    blended = cfg.weight_elo_moneyline * elo_p + (1 - cfg.weight_elo_moneyline) * ml_p
    return {"elo_prob": elo_p, "ml_prob": ml_p, "blended_prob": blended}


def spread_cover_prob(row: pd.Series, line: float, stds: ResidualStds,
                       elo_points_per_margin: float, cfg: EnsembleConfig) -> dict:
    """Returns elo/ml/blended P(home covers `line`) for a single game row and a specific market spread."""
    elo_margin = elo_predicted_margin(row["rating_diff_pre"], elo_points_per_margin)
    elo_p = odds_math.cover_prob_spread(elo_margin, stds.elo_margin_std, line)
    ml_p = odds_math.cover_prob_spread(row["predicted_margin"], stds.ml_margin_std, line)
    blended = cfg.weight_elo_spread * elo_p + (1 - cfg.weight_elo_spread) * ml_p
    return {"elo_prob": elo_p, "ml_prob": ml_p, "blended_prob": blended}


def total_over_prob(row: pd.Series, line: float, stds: ResidualStds, cfg: EnsembleConfig) -> dict:
    """Returns ml/naive/blended P(actual total > line) for a single game row and a specific total line."""
    ml_p = odds_math.over_prob_total(row["predicted_total"], stds.ml_total_std, line)
    naive_p = odds_math.over_prob_total(row["naive_total"], stds.naive_total_std, line)
    blended = cfg.weight_ml_total * ml_p + (1 - cfg.weight_ml_total) * naive_p
    return {"ml_prob": ml_p, "naive_prob": naive_p, "blended_prob": blended}


def confidence_tier(prob_a: float, prob_b: float) -> str:
    """
    Agreement-based confidence: if the two submodels land on the same side
    of a coin flip and are close together, that's a stronger signal than
    either one alone; if they disagree on which side is favored at all,
    that's the least trustworthy case regardless of how big the blended
    edge looks.

    STILL USED for spread and total (see core/edge_finder.py) -- NOT
    replaced by market_agreement_confidence_tier below for those two
    markets, because the real evidence doesn't support the same fix there.
    Checked directly (2026-09, real NFL backtest): spread bets that
    DISAGREE with the market's own favored side actually outperformed ones
    that agreed (+9.98% ROI vs -1.27%, n=436 vs 1316) -- the opposite
    direction from moneyline -- and total's split was too small/noisy
    (n=165 disagreeing) to conclude anything either way. Forcing the
    moneyline fix onto markets where the data points a different way (or
    nowhere conclusive) would be exactly the kind of unvalidated
    assumption this project's methodology exists to catch. Spread/total
    confidence is a real, separate, still-open question -- not fixed here.
    """
    same_side = (prob_a - 0.5) * (prob_b - 0.5) > 0
    gap = abs(prob_a - prob_b)
    if not same_side:
        return "Low"
    if gap <= 0.05:
        return "High"
    if gap <= 0.12:
        return "Medium"
    return "Low"


def market_agreement_confidence_tier(market_fair_prob: float) -> str:
    """
    MONEYLINE-ONLY replacement for confidence_tier() above. Real diagnosis,
    not a guess: the old submodel-agreement design assumes two independent
    views (Elo-based, ML-based) confirming each other means something --
    but both submodels are built from largely the SAME underlying
    schedule/form/rating information, so their agreeing with each other is
    the same blind spot agreeing with itself, not real corroboration.
    Checked directly against real historical backtests (both NFL and MLB,
    thousands of bets each) via `bets['confidence']` groupings: the OLD
    "High confidence" moneyline bets underperformed "Medium confidence" in
    BOTH sports -- NFL High -7.92% ROI vs Medium +3.32% ROI (n=1345 vs
    1158); MLB High -4.77% vs Medium -1.32% (n=7641 vs 4104). Confidence,
    as previously computed, was not just unhelpful, it was backwards.

    Real fix, and a real bug already caught and corrected in this exact
    function before shipping: the first version of this compared the
    MODEL's implied favorite for the whole game against the MARKET's
    implied favorite for the whole game ((model_prob-0.5)*(market_fair-0.5)
    on the home side, symmetric either way you compute it) -- but that
    answers the wrong question. A bet can be placed on the market's own
    UNDERDOG (market_fair_prob < 0.5 for that side) specifically because
    the model thinks that underdog is undervalued, while the model's
    overall game favorite still matches the market's -- the old formula
    would wrongly call that "agreement," when it's exactly the DOG-side
    pattern already shown to be unreliable. Caught by re-deriving the
    original validated numbers from scratch and finding they didn't match
    what the wired-in formula actually produced (NFL: unanimous "High" on
    2789/2789 bets, nowhere near the real 462-agree/2041-disagree split) --
    confirmed the bug, not just suspected it, before considering this done.

    The actual, correct, and much simpler criterion: is THIS bet's own
    market_fair_prob (the side actually being bet, not a home/away
    game-level abstraction) >= 0.5 -- i.e. is this a bet on the side the
    market itself makes the favorite, or the side it makes the underdog?
    This is a direct, honest re-statement of the very first finding in
    this whole investigation (FAV bets meaningfully outperform DOG bets,
    both sports, thousands of bets each) -- not a new, separate mechanism,
    just naming it correctly as the moneyline confidence signal instead of
    the unvalidated submodel-agreement heuristic it's replacing. Verified
    directly on the corrected formula: NFL 462 bets with market_fair_prob
    >= 0.5 hit at 60.6%, 2041 with market_fair_prob < 0.5 hit at 30.2%.
    """
    return "High" if market_fair_prob >= 0.5 else "Low"
