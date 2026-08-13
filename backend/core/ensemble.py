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
