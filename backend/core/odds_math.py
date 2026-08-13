"""
Sport-agnostic odds math: American<->decimal conversion, vig removal,
implied probability, expected value, Kelly stake sizing, and closing-line
value. Every other module (power ratings, ML, ensemble, edge finder,
backtest) builds on these pure functions — keep them dependency-free and
well-tested by hand, since a bug here silently corrupts every downstream
number.

All odds in this project are stored as DECIMAL odds (matches the source
dataset). A decimal price of 1.91 means a $1 stake returns $1.91 total
($0.91 profit) if it wins.
"""

import math


# ---------- American <-> Decimal ----------

def american_to_decimal(american: float) -> float:
    """Convert American odds (e.g. -110, +150) to decimal odds (e.g. 1.91, 2.50)."""
    if american > 0:
        return 1.0 + (american / 100.0)
    else:
        return 1.0 + (100.0 / abs(american))


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal}")
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100.0)
    else:
        return round(-100.0 / (decimal - 1.0))


# ---------- Implied probability & vig removal ----------

def implied_prob(decimal_odds: float) -> float:
    """Raw implied probability from a single decimal price, vig included."""
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def devig_two_way(decimal_a: float, decimal_b: float) -> tuple:
    """
    Remove the vig from a two-sided market (moneyline, one side of a spread,
    over/under) using the multiplicative method: raw implied probabilities
    are normalized so they sum to 1.0.

    Returns (fair_prob_a, fair_prob_b). Both are None if either price is
    missing/invalid (caller should skip the game for that market rather
    than guess).
    """
    if decimal_a is None or decimal_b is None:
        return None, None
    if decimal_a <= 1.0 or decimal_b <= 1.0:
        return None, None
    p_a = implied_prob(decimal_a)
    p_b = implied_prob(decimal_b)
    overround = p_a + p_b
    if overround <= 0:
        return None, None
    return p_a / overround, p_b / overround


def vig_pct(decimal_a: float, decimal_b: float) -> float:
    """The bookmaker's overround on a two-way market, as a percentage (e.g. 4.5)."""
    if decimal_a is None or decimal_b is None or decimal_a <= 1.0 or decimal_b <= 1.0:
        return None
    return (implied_prob(decimal_a) + implied_prob(decimal_b) - 1.0) * 100.0


# ---------- Expected value & Kelly ----------

def expected_value(model_prob: float, decimal_odds: float, stake: float = 1.0) -> float:
    """EV of a bet in stake units: model_prob * profit - (1 - model_prob) * stake."""
    profit = (decimal_odds - 1.0) * stake
    return model_prob * profit - (1.0 - model_prob) * stake


def edge_pct(model_prob: float, decimal_odds: float) -> float:
    """
    Edge as (model probability - market's devigged fair probability) is the
    "true" edge, but when comparing directly against a single price it's
    often more useful to express edge as the gap between our win probability
    and the price's break-even probability. Returns a percentage.
    """
    breakeven = implied_prob(decimal_odds)
    return (model_prob - breakeven) * 100.0


def kelly_fraction(model_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly stake as a fraction of bankroll. `fraction` defaults to
    quarter-Kelly — full Kelly is mathematically optimal for log-bankroll
    growth ONLY if model_prob is exactly correct, and real model error makes
    full Kelly dangerously oversized in practice. Returns 0.0 if there is no
    edge (never suggests betting against your own model).
    """
    b = decimal_odds - 1.0  # net odds
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    full_kelly = (b * model_prob - q) / b
    if full_kelly <= 0:
        return 0.0
    return full_kelly * fraction


# ---------- Closing line value ----------

def clv_pct(price_taken: float, closing_price: float) -> float:
    """
    Closing line value: how much better (or worse) the price taken was vs.
    the closing price, in probability terms. Positive CLV (consistently
    beating the close) is the single most reliable indicator of a bettor's
    long-run edge, independent of whether any individual bet won.
    """
    if price_taken is None or closing_price is None:
        return None
    if price_taken <= 1.0 or closing_price <= 1.0:
        return None
    return (implied_prob(closing_price) - implied_prob(price_taken)) * 100.0 / implied_prob(closing_price)


# ---------- Probability from a point prediction (margin / total) ----------

def _normal_cdf(x: float, mean: float, std: float) -> float:
    """P(X <= x) for X ~ Normal(mean, std), via the stdlib erf (no scipy dependency)."""
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def cover_prob_spread(predicted_margin: float, residual_std: float, line: float) -> float:
    """
    Probability the home team covers a spread `line` (home perspective,
    e.g. line = -3.5 means home favored by 3.5), given a model's predicted
    home margin and the historical std-dev of (actual margin - predicted
    margin). Assumes residuals are ~Normal, a standard simplifying
    approximation for spread markets.
    """
    threshold = -line  # home covers if actual_margin > -line
    return 1.0 - _normal_cdf(threshold, predicted_margin, residual_std)


def over_prob_total(predicted_total: float, residual_std: float, line: float) -> float:
    """Probability the actual total goes OVER `line`, given a predicted total and residual std-dev."""
    return 1.0 - _normal_cdf(line, predicted_total, residual_std)
