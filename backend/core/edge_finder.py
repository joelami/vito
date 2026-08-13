"""
For one game (historical or manually entered), compares the ensemble's
model probability against the market's devigged fair probability for every
side of every bet type the market offers, and reports edge %, EV, and
suggested stake (flat + fractional Kelly).

Historical rows and manually-entered upcoming games are evaluated through
this exact same function — the odds columns just need to use the same
names the NFL loader produces (`Home Odds Close`, `Home Line Close`, etc.),
which is why the manual-game entry form asks for the same fields.

`price_point` controls which set of columns is used ("Open", "Close", "Min",
"Max"). This matters for backtesting: you can never actually bet the
CLOSING line (it's only known once betting closes), so a realistic
simulation of "would this have found value" evaluates edge against the
OPENING line — the price actually available before kickoff — and separately
measures closing-line value (how the price taken compares to where the
line ended up) as the credibility check.
"""

from dataclasses import dataclass, asdict

from . import odds_math
from . import ensemble


@dataclass
class BetOpportunity:
    market: str          # "moneyline" | "spread" | "total"
    side: str             # "home" | "away" | "over" | "under"
    line: float           # spread/total number, None for moneyline
    market_odds: float    # decimal price for this side
    market_fair_prob: float   # devigged
    model_prob: float
    confidence: str
    edge_pct: float
    ev_per_unit: float
    kelly_stake: float    # fraction of bankroll, fractional Kelly

    def to_dict(self):
        return asdict(self)


def _opportunity(market, side, line, model_prob, market_fair_prob, market_odds,
                  confidence, kelly_frac) -> BetOpportunity:
    return BetOpportunity(
        market=market, side=side, line=line, market_odds=market_odds,
        market_fair_prob=market_fair_prob, model_prob=model_prob, confidence=confidence,
        edge_pct=(model_prob - market_fair_prob) * 100.0,
        ev_per_unit=odds_math.expected_value(model_prob, market_odds),
        kelly_stake=odds_math.kelly_fraction(model_prob, market_odds, kelly_frac),
    )


def evaluate_game(row, stds: ensemble.ResidualStds, elo_points_per_margin: float,
                   cfg: ensemble.EnsembleConfig, kelly_frac: float = 0.25,
                   price_point: str = "Close") -> list:
    """
    `row` must expose (dict-like or pandas Series) `rating_diff_pre`,
    `predicted_margin`, `predicted_total`, `naive_total`, plus the market
    odds columns for the requested `price_point`.
    """
    opps = []

    def get(col):
        v = row.get(col) if hasattr(row, "get") else row[col]
        return v if v is not None and v == v else None  # NaN check without importing numpy here

    # ---------- Moneyline ----------
    home_ml, away_ml = get(f"Home Odds {price_point}"), get(f"Away Odds {price_point}")
    if home_ml and away_ml:
        fair_home, fair_away = odds_math.devig_two_way(home_ml, away_ml)
        if fair_home is not None:
            ml = ensemble.moneyline_prob(row, stds, elo_points_per_margin, cfg)
            conf = ensemble.confidence_tier(ml["elo_prob"], ml["ml_prob"])
            opps.append(_opportunity("moneyline", "home", None, ml["blended_prob"],
                                      fair_home, home_ml, conf, kelly_frac))
            opps.append(_opportunity("moneyline", "away", None, 1.0 - ml["blended_prob"],
                                      fair_away, away_ml, conf, kelly_frac))

    # ---------- Spread ----------
    home_line = get(f"Home Line {price_point}")
    home_line_odds, away_line_odds = get(f"Home Line Odds {price_point}"), get(f"Away Line Odds {price_point}")
    if home_line is not None and home_line_odds and away_line_odds:
        fair_home, fair_away = odds_math.devig_two_way(home_line_odds, away_line_odds)
        if fair_home is not None:
            sp = ensemble.spread_cover_prob(row, home_line, stds, elo_points_per_margin, cfg)
            conf = ensemble.confidence_tier(sp["elo_prob"], sp["ml_prob"])
            opps.append(_opportunity("spread", "home", home_line, sp["blended_prob"],
                                      fair_home, home_line_odds, conf, kelly_frac))
            opps.append(_opportunity("spread", "away", -home_line, 1.0 - sp["blended_prob"],
                                      fair_away, away_line_odds, conf, kelly_frac))

    # ---------- Total ----------
    total_line = get(f"Total Score {price_point}")
    over_odds, under_odds = get(f"Total Score Over {price_point}"), get(f"Total Score Under {price_point}")
    if total_line is not None and over_odds and under_odds:
        fair_over, fair_under = odds_math.devig_two_way(over_odds, under_odds)
        if fair_over is not None:
            tot = ensemble.total_over_prob(row, total_line, stds, cfg)
            conf = ensemble.confidence_tier(tot["ml_prob"], tot["naive_prob"])
            opps.append(_opportunity("total", "over", total_line, tot["blended_prob"],
                                      fair_over, over_odds, conf, kelly_frac))
            opps.append(_opportunity("total", "under", total_line, 1.0 - tot["blended_prob"],
                                      fair_under, under_odds, conf, kelly_frac))

    return opps
