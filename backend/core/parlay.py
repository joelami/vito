"""
Combines individual bet legs into a parlay. Sport-agnostic — works on any
mix of `edge_finder.BetOpportunity`-shaped legs from any sport this app
supports.

The one thing this module refuses to do by default: silently compute a
"combined edge" for legs from the SAME game. Parlay legs from different,
unrelated games are safe to treat as statistically independent (multiplying
probabilities is valid), but legs from the same game are usually
CORRELATED — e.g. "Team A wins" and "Team A covers -3.5" tend to happen
together far more often than independence would predict. Multiplying their
probabilities as if independent systematically overstates the true combined
probability, manufacturing an edge that isn't real. This is a well-known way
bettors fool themselves with same-game parlays (and, not coincidentally, why
books often price them worse than the naive independent math would suggest —
they know bettors make this exact mistake). Without a fitted correlation
model (which this project doesn't have), the honest move is to flag it and
withhold a fabricated number, not guess.
"""

import itertools
from dataclasses import dataclass, field


@dataclass
class ParlayLeg:
    game_key: str          # anything that uniquely identifies the game (e.g. game_id or "date|home|away")
    market: str
    side: str
    model_prob: float
    market_odds: float     # decimal
    market_fair_prob: float
    line: float = None     # None for moneyline
    confidence: str = None


@dataclass
class ParlayResult:
    legs: list
    combined_decimal_odds: float          # always computable — matches how books actually price a parlay
    same_game_legs_present: bool
    combined_model_prob: float = None     # None if same_game_legs_present (can't honestly compute it)
    combined_market_fair_prob: float = None
    edge_pct: float = None
    ev_per_unit: float = None
    kelly_stake: float = None
    warning: str = None

    def to_dict(self):
        return {
            "legs": [vars(l) for l in self.legs],
            "combined_decimal_odds": self.combined_decimal_odds,
            "same_game_legs_present": self.same_game_legs_present,
            "combined_model_prob": self.combined_model_prob,
            "combined_market_fair_prob": self.combined_market_fair_prob,
            "edge_pct": self.edge_pct,
            "ev_per_unit": self.ev_per_unit,
            "kelly_stake": self.kelly_stake,
            "warning": self.warning,
        }


def build_parlay(legs: list, kelly_frac: float = 0.25) -> ParlayResult:
    """
    `legs` is a list of `ParlayLeg`. Decimal odds always multiply (that's
    the actual payout structure of a parlay, regardless of correlation —
    correlation affects whether the PROBABILITY of winning is what naive
    multiplication suggests, not the payout if you do win). Model/market
    probabilities only get combined (and an edge/EV/Kelly computed) when
    every leg is from a distinct game; otherwise those fields come back
    `None` with an explicit warning rather than a number that would be
    quietly wrong.
    """
    from . import odds_math  # local import, avoids a module-load-order dependency

    if not legs:
        raise ValueError("build_parlay requires at least one leg")

    combined_decimal_odds = 1.0
    for leg in legs:
        combined_decimal_odds *= leg.market_odds

    game_keys = [l.game_key for l in legs]
    same_game_legs_present = len(set(game_keys)) < len(game_keys)

    if same_game_legs_present or len(legs) < 2:
        warning = (
            "Two or more legs share a game — correlation between them isn't modeled, so no "
            "combined probability/edge/EV is computed. The combined payout odds above are real; "
            "whether they represent value is NOT something this tool can currently tell you for a "
            "same-game parlay. Evaluate each leg's own individual edge instead."
            if same_game_legs_present else None
        )
        return ParlayResult(
            legs=legs, combined_decimal_odds=combined_decimal_odds,
            same_game_legs_present=same_game_legs_present, warning=warning,
        )

    combined_model_prob = 1.0
    combined_market_fair_prob = 1.0
    for leg in legs:
        combined_model_prob *= leg.model_prob
        combined_market_fair_prob *= leg.market_fair_prob

    return ParlayResult(
        legs=legs, combined_decimal_odds=combined_decimal_odds,
        same_game_legs_present=False,
        combined_model_prob=combined_model_prob,
        combined_market_fair_prob=combined_market_fair_prob,
        edge_pct=(combined_model_prob - combined_market_fair_prob) * 100.0,
        ev_per_unit=odds_math.expected_value(combined_model_prob, combined_decimal_odds),
        kelly_stake=odds_math.kelly_fraction(combined_model_prob, combined_decimal_odds, kelly_frac),
        warning="Legs are from different games, treated as statistically independent. This is a "
                "reasonable approximation across unrelated games, but real independence isn't "
                "guaranteed (e.g. weather affecting multiple outdoor games the same day) — treat "
                "as approximate, not exact.",
    )


def _theme(legs: list) -> str:
    """A short, purely descriptive label for a candidate parlay — not part of
    the edge calculation, just a human-readable summary of what the legs
    have in common, if anything obvious."""
    sides = {l.side for l in legs}
    markets = {l.market for l in legs}
    if len(sides) == 1 and len(markets) == 1:
        return f"all {next(iter(sides))} {next(iter(markets))}s"
    if sides <= {"home", "favorite"}:
        return "all favorites"
    if sides <= {"away", "underdog"}:
        return "all underdogs"
    if len(markets) == 1:
        return f"all {next(iter(markets))} bets"
    return f"{len(legs)}-leg mixed parlay"


def suggest_parlays(available_legs: list, max_legs: int = 4, top_n: int = 10,
                     min_leg_edge_pct: float = 3.0, min_confidence: tuple = ("Medium", "High"),
                     kelly_frac: float = 0.25) -> list:
    """
    Builds and ranks candidate parlays from a pool of individual
    opportunities (typically everything on the Dashboard / Suggestions
    page). Only ever combines legs from DIFFERENT games — same-game legs
    are excluded from this pool entirely rather than risk silently
    combining correlated legs into a fabricated-edge suggestion, the same
    honesty boundary `build_parlay` already enforces for a manually-built
    parlay.

    To keep this tractable: at most one leg per game is considered (the
    single best-edge qualifying opportunity for that game — including a
    second, correlated leg from the same game wouldn't be usable in a
    parlay anyway per the rule above, so there's no honest use for keeping
    the runner-up). From that reduced pool, every combination of size 2 to
    `max_legs` is evaluated and ranked by edge_pct.
    """
    qualifying = [l for l in available_legs if l.model_prob is not None
                  and (l.confidence is None or l.confidence in min_confidence)]

    best_per_game = {}
    for leg in qualifying:
        edge = (leg.model_prob - leg.market_fair_prob) * 100.0
        if edge < min_leg_edge_pct:
            continue
        current = best_per_game.get(leg.game_key)
        current_edge = (current.model_prob - current.market_fair_prob) * 100.0 if current else -1
        if current is None or edge > current_edge:
            best_per_game[leg.game_key] = leg

    pool = list(best_per_game.values())
    if len(pool) < 2:
        return []

    candidates = []
    max_k = min(max_legs, len(pool))
    for k in range(2, max_k + 1):
        for combo in itertools.combinations(pool, k):
            result = build_parlay(list(combo), kelly_frac=kelly_frac)
            if result.edge_pct is None:
                continue  # shouldn't happen given the one-leg-per-game pool, but stay defensive
            candidates.append((result, _theme(list(combo))))

    candidates.sort(key=lambda c: c[0].edge_pct, reverse=True)
    return [{"theme": theme, **result.to_dict()} for result, theme in candidates[:top_n]]
