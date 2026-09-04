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


# Sports where market_agreement_confidence_tier (moneyline only -- see
# core/ensemble.py's docstring) has been directly validated against a real
# historical backtest and shown to actually separate winners from losers
# correctly (not backwards). NOT a default-on behavior: checked directly,
# CFB's moneyline backtest came back CATASTROPHIC under this same tier
# (24.4% hit rate, -17.5% ROI, n=520) -- almost certainly because CFB's
# odds coverage is sparse/partly-assumed (see docs/METHODOLOGY.md's NCAAF
# section), so `market_fair_prob` there often isn't a genuine market read
# to agree or disagree with. NBA checked 2026-09 and explicitly NOT
# added -- see ensemble.py's market_agreement_confidence_tier() docstring:
# an initial pass found a real-looking split, but it was measured on a
# confidence-pre-filtered population; re-checked on the true confidence-
# blind population the split shrinks to noise AND flips sign between
# season halves. NBA moneyline stays backwards and unfixed, honestly.
# NHL still hasn't been checked (zero live picks, lower urgency) -- gets
# the OLD confidence_tier() until someone actually validates this for it
# too. Every sport not in this set keeps the pre-existing submodel-
# agreement confidence_tier() for moneyline, unchanged.
MARKET_AGREEMENT_CONFIDENCE_SPORTS = {"NFL", "MLB"}

# NFL-SPREAD-ONLY: spread_market_disagreement_confidence_tier (see
# ensemble.py's docstring) -- the INVERTED version of the moneyline
# mechanism above, validated separately and specifically for NFL spread.
# Checked directly and NOT generalized to the other sports: MLB/NBA show no
# real ROI gap on this same criterion (hit-rate differs but ROI doesn't --
# just odds-asymmetry noise), and CFB/NHL have degenerate spread odds data
# (market_fair_prob is a constant ~0.5 for every qualifying bet in both --
# no real two-way price to agree or disagree with in the first place).
SPREAD_MARKET_DISAGREEMENT_CONFIDENCE_SPORTS = {"NFL"}

# NHL-SPREAD-ONLY: edge_magnitude_confidence_tier (see ensemble.py's
# docstring) -- a different mechanism again (edge SIZE, not market
# agreement), validated specifically for NHL because NHL's spread odds are
# themselves degenerate (constant -110 for every bet, see above), so the
# only real signal left is how far the model's own blended probability sits
# from a coin flip. Checked directly and NOT generalized: this same
# edge-bucket check is NOT monotonic for NFL/MLB/NBA/CFB spread (middling
# buckets under- or out-perform inconsistently there), so it stays NHL-only.
SPREAD_EDGE_MAGNITUDE_CONFIDENCE_SPORTS = {"NHL"}

# NFL+MLB TOTAL ONLY: total_side_confidence_tier (see ensemble.py's
# docstring) -- checked directly and validated independently in BOTH
# sports (same "under beats over" direction, each with its own real
# effect size and its own season-based split-half stability check), the
# same two-sport-independent-confirmation bar moneyline's fix was held to.
# NOT extended to NBA/NHL/CFB total -- the standing audit already shows
# those three are monotonic (not backwards) under the OLD confidence_tier(),
# so there is no problem there to fix and no reason to touch them.
TOTAL_UNDER_BIAS_CONFIDENCE_SPORTS = {"NFL", "MLB"}


def evaluate_game(row, stds: ensemble.ResidualStds, elo_points_per_margin: float,
                   cfg: ensemble.EnsembleConfig, kelly_frac: float = 0.25,
                   price_point: str = "Close", sport: str = None) -> list:
    """
    `row` must expose (dict-like or pandas Series) `rating_diff_pre`,
    `predicted_margin`, `predicted_total`, `naive_total`, plus the market
    odds columns for the requested `price_point`.

    `sport`, if given, gates which moneyline confidence function is used
    (see MARKET_AGREEMENT_CONFIDENCE_SPORTS above) and which spread
    confidence function is used (see SPREAD_MARKET_DISAGREEMENT_CONFIDENCE_
    SPORTS / SPREAD_EDGE_MAGNITUDE_CONFIDENCE_SPORTS above) -- defaults to
    the old, always-safe submodel-agreement confidence_tier() for both
    markets when sport is None or not in the relevant set, so an
    uncertain/unvalidated caller never silently picks up new behavior.
    """
    opps = []
    sport_u = sport.upper() if sport is not None else None
    use_market_agreement = sport_u in MARKET_AGREEMENT_CONFIDENCE_SPORTS

    def get(col):
        v = row.get(col) if hasattr(row, "get") else row[col]
        return v if v is not None and v == v else None  # NaN check without importing numpy here

    # ---------- Moneyline ----------
    home_ml, away_ml = get(f"Home Odds {price_point}"), get(f"Away Odds {price_point}")
    if home_ml and away_ml:
        fair_home, fair_away = odds_math.devig_two_way(home_ml, away_ml)
        if fair_home is not None:
            ml = ensemble.moneyline_prob(row, stds, elo_points_per_margin, cfg)
            # market_agreement_confidence_tier only for sports where this has
            # been directly validated (see MARKET_AGREEMENT_CONFIDENCE_SPORTS
            # above) -- every other sport keeps the old confidence_tier(),
            # unchanged. Computed PER SIDE, not once per game -- home and
            # away can (and often do) land in different tiers now, since
            # this asks "is THIS side the market's favorite or its
            # underdog," not a game-level abstraction (see that function's
            # docstring for a real bug this replaced, caught before shipping).
            if use_market_agreement:
                conf_home = ensemble.market_agreement_confidence_tier(fair_home)
                conf_away = ensemble.market_agreement_confidence_tier(fair_away)
            else:
                conf_home = conf_away = ensemble.confidence_tier(ml["elo_prob"], ml["ml_prob"])
            opps.append(_opportunity("moneyline", "home", None, ml["blended_prob"],
                                      fair_home, home_ml, conf_home, kelly_frac))
            opps.append(_opportunity("moneyline", "away", None, 1.0 - ml["blended_prob"],
                                      fair_away, away_ml, conf_away, kelly_frac))

    # ---------- Spread ----------
    home_line = get(f"Home Line {price_point}")
    home_line_odds, away_line_odds = get(f"Home Line Odds {price_point}"), get(f"Away Line Odds {price_point}")
    if home_line is not None and home_line_odds and away_line_odds:
        fair_home, fair_away = odds_math.devig_two_way(home_line_odds, away_line_odds)
        if fair_home is not None:
            sp = ensemble.spread_cover_prob(row, home_line, stds, elo_points_per_margin, cfg)
            # Computed PER SIDE, like moneyline's fix -- home and away can
            # land in different tiers, since both new mechanisms below key
            # off per-side values (market_fair_prob / edge_pct), not a
            # game-level abstraction. See SPREAD_MARKET_DISAGREEMENT_
            # CONFIDENCE_SPORTS / SPREAD_EDGE_MAGNITUDE_CONFIDENCE_SPORTS
            # above for which sports are validated for which mechanism.
            if sport_u in SPREAD_MARKET_DISAGREEMENT_CONFIDENCE_SPORTS:
                conf_home = ensemble.spread_market_disagreement_confidence_tier(fair_home)
                conf_away = ensemble.spread_market_disagreement_confidence_tier(fair_away)
            elif sport_u in SPREAD_EDGE_MAGNITUDE_CONFIDENCE_SPORTS:
                edge_home_pct = (sp["blended_prob"] - fair_home) * 100.0
                edge_away_pct = ((1.0 - sp["blended_prob"]) - fair_away) * 100.0
                conf_home = ensemble.edge_magnitude_confidence_tier(edge_home_pct)
                conf_away = ensemble.edge_magnitude_confidence_tier(edge_away_pct)
            else:
                conf_home = conf_away = ensemble.confidence_tier(sp["elo_prob"], sp["ml_prob"])
            opps.append(_opportunity("spread", "home", home_line, sp["blended_prob"],
                                      fair_home, home_line_odds, conf_home, kelly_frac))
            opps.append(_opportunity("spread", "away", -home_line, 1.0 - sp["blended_prob"],
                                      fair_away, away_line_odds, conf_away, kelly_frac))

    # ---------- Total ----------
    total_line = get(f"Total Score {price_point}")
    over_odds, under_odds = get(f"Total Score Over {price_point}"), get(f"Total Score Under {price_point}")
    if total_line is not None and over_odds and under_odds:
        fair_over, fair_under = odds_math.devig_two_way(over_odds, under_odds)
        if fair_over is not None:
            tot = ensemble.total_over_prob(row, total_line, stds, cfg)
            # total_side_confidence_tier only for sports directly validated
            # (see TOTAL_UNDER_BIAS_CONFIDENCE_SPORTS above) -- every other
            # sport keeps the old confidence_tier(), unchanged. This one
            # doesn't need a per-side probability at all (unlike moneyline/
            # spread's fixes) -- the validated signal is the literal side
            # label itself (over vs under), not a market or model quantity.
            if sport_u in TOTAL_UNDER_BIAS_CONFIDENCE_SPORTS:
                conf_over = ensemble.total_side_confidence_tier("over")
                conf_under = ensemble.total_side_confidence_tier("under")
            else:
                conf_over = conf_under = ensemble.confidence_tier(tot["ml_prob"], tot["naive_prob"])
            opps.append(_opportunity("total", "over", total_line, tot["blended_prob"],
                                      fair_over, over_odds, conf_over, kelly_frac))
            opps.append(_opportunity("total", "under", total_line, 1.0 - tot["blended_prob"],
                                      fair_under, under_odds, conf_under, kelly_frac))

    return opps
