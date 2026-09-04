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

    NBA checked (2026-09, same audit pass that diagnosed spread/total
    below) and explicitly NOT adopted -- caught the exact population-
    contamination mistake this whole module's docstrings warn about, one
    more time, before shipping: a first pass measured NBA's FAV/DOG split
    on a `bets` dataframe that had already been filtered down by the OLD
    confidence_tier() (Medium/High only), which is a different, smaller,
    non-representative population than what the NEW tier would actually
    gate -- and on that biased population the split looked real (delta
    ~5.7pp ROI). Re-run on the true confidence-blind population (every
    edge>=3% opportunity, no confidence filter at all) as the "simple,
    direct" check this project's discipline requires: fair_prob>=0.5
    n=2125, -1.45% ROI vs fair_prob<0.5 n=9340, -2.15% ROI -- delta only
    0.70pp on a combined stderr of ~2.6pp, not distinguishable from noise.
    Worse, a season-based split-half check flips SIGN outright (first half
    delta +3.48pp, second half delta -2.04pp -- DOG actually beat FAV in
    the second half) -- the opposite of a stable effect. NBA moneyline
    stays on the old confidence_tier() -- still genuinely backwards per
    the standing audit, with no validated fix found this pass. Left
    honestly unresolved rather than shipped on a number that didn't survive
    its own re-check.
    """
    return "High" if market_fair_prob >= 0.5 else "Low"


def spread_market_disagreement_confidence_tier(market_fair_prob: float) -> str:
    """
    NFL-SPREAD-ONLY. Looks like the exact same criterion as
    market_agreement_confidence_tier() above but the verdict is INVERTED --
    documented separately (not just a flag flip on that function) because
    the reasoning is genuinely different and this must never be copy-pasted
    onto another market without its own direct check.

    Diagnosed directly, 2026-09, as the required follow-up to the
    already-logged finding that NFL spread confidence_tier() (submodel
    agreement) is backwards: same "does this bet's own devigged
    market_fair_prob sit >= 0.5" question market_agreement_confidence_tier()
    asks for moneyline -- but for spread, the side the market itself prices
    WORSE after devigging (market_fair_prob < 0.5) is the side that
    actually performs better. Measured on the TRUE confidence-blind
    population (every edge>=3% spread opportunity from evaluate_game(),
    with no confidence-tier filter applied at all -- not a `bets` dataframe
    already filtered by the OLD confidence_tier(), which is a different,
    smaller, non-representative population; this distinction bit the first
    draft of the NBA moneyline check in market_agreement_confidence_tier()'s
    docstring above and is deliberately re-stated here): n=672 bets with
    market_fair_prob < 0.5 hit 53.8%, +7.92% ROI, vs n=1625 with
    market_fair_prob >= 0.5, 51.7% hit, -1.81% ROI. Checked for stability,
    not just a single lucky split: same direction and similar magnitude in
    BOTH halves of the season range (first half: +9.73% n=399 vs +0.49%
    n=766; second half: +5.27% n=273 vs -3.86% n=859) -- not a one-season
    fluke. (An earlier measurement on the confidence-pre-filtered
    population gave 436/1316 bets and +9.98%/-1.27% ROI -- same direction,
    different exact numbers because of the population-contamination issue
    above; the wired-in path was verified to reproduce the 672/1625 numbers
    exactly, not the pre-filtered ones.)

    Real, externally-plausible mechanism (not just "the number went the
    other way"): NFL spread lines are set (and vigged) knowing public money
    leans toward standard/round favorites, so the side of a two-way spread
    market that devigs to a WORSE-than-50% fair probability is often the
    side getting less public action for reasons unrelated to true win
    probability -- the classic contrarian/"fade the public" edge already
    well documented in sports-betting literature, and the mirror image of
    moneyline's FAV/DOG pattern rather than a contradiction of it (moneyline
    prices a much wider spread of true win probabilities where the
    market's favorite really is more reliable; a point spread is
    engineered to be near a coin flip either way, so its bias signal comes
    from vig asymmetry, not favorite/underdog strength).

    Checked directly and NOT generalized (all on the same true confidence-
    blind population): MLB spread shows a big HIT RATE gap on this same
    split (56.4% vs 39.7%) but virtually IDENTICAL ROI (-3.19% vs -3.24%,
    delta 0.05pp on ~1.4pp stderr each) -- the hit-rate gap is just
    run-line odds asymmetry, not real edge, so this is correctly a null
    result for MLB, not adopted there. NBA shows no real gap either
    (-1.04% vs -0.12%, well within noise). CFB and NHL couldn't
    even be checked on this axis -- both have degenerate spread odds data
    (CFB: 100% of qualifying bets show market_fair_prob exactly 0.500;
    NHL: same, market_odds is a constant 1.909091/-110 for every single
    spread bet) -- i.e. no real two-way spread price exists to agree or
    disagree with in either sport's dataset, the same "sparse/assumed odds"
    problem already documented for CFB moneyline, just discovered here for
    two more sport/market combinations.
    """
    return "High" if market_fair_prob < 0.5 else "Low"


def edge_magnitude_confidence_tier(edge_pct: float) -> str:
    """
    NHL-SPREAD-ONLY. A genuinely different mechanism from every other
    confidence function in this module: instead of comparing two submodels
    or the model against the market, this just asks how big the model's own
    claimed edge (model_prob - market_fair_prob, already computed for every
    BetOpportunity) is -- on the theory that for NHL specifically, the
    market side of that comparison carries no real information (see
    spread_market_disagreement_confidence_tier()'s docstring above: NHL
    puck-line spread odds are a constant -110/1.909091 for literally every
    bet in this dataset, so market_fair_prob is always ~0.5 and
    "agreement with the market" is meaningless here). With the market side
    neutralized, the entire remaining signal in "how big is this edge" is
    how far the BLENDED MODEL PROBABILITY itself sits from a coin flip --
    and diagnosed directly, that turns out to be a strong, clean,
    monotonic real predictor for NHL puck-line bets specifically, not
    circular reasoning about "big edges being good by construction" (the
    same edge-bucket check on NFL/MLB/NBA/CFB spread shows NO such
    monotonic pattern -- see decision log).

    Real numbers, measured on the TRUE confidence-blind population (every
    edge>=3% spread opportunity, no confidence-tier filter applied -- see
    spread_market_disagreement_confidence_tier()'s docstring above for why
    that distinction matters and bit an earlier NBA check): edge 5-10%:
    n=1133, -0.25% ROI; edge 10-20%: n=4128, +21.63% ROI; edge 20%+: n=950,
    +32.63% ROI -- cleanly monotonic and each step is many standard errors
    apart (5-10% vs 10-20%: delta ~21.9pp on ~3.2pp combined stderr;
    10-20% vs 20%+: delta ~11.0pp on ~3.2pp combined stderr). Checked for
    stability with a season-based split-half: 10-20% bucket is +22.31%
    (first half) vs +21.22% (second half); 20%+ is +35.71% vs +30.98% --
    both bands nearly identical across the split, about as stable as this
    project has seen any single mechanism hold up.

    Mapped to the existing three-tier vocabulary using natural breaks in
    that same data: edge < 10% -> "Low" (the two worst-performing buckets,
    -0.25% and a noisy tiny -20.45% n=60 sub-5-bucket), 10% <= edge < 20% ->
    "Medium" (+21.63%), edge >= 20% -> "High" (+32.63%) -- reproduces the
    monotonic High > Medium > Low ordering confidence badges are supposed
    to mean, which the old submodel-agreement confidence_tier() did not
    (old High +2.55% actually UNDERPERFORMED old Medium +21.63% for NHL
    spread -- this replaces that, gated to NHL spread only). Verified the
    wired-in path reproduces these exact n=4128/n=950 counts and ROI
    figures, not the smaller pre-filtered-population numbers an earlier
    draft of this diagnosis initially found (n=2423/n=719 at +17.95%/
    +31.96% -- same direction, wrong population).
    """
    if edge_pct >= 20.0:
        return "High"
    if edge_pct >= 10.0:
        return "Medium"
    return "Low"


def total_side_confidence_tier(side: str) -> str:
    """
    NFL+MLB TOTAL ONLY. Diagnosed directly, 2026-09, as the required
    follow-up to the already-logged finding that NFL/MLB total
    confidence_tier() (submodel agreement) is backwards. Unlike every
    other tier function in this module, this one isn't a market- or
    model-probability comparison at all -- it's the literal bet side
    (over/under) itself, because that turned out to be the actual signal.

    Real, externally-motivated reasoning up front (not "let's see what
    happens if we split by side"): total betting is one of the most
    widely documented public-bias markets in sports-betting literature --
    the "over" is the more entertaining, more heavily bet public side in
    both NFL and MLB, which pushes total lines up and/or the over's vig
    price worse than fair value, meaning the "under" side is
    systematically the side carrying more real value after the public's
    money has already been priced in. This predicts exactly the direction
    found in the data, in TWO independently-checked sports, not one.

    Measured on the TRUE confidence-blind population (every edge>=3% total
    opportunity from evaluate_game(), no confidence-tier filter applied --
    see spread_market_disagreement_confidence_tier()'s docstring for why
    that population distinction matters): NFL under n=1396, +6.64% ROI vs
    over n=949, -5.41% ROI (delta 12.05pp on ~4.0pp combined stderr,
    z~3.0). MLB under n=6684, -0.76% ROI vs over n=10318, -4.56% ROI
    (delta 3.80pp on ~1.5pp combined stderr, z~2.5). Checked for stability
    with a season-based split-half in both sports, not just the full-
    sample number: NFL first half delta +10.19pp, second half +14.84pp
    (same direction, actually grows); MLB first half delta +3.81pp, second
    half +3.25pp (same direction, stable) -- real and consistent across
    time in both sports independently, the same two-sport-confirmation bar
    the moneyline fix was held to. (An earlier measurement on the
    confidence-pre-filtered population found a similar direction but
    weaker/less stable numbers for both sports -- MLB's second-half delta
    in particular looked like it had decayed to ~0 there; re-measured on
    the correct confidence-blind population, both sports' effects are
    real and hold up across time. This was the same population-
    contamination trap documented in the moneyline/spread docstrings
    above, caught here before shipping rather than after.)

    NOT extended to NBA/NHL/CFB total -- the standing audit already shows
    those three are monotonic (not backwards) under the old
    confidence_tier(), so there's no problem to fix there and this
    mechanism was never even checked against their data.
    """
    return "High" if side == "under" else "Low"
