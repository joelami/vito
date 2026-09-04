"""
Disciplined hypothesis-testing loop — the "self-learning" mechanism behind
continued model improvement. Exists to answer "keep making the model
better" in a way that stays trustworthy, not just a fancier version of
tuning knobs until a backtest looks good (see decision_log.jsonl: "Reject
'keep tuning parameters until the backtest turns positive'").

Four rules, enforced by this module's shape rather than left to discipline
alone:

1. Every hypothesis requires `reasoning` up front — a real, externally
   motivated reason (domain knowledge, a diagnosed model weakness, outside
   research), never "let's see what happens if we change X." A hypothesis
   object literally cannot be constructed without one.
2. Every hypothesis is scored on BOTH statistical fit (does the model
   predict the game itself better — margin/total correlation, MAE) AND
   backtest ROI. These are meant to move together for a real improvement;
   when ROI moves but statistical fit doesn't, that's flagged as
   `suspicious` — the exact signature of noise dressed up as edge, not a
   result to celebrate.
3. Every run is logged via `versioning.log_decision()` — reasoning AND
   outcome, whether the hypothesis helped, hurt, or did nothing — so the
   full experiment history stays visible. That visibility is itself a
   guardrail: it's much harder to quietly retry the same idea until it
   looks good when every previous attempt is sitting in a permanent log.
4. A hypothesis is only "adopted" into the live feature/config set if it
   doesn't hurt AND was well-motivated — never adopted purely because ROI
   went up, since ROI on one fixed historical sample is exactly the number
   most vulnerable to being gamed by trying enough things (the
   multiple-comparisons trap).
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import versioning
from . import shrinkage, watchlist


@dataclass
class Hypothesis:
    name: str
    reasoning: str       # REQUIRED — why this might genuinely help, not "let's try it"
    sport: str = "NFL"

    def __post_init__(self):
        if not self.reasoning or len(self.reasoning.strip()) < 20:
            raise ValueError(
                f"Hypothesis {self.name!r} needs real reasoning (>=20 chars), not a placeholder. "
                "This is enforced on purpose — see core/research.py's module docstring."
            )


@dataclass
class HypothesisResult:
    hypothesis: Hypothesis
    baseline: dict        # {margin_corr, total_corr, roi_pct, roi_stderr_pct, ...}
    variant: dict          # same shape, measured with the hypothesis applied
    margin_corr_delta: float
    total_corr_delta: float
    roi_delta_pct: float
    roi_delta_within_noise: bool
    suspicious: bool       # ROI moved meaningfully but statistical fit didn't
    recommendation: str    # "adopt" | "reject" | "adopt_cautiously" | "inconclusive"

    def to_dict(self):
        return {
            "name": self.hypothesis.name, "sport": self.hypothesis.sport,
            "reasoning": self.hypothesis.reasoning,
            "baseline": self.baseline, "variant": self.variant,
            "margin_corr_delta": self.margin_corr_delta, "total_corr_delta": self.total_corr_delta,
            "roi_delta_pct": self.roi_delta_pct, "roi_delta_within_noise": self.roi_delta_within_noise,
            "suspicious": self.suspicious, "recommendation": self.recommendation,
        }


# A statistical-fit change this small isn't distinguishable from noise on
# datasets this size — below this, treat correlation as "didn't move."
CORR_NOISE_FLOOR = 0.005


def _inverse_normal_cdf(p: float) -> float:
    """Standard normal quantile via bisection on math.erf (no scipy). Simple
    and directly verifiable (z=1.96 for p=0.975, the textbook value) rather
    than a transcription-prone rational-approximation formula — precision
    doesn't matter much here, but a subtle bug in a magic formula would."""
    import math
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tests_run_so_far(sport: str = None) -> int:
    """Counts real hypothesis tests already logged, for multiple-comparisons
    awareness — the more tests run, the more of them will clear a fixed
    noise bar by pure chance, so the bar for any individual test needs to
    rise as the count grows. Excludes this module's own regression-test
    entries (named with 'regression_test'/'regression_check')."""
    decisions = versioning.read_decision_log()
    count = 0
    for d in decisions:
        name = d.get("decision", "")
        if "Hypothesis test" not in name:
            continue
        if sport and d.get("sport") != sport:
            continue
        if "regression_test" in name or "regression_check" in name:
            continue
        count += 1
    return count


def bonferroni_stderr_multiplier(n_tests: int, family_alpha: float = 0.05) -> float:
    """
    How many standard errors a result needs to clear to still count as
    "clearly not noise" once you account for how many tests have already
    been run this "family" (Bonferroni correction: target a 5% chance of
    ANY false positive across all n tests combined, not 5% per test —
    which means each individual test needs a stricter bar as n grows).
    n_tests=1 gives the familiar ~1.96 (95% single-test bar); n_tests=13
    (this session's real count as of writing) gives ~2.87.
    """
    n = max(n_tests, 1)
    return _inverse_normal_cdf(1.0 - family_alpha / (2 * n))


def evaluate_hypothesis(hypothesis: Hypothesis, baseline: dict, variant: dict) -> HypothesisResult:
    """
    `baseline`/`variant` are metrics dicts from running the standard
    pipeline (walk-forward ML + backtest) with and without the hypothesis's
    change applied — same shape as what every sport's `verify.py` already
    prints: {margin_corr, total_corr, roi_pct, roi_stderr_pct}.
    """
    margin_delta = variant["margin_corr"] - baseline["margin_corr"]
    total_delta = variant["total_corr"] - baseline["total_corr"]
    roi_delta = variant["roi_pct"] - baseline["roi_pct"]

    # Multiple-comparisons aware: the more hypotheses have already been run
    # (across ALL sports — every test we run is a chance for a fluky ROI
    # swing to look real, regardless of which sport it happened in), the
    # bigger an ROI move needs to be, relative to its own standard error,
    # before it's trusted as "clearly not noise." A flat 1-stderr bar looks
    # reasonable for one isolated test but is guaranteed to rack up false
    # positives as more tests accumulate — see the "challenge our
    # understanding of noise" discussion in decision_log.jsonl for the real
    # numbers this was checked against (13 tests in, expected <1 false
    # positive at a flat bar — this makes that arithmetic automatic instead
    # of something that has to be separately remembered and checked).
    stderr_multiplier = bonferroni_stderr_multiplier(tests_run_so_far() + 1)
    within_noise = abs(roi_delta) < baseline.get("roi_stderr_pct", float("inf")) * stderr_multiplier

    # improved/degraded are checked SEPARATELY (not just "did it move") — a
    # bug caught in real use (an NHL research pass, logged in decision_log.jsonl):
    # the old logic only asked "did ROI move," so a hypothesis that made the
    # model's actual grip on the game WORSE (both correlations dropped beyond
    # the noise floor) still got labeled "adopt_cautiously" just because ROI
    # happened to stay flat. Degraded fit is disqualifying on its own,
    # regardless of what ROI did — the whole point of checking fit at all is
    # that ROI alone isn't trustworthy enough to decide with.
    fit_improved = margin_delta > CORR_NOISE_FLOOR or total_delta > CORR_NOISE_FLOOR
    fit_degraded = margin_delta < -CORR_NOISE_FLOOR or total_delta < -CORR_NOISE_FLOOR
    fit_moved = fit_improved or fit_degraded
    suspicious = (roi_delta > 0) and not fit_improved and not within_noise

    if suspicious:
        recommendation = "reject"  # ROI improvement with no statistical basis — likely overfit to this sample
    elif fit_degraded:
        recommendation = "reject"  # model understands the game WORSE — never adopt regardless of ROI
    elif fit_improved and roi_delta >= 0:
        recommendation = "adopt"  # real improvement in game-understanding, doesn't hurt ROI
    elif not fit_moved and within_noise:
        recommendation = "adopt_cautiously"  # null result but harmless and well-motivated — same standard already applied to Pythagorean/weather/MAE-loss
    elif roi_delta < 0 and within_noise:
        recommendation = "adopt_cautiously"  # ROI dipped but within noise, real reasoning behind it, fit didn't get worse
    else:
        recommendation = "inconclusive"

    result = HypothesisResult(
        hypothesis=hypothesis, baseline=baseline, variant=variant,
        margin_corr_delta=margin_delta, total_corr_delta=total_delta,
        roi_delta_pct=roi_delta, roi_delta_within_noise=within_noise,
        suspicious=suspicious, recommendation=recommendation,
    )

    experience = (
        f"margin_corr {baseline['margin_corr']:.4f}->{variant['margin_corr']:.4f} "
        f"({margin_delta:+.4f}), total_corr {baseline['total_corr']:.4f}->{variant['total_corr']:.4f} "
        f"({total_delta:+.4f}), ROI {baseline['roi_pct']:+.2f}%->{variant['roi_pct']:+.2f}% "
        f"({roi_delta:+.2f}pp, stderr {baseline.get('roi_stderr_pct', float('nan')):.2f}pp). "
        f"Recommendation: {recommendation}."
        + (" FLAGGED SUSPICIOUS: ROI improved with no statistical-fit basis — classic overfit signature, not adopted."
           if suspicious else "")
    )
    versioning.log_decision(
        decision=f"Hypothesis test: {hypothesis.name}",
        reasoning=hypothesis.reasoning,
        experience=experience,
        sport=hypothesis.sport,
    )
    return result


# ---------------------------------------------------------------------------
# Subgroup effects: the middle ground between `evaluate_hypothesis()`'s
# binary adopt/reject and just discarding a directionally-real-but-thin
# signal. Real motivating case: the NBA spread investigation found
# favorite bets outperform underdog bets by 3.16pp ROI, stable in DIRECTION
# across both season halves (+2.82pp, then +3.54pp) but never reaching the
# z>=2.5-3 bar every actually-adopted fix in this project has cleared
# (z~1.34 on the combined sample). Neither "adopt it, ROI went up" (this
# project's whole discipline exists to prevent exactly that) nor "the
# number wasn't big enough, forget it" (throws away a real, consistently-
# signed effect just because the sample is still thin) is the right call --
# the honest middle is: use shrinkage to say precisely how much this
# should be trusted GIVEN the evidence so far, log it as a WATCHED effect
# rather than a done deal, and let it earn full adoption automatically as
# more live data accumulates and the shrinkage weight rises on its own,
# rather than a human relaxing the bar to manufacture a "win."
# ---------------------------------------------------------------------------
@dataclass
class SubgroupHypothesisResult:
    hypothesis: Hypothesis
    market: str
    shrinkage: "shrinkage.ShrinkageResult"
    z_score: float                  # raw (unshrunk) subgroup-vs-rest-of-population gap, in combined stderrs
    stderr_multiplier: float        # Bonferroni-aware bar this z_score is judged against
    stable_direction: Optional[bool]  # same-sign effect across the two provided split-halves; None if not supplied
    recommendation: str             # "adopt" | "watch" | "reject"

    def to_dict(self):
        return {
            "name": self.hypothesis.name, "sport": self.hypothesis.sport, "market": self.market,
            "reasoning": self.hypothesis.reasoning, "shrinkage": self.shrinkage.to_dict(),
            "z_score": self.z_score, "stderr_multiplier": self.stderr_multiplier,
            "stable_direction": self.stable_direction, "recommendation": self.recommendation,
        }


def evaluate_subgroup_hypothesis(hypothesis: Hypothesis, market: str, subgroup_metrics: dict,
                                  rest_metrics: dict, split_half_deltas: tuple = None,
                                  min_bets: int = 100) -> SubgroupHypothesisResult:
    """
    `subgroup_metrics`/`rest_metrics`: {roi_pct, roi_stderr_pct, bets} for
    the subgroup (e.g. NBA spread favorites) and for the REST of the
    confidence-blind population (everything NOT in the subgroup) -- never
    the whole population including the subgroup, which would dilute the
    comparison and isn't how any prior investigation in this project framed
    a subgroup split (see decision_log.jsonl's favorite-vs-underdog,
    home-vs-away, under-vs-over entries: always subgroup vs. its complement).

    `split_half_deltas`, if given: (delta_first_half, delta_second_half),
    each computed independently as subgroup_roi - rest_roi within that half
    -- required for anything beyond "reject", matching every prior fix's
    own bar (stable across a season split-half, not just significant on
    the pooled sample, which can hide a fluky single stretch).

    Recommendation logic:
      - "adopt": z-score clears the same Bonferroni-aware bar every other
        adopted fix has cleared, AND direction didn't flip across halves.
        Equivalent in strictness to evaluate_hypothesis()'s "adopt".
      - "watch": doesn't clear the bar yet, but the direction is real and
        stable (same sign in both halves) and the sample is large enough
        to not be pure noise (>= min_bets). Logged to the watchlist for
        automatic re-evaluation as more data accumulates -- NOT wired into
        live scoring at any partial strength. The `shrinkage.subgroup_weight`
        this returns IS the honest "how much to trust this today" number;
        it rises on its own as n grows and stderr shrinks, with zero manual
        tuning -- that's the actual mechanism for "change the weight as
        evidence accumulates" instead of a hand-picked dial.
      - "reject": direction flips across halves, or sample too thin to mean
        anything -- a real null, not parked for later.
    """
    raw_delta = subgroup_metrics["roi_pct"] - rest_metrics["roi_pct"]
    combined_stderr = math.sqrt(subgroup_metrics["roi_stderr_pct"] ** 2 + rest_metrics["roi_stderr_pct"] ** 2)
    stderr_mult = bonferroni_stderr_multiplier(tests_run_so_far() + 1)
    z = raw_delta / combined_stderr if combined_stderr else float("inf")
    clears_bar = abs(raw_delta) >= combined_stderr * stderr_mult

    stable_direction = None
    if split_half_deltas is not None:
        d1, d2 = split_half_deltas
        stable_direction = (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0)

    shrink_result = shrinkage.shrink_toward_baseline(
        subgroup_effect=subgroup_metrics["roi_pct"], subgroup_stderr=subgroup_metrics["roi_stderr_pct"],
        n_subgroup=subgroup_metrics["bets"],
        baseline_effect=rest_metrics["roi_pct"], baseline_stderr=rest_metrics["roi_stderr_pct"],
        n_baseline=rest_metrics["bets"],
    )

    enough_sample = subgroup_metrics["bets"] >= min_bets
    if clears_bar and stable_direction is not False:
        recommendation = "adopt"
    elif stable_direction and enough_sample:
        recommendation = "watch"
    else:
        recommendation = "reject"

    result = SubgroupHypothesisResult(
        hypothesis=hypothesis, market=market, shrinkage=shrink_result, z_score=z,
        stderr_multiplier=stderr_mult, stable_direction=stable_direction, recommendation=recommendation,
    )

    experience = (
        f"subgroup (n={subgroup_metrics['bets']}) ROI {subgroup_metrics['roi_pct']:+.2f}% vs. "
        f"rest-of-population (n={rest_metrics['bets']}) ROI {rest_metrics['roi_pct']:+.2f}% "
        f"-- raw delta {raw_delta:+.2f}pp, combined stderr {combined_stderr:.2f}pp, z={z:+.2f} "
        f"(needs {stderr_mult:.2f} to clear the current Bonferroni bar). "
        f"Split-half direction: {'stable' if stable_direction else ('flips' if stable_direction is False else 'not checked')}. "
        f"Shrinkage: subgroup gets {shrink_result.subgroup_weight*100:.0f}% weight, shrunk estimate "
        f"{shrink_result.shrunk_effect:+.2f}pp. Recommendation: {recommendation}."
    )
    versioning.log_decision(
        decision=f"Subgroup hypothesis test: {hypothesis.name}",
        reasoning=hypothesis.reasoning,
        experience=experience,
        sport=hypothesis.sport,
    )

    if recommendation == "watch":
        watchlist.add_or_update(
            name=hypothesis.name, sport=hypothesis.sport, market=market,
            shrinkage_dict=shrink_result.to_dict(), z_score=z, stable_direction=stable_direction,
            note=f"z={z:+.2f}, needs {stderr_mult:.2f}; stable across both season halves; "
                 f"re-check next research pass with more accumulated data.",
        )

    return result


def hypothesis_queue_template() -> list:
    """
    Not auto-generated combinatorial search — a curated list of specific,
    externally-motivated next experiments, added to as real ideas come up
    (research reviewed, a diagnosed weakness, a new data source). Empty by
    design until a real reasoning-backed hypothesis is added; this function
    exists so the queue has one obvious place to live and a documented
    convention, not to pre-populate speculative tests.
    """
    return []


# ---------------------------------------------------------------------------
# Held-out validation window — a real gap the app owner asked about directly:
# every hypothesis test this session (correctly) uses walk-forward,
# out-of-sample predictions, and the Bonferroni correction above raises the
# bar as more tests accumulate. But every one of those tests has still been
# scored against the SAME full historical dataset, repeatedly — the overall
# RESEARCH PROCESS has "seen" the whole history many times over, even though
# any individual prediction hasn't. Bonferroni corrects for false-positive
# RATE across tests; it doesn't create a truly untouched final check.
#
# The fix: reserve the most recent N seasons with real walk-forward
# predictions as a genuine holdout. `split_holdout()` is what every
# hypothesis-test script should use to build the `baseline`/`variant` dicts
# passed to `evaluate_hypothesis()` — computed on the ITERATION set only.
# `evaluate_holdout()` (below, and see core/holdout_check.py for the script
# that actually calls it) is the one place allowed to look at the holdout
# season, and is meant to be run rarely — once at the end of a research
# pass, not after every hypothesis — as one last honest read of whether the
# whole accumulated set of adopted changes holds up on data none of them
# were individually tuned against.
# ---------------------------------------------------------------------------
def holdout_seasons(df, season_col: str = "season", n_seasons: int = 1) -> set:
    """The most recent N seasons actually present in `df` (typically an
    oos_df / walk-forward-predictions frame). Computed dynamically rather
    than hardcoded per sport, so it stays correct as new seasons complete."""
    seasons = sorted(df[season_col].dropna().unique())
    return set(seasons[-n_seasons:]) if seasons else set()


def split_holdout(df, season_col: str = "season", n_seasons: int = 1):
    """Returns (iteration_df, holdout_df). `iteration_df` excludes the
    holdout season(s) entirely — safe to use for any number of hypothesis
    tests, no matter how many accumulate. `holdout_df` contains ONLY the
    holdout season(s) — meant to be looked at exactly once, at the end of a
    research pass, via `evaluate_holdout()` below, not per-hypothesis."""
    hs = holdout_seasons(df, season_col, n_seasons)
    return df[~df[season_col].isin(hs)].copy(), df[df[season_col].isin(hs)].copy()


def evaluate_holdout(sport: str, holdout_metrics: dict, iteration_metrics: dict) -> dict:
    """
    The one honest final check: how does the model — as it stands right
    now, with everything adopted so far — perform on the season(s) reserved
    by `split_holdout()`, compared to its performance on the seasons used
    during iteration? `holdout_metrics`/`iteration_metrics` are the same
    shape verify.py already prints: {margin_corr, total_corr, roi_pct,
    roi_stderr_pct, bets}.

    This is NOT a hypothesis test (no baseline/variant, nothing to adopt or
    reject) — it's a report. A holdout ROI within noise of the iteration
    ROI is reassuring (the research process generalized); a holdout ROI
    meaningfully worse is a real warning sign that the accumulated set of
    adopted changes fit the iteration seasons more than they found anything
    truly general — logged either way, honestly.
    """
    roi_delta = holdout_metrics["roi_pct"] - iteration_metrics["roi_pct"]
    stderr = holdout_metrics.get("roi_stderr_pct", float("inf"))
    within_noise = abs(roi_delta) < stderr * bonferroni_stderr_multiplier(1)  # single check, not accumulated

    result = {
        "sport": sport, "iteration_metrics": iteration_metrics, "holdout_metrics": holdout_metrics,
        "roi_delta_pct": roi_delta, "within_noise": within_noise,
        "warning": (not within_noise) and roi_delta < 0,
    }
    experience = (
        f"Iteration ROI {iteration_metrics['roi_pct']:+.2f}% (n={iteration_metrics.get('bets', '?')}) vs. "
        f"holdout ROI {holdout_metrics['roi_pct']:+.2f}% (n={holdout_metrics.get('bets', '?')}, "
        f"stderr {stderr:.2f}pp) — delta {roi_delta:+.2f}pp, "
        f"{'within noise' if within_noise else 'OUTSIDE noise — possible accumulated overfit'}."
    )
    versioning.log_decision(
        decision=f"Holdout check: {sport}",
        reasoning="One-time end-of-pass check of the current adopted feature set against a season never "
                  "used during any individual hypothesis test's baseline/variant comparison.",
        experience=experience,
        sport=sport,
    )
    return result
