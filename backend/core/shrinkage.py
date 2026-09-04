"""
Shrinkage (a.k.a. partial pooling / empirical Bayes) -- the real, principled
middle ground between "this subgroup effect is real, use it at full
strength" and "the sample is noisy, throw it away entirely."

Real gap this closes: `core/research.py`'s evaluate_hypothesis() is
deliberately binary (adopt / adopt_cautiously / reject / inconclusive),
which is the right call for a WHOLE-POPULATION feature (does this help
every game, on average) -- but it has no answer for a genuine SUBGROUP
question like "do Pacific-timezone teams specifically get hit harder by
travel." Testing 32 individual teams for that would be a real multiple-
comparisons trap (check enough subgroups and something looks significant
by chance); testing ONE pre-specified subgroup honestly (see
sports/nfl/research_pacific_travel_effect.py) avoids that trap, but still
leaves a real question once you have a subgroup's OWN estimated effect: how
much of that specific number should the model actually trust, given the
subgroup's sample is smaller and noisier than the full league?

The standard, well-established answer (inverse-variance weighting, the
same math used in fixed-effect meta-analysis): blend the subgroup's own
estimate and the league-wide baseline estimate, weighted by how PRECISE
each one is (1/variance). A subgroup with a huge, tight sample pulls the
final estimate almost entirely toward its own number; a subgroup with a
thin, noisy sample gets pulled almost entirely back to the safe baseline.
This is not a hack to sneak an unproven effect into the model at full
strength -- it's a mathematically honest way to use a real, if uncertain,
signal without letting a small sample dominate a live betting decision.
"""

from dataclasses import dataclass


@dataclass
class ShrinkageResult:
    subgroup_effect: float       # the subgroup's own raw estimate
    subgroup_stderr: float
    baseline_effect: float       # the league-wide / full-population estimate
    baseline_stderr: float
    shrunk_effect: float         # the actual number to use -- a weighted blend, not either raw input
    subgroup_weight: float       # 0.0-1.0 -- how much of the final estimate came from the subgroup's OWN data
    n_subgroup: int
    n_baseline: int

    def to_dict(self):
        return {
            "subgroup_effect": self.subgroup_effect, "subgroup_stderr": self.subgroup_stderr,
            "baseline_effect": self.baseline_effect, "baseline_stderr": self.baseline_stderr,
            "shrunk_effect": self.shrunk_effect, "subgroup_weight": round(self.subgroup_weight, 3),
            "n_subgroup": self.n_subgroup, "n_baseline": self.n_baseline,
        }


def shrink_toward_baseline(subgroup_effect: float, subgroup_stderr: float, n_subgroup: int,
                            baseline_effect: float, baseline_stderr: float, n_baseline: int) -> ShrinkageResult:
    """
    Inverse-variance-weighted blend of a subgroup's own estimate and a
    safer, larger baseline estimate. `stderr` for each should be the real
    standard error of that specific estimate (e.g. an ROI stderr from
    `backtest.summarize()`, or a regression coefficient's own stderr) --
    not guessed. A stderr of 0 is treated as "essentially infinite
    precision" (full weight to that number) since a true zero-variance
    estimate is a degenerate edge case, not something this function should
    silently divide by.

    The subgroup_weight this returns is itself the honest, reportable
    answer to "how much do we actually trust this subgroup's own signal" --
    a subgroup_weight of 0.85 means the final number is mostly the
    subgroup's own data speaking; 0.15 means it's mostly still the safe
    baseline, with the subgroup only nudging it slightly. Report this
    number alongside any shrunk effect, don't just show the blended
    number alone -- the weight IS the calibration honesty this function
    exists to provide.
    """
    sub_var = subgroup_stderr ** 2
    base_var = baseline_stderr ** 2

    if sub_var <= 0 and base_var <= 0:
        # both "infinitely precise" (degenerate) -- fall back to an even split
        # rather than a 0/0 division, since neither claim to being more
        # certain than the other is actually meaningful here.
        subgroup_weight = 0.5
    elif sub_var <= 0:
        subgroup_weight = 1.0
    elif base_var <= 0:
        subgroup_weight = 0.0
    else:
        sub_precision = 1.0 / sub_var
        base_precision = 1.0 / base_var
        subgroup_weight = sub_precision / (sub_precision + base_precision)

    shrunk = subgroup_weight * subgroup_effect + (1.0 - subgroup_weight) * baseline_effect

    return ShrinkageResult(
        subgroup_effect=subgroup_effect, subgroup_stderr=subgroup_stderr,
        baseline_effect=baseline_effect, baseline_stderr=baseline_stderr,
        shrunk_effect=shrunk, subgroup_weight=subgroup_weight,
        n_subgroup=n_subgroup, n_baseline=n_baseline,
    )


def print_shrinkage(label: str, result: ShrinkageResult) -> None:
    print(f"\n=== Shrinkage: {label} ===")
    print(f"  subgroup (n={result.n_subgroup}):  {result.subgroup_effect:+.4f} +/- {result.subgroup_stderr:.4f}")
    print(f"  baseline (n={result.n_baseline}):  {result.baseline_effect:+.4f} +/- {result.baseline_stderr:.4f}")
    print(f"  --> shrunk estimate: {result.shrunk_effect:+.4f}  "
          f"(subgroup gets {result.subgroup_weight*100:.0f}% weight, baseline gets {(1-result.subgroup_weight)*100:.0f}%)")
