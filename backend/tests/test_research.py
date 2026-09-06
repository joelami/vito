"""
Unit tests for core/research.py -- the hypothesis-testing discipline that
gates every real model change in this project (see its own module
docstring: enforced by shape, not left to discipline alone). These tests
protect the actual DECISION LOGIC (adopt/reject/watch boundaries,
Bonferroni multiple-comparisons awareness, the "suspicious" overfit flag)
-- the part of the codebase this project's entire credibility rests on,
and the part most at risk from a well-intentioned refactor quietly
loosening a bar "just this once."

decision_log.jsonl / subgroup_watchlist.jsonl are real, append-only
project history -- these tests redirect both to a tmp_path file (via
monkeypatch) so running the suite never adds test noise to the real files,
and so tests_run_so_far()/bonferroni bars are deterministic regardless of
how many real hypotheses have been logged since.
"""

import pytest

import versioning
from core import research, watchlist


@pytest.fixture(autouse=True)
def isolate_log_files(tmp_path, monkeypatch):
    monkeypatch.setattr(versioning, "DECISION_LOG_PATH", tmp_path / "decision_log.jsonl")
    monkeypatch.setattr(watchlist, "WATCHLIST_PATH", tmp_path / "subgroup_watchlist.jsonl")


def hyp(name="test_hypothesis", sport="NFL", reasoning=None):
    return research.Hypothesis(
        name=name, sport=sport,
        reasoning=reasoning or "A real, specific, externally-motivated reason this might help.",
    )


def metrics(margin_corr, total_corr, roi_pct, roi_stderr_pct=1.0, bets=1000):
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": roi_pct, "roi_stderr_pct": roi_stderr_pct, "bets": bets}


class TestHypothesisRequiresReasoning:
    def test_empty_reasoning_raises(self):
        with pytest.raises(ValueError):
            research.Hypothesis(name="x", reasoning="", sport="NFL")

    def test_placeholder_reasoning_raises(self):
        with pytest.raises(ValueError):
            research.Hypothesis(name="x", reasoning="try it", sport="NFL")

    def test_real_reasoning_is_accepted(self):
        h = research.Hypothesis(name="x", reasoning="A real, specific, motivated reason.", sport="NFL")
        assert h.name == "x"


class TestTestsRunSoFar:
    def test_rerunning_the_same_hypothesis_does_not_double_count(self):
        # Real bug this guards: a human (or the watchlist recheck flow)
        # re-running the SAME research script a second time with fresh
        # data must not silently inflate the Bonferroni bar for every
        # future NEW hypothesis in the project.
        baseline = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.0)
        variant = metrics(margin_corr=0.32, total_corr=0.10, roi_pct=2.5)
        research.evaluate_hypothesis(hyp(name="rerun_check"), baseline, variant)
        assert research.tests_run_so_far() == 1
        research.evaluate_hypothesis(hyp(name="rerun_check"), baseline, variant)
        assert research.tests_run_so_far() == 1   # still 1, not 2

    def test_two_distinct_hypotheses_both_count(self):
        baseline = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.0)
        variant = metrics(margin_corr=0.32, total_corr=0.10, roi_pct=2.5)
        research.evaluate_hypothesis(hyp(name="distinct_a"), baseline, variant)
        research.evaluate_hypothesis(hyp(name="distinct_b"), baseline, variant)
        assert research.tests_run_so_far() == 2

    def test_regression_test_named_entries_are_excluded(self):
        baseline = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.0)
        variant = metrics(margin_corr=0.32, total_corr=0.10, roi_pct=2.5)
        research.evaluate_hypothesis(hyp(name="regression_test_of_something"), baseline, variant)
        assert research.tests_run_so_far() == 0


class TestBonferroniStderrMultiplier:
    def test_single_test_matches_the_familiar_95pct_bar(self):
        # Documented anchor value in the function's own docstring.
        assert research.bonferroni_stderr_multiplier(1) == pytest.approx(1.96, abs=0.01)

    def test_multiplier_grows_with_more_tests(self):
        m1 = research.bonferroni_stderr_multiplier(1)
        m10 = research.bonferroni_stderr_multiplier(10)
        m100 = research.bonferroni_stderr_multiplier(100)
        assert m1 < m10 < m100

    def test_zero_or_negative_treated_as_one_test(self):
        assert research.bonferroni_stderr_multiplier(0) == research.bonferroni_stderr_multiplier(1)


class TestEvaluateHypothesis:
    def test_real_fit_improvement_with_nonneg_roi_is_adopt(self):
        baseline = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.0, roi_stderr_pct=1.0)
        variant = metrics(margin_corr=0.32, total_corr=0.10, roi_pct=2.5, roi_stderr_pct=1.0)
        result = research.evaluate_hypothesis(hyp(), baseline, variant)
        assert result.recommendation == "adopt"
        assert result.suspicious is False

    def test_fit_degraded_is_reject_even_if_roi_improved(self):
        # The exact real bug this project's own decision_log documents
        # catching: ROI looked better while the model's actual grip on the
        # game got WORSE. Must never be "adopt" regardless of ROI.
        baseline = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.0, roi_stderr_pct=1.0)
        variant = metrics(margin_corr=0.20, total_corr=0.10, roi_pct=10.0, roi_stderr_pct=1.0)
        result = research.evaluate_hypothesis(hyp(), baseline, variant)
        assert result.recommendation == "reject"

    def test_roi_jump_with_no_fit_basis_is_flagged_suspicious_and_rejected(self):
        # ROI moved a lot, fit didn't move at all, move is outside noise --
        # the classic overfit signature this project's methodology exists
        # to catch (see evaluate_hypothesis's own docstring).
        baseline = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.0, roi_stderr_pct=0.5)
        variant = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=20.0, roi_stderr_pct=0.5)
        result = research.evaluate_hypothesis(hyp(), baseline, variant)
        assert result.suspicious is True
        assert result.recommendation == "reject"

    def test_flat_null_result_within_noise_is_adopt_cautiously(self):
        baseline = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.0, roi_stderr_pct=5.0)
        variant = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.3, roi_stderr_pct=5.0)
        result = research.evaluate_hypothesis(hyp(), baseline, variant)
        assert result.recommendation == "adopt_cautiously"

    def test_small_roi_dip_within_noise_not_degraded_is_adopt_cautiously(self):
        baseline = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.0, roi_stderr_pct=5.0)
        variant = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=1.5, roi_stderr_pct=5.0)
        result = research.evaluate_hypothesis(hyp(), baseline, variant)
        assert result.recommendation == "adopt_cautiously"

    def test_logs_a_real_decision_log_entry(self, tmp_path):
        baseline = metrics(margin_corr=0.30, total_corr=0.10, roi_pct=2.0)
        variant = metrics(margin_corr=0.32, total_corr=0.10, roi_pct=2.5)
        research.evaluate_hypothesis(hyp(name="logging_check"), baseline, variant)
        logged = versioning.read_decision_log()
        assert any("logging_check" in d["decision"] for d in logged)


class TestEvaluateSubgroupHypothesis:
    def test_large_stable_effect_is_adopt(self):
        # Shape modeled on the real CFB extreme-underdog finding this
        # session validated: huge gap, tiny stderr, stable across halves.
        subgroup = {"roi_pct": -76.30, "roi_stderr_pct": 16.68, "bets": 98}
        rest = {"roi_pct": -4.05, "roi_stderr_pct": 6.06, "bets": 699}
        result = research.evaluate_subgroup_hypothesis(
            hyp(sport="CFB"), market="moneyline", subgroup_metrics=subgroup, rest_metrics=rest,
            split_half_deltas=(-66.20, -75.43), min_bets=30,
        )
        assert result.recommendation == "adopt"
        assert result.stable_direction is True

    def test_direction_flip_across_halves_is_reject_even_with_a_big_pooled_gap(self):
        subgroup = {"roi_pct": -76.30, "roi_stderr_pct": 16.68, "bets": 98}
        rest = {"roi_pct": -4.05, "roi_stderr_pct": 6.06, "bets": 699}
        result = research.evaluate_subgroup_hypothesis(
            hyp(sport="CFB"), market="moneyline", subgroup_metrics=subgroup, rest_metrics=rest,
            split_half_deltas=(-40.0, +20.0),   # opposite signs -- not stable
            min_bets=30,
        )
        assert result.stable_direction is False
        assert result.recommendation == "reject"

    def test_weak_but_stable_effect_with_enough_sample_is_watch(self):
        # Modeled on the real NBA favorite/underdog watchlist entry.
        subgroup = {"roi_pct": 2.00, "roi_stderr_pct": 2.14, "bets": 2069}
        rest = {"roi_pct": -1.16, "roi_stderr_pct": 1.00, "bets": 9463}
        result = research.evaluate_subgroup_hypothesis(
            hyp(sport="NBA"), market="spread", subgroup_metrics=subgroup, rest_metrics=rest,
            split_half_deltas=(2.82, 3.54), min_bets=100,
        )
        assert result.recommendation == "watch"

    def test_thin_sample_is_reject_not_watch(self):
        subgroup = {"roi_pct": 5.0, "roi_stderr_pct": 4.0, "bets": 10}
        rest = {"roi_pct": -1.0, "roi_stderr_pct": 1.0, "bets": 9463}
        result = research.evaluate_subgroup_hypothesis(
            hyp(sport="NBA"), market="spread", subgroup_metrics=subgroup, rest_metrics=rest,
            split_half_deltas=(4.0, 6.0), min_bets=100,
        )
        assert result.recommendation == "reject"

    def test_watch_result_is_added_to_the_watchlist(self, tmp_path):
        subgroup = {"roi_pct": 2.00, "roi_stderr_pct": 2.14, "bets": 2069}
        rest = {"roi_pct": -1.16, "roi_stderr_pct": 1.00, "bets": 9463}
        research.evaluate_subgroup_hypothesis(
            hyp(name="watchlist_check", sport="NBA"), market="spread",
            subgroup_metrics=subgroup, rest_metrics=rest,
            split_half_deltas=(2.82, 3.54), min_bets=100,
        )
        assert watchlist.WATCHLIST_PATH.exists()
        assert "watchlist_check" in watchlist.WATCHLIST_PATH.read_text()
