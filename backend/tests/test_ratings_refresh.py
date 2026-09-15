"""
Regression test for a real bug found 2026-09-14: the app owner reported
"I don't see the rankings changing at all" on the Ratings tab. Root cause:
main.py's /api/ratings reads pipeline["current_ratings"] from the
in-process `_data["pipelines"]` dict, which was built exactly ONCE at
server boot and never refreshed afterward -- harness.run() rebuilt a
genuinely fresh pipeline every scheduled cycle (current_ratings included),
but discarded it rather than handing it back to anything main.py could
see. Real picks stayed fresh the whole time (pure DB operations); only
the in-memory ratings snapshot was frozen for however long the process
had been running since its last deploy.

Same verification style as tests/test_harness_reconcile.py's wiring
check, for the same reason: the bug was an absent connection between two
modules, which a function-level unit test naturally can't catch (there's
nothing to call that would exercise the missing wiring) -- reading the
real source is what actually verifies the fix stays in place.
"""

import re
from pathlib import Path


def _read(name):
    return Path(__file__).parent.parent.joinpath(name).read_text()


class TestHarnessRunReturnsThePipeline:
    def test_run_ends_with_return_pipeline(self):
        src = _read("harness.py")
        func_start = src.index("def run(sport:")
        func_body = src[func_start:]
        next_def = re.search(r"\ndef \w", func_body[1:])
        func_body = func_body[:next_def.start() + 1] if next_def else func_body
        assert "return pipeline" in func_body, (
            "harness.run() no longer returns the pipeline it built/reused -- this is the exact "
            "real incident (2026-09-14) where a fresh pipeline was silently discarded instead of "
            "being handed back to main.py's _data, freezing the Ratings tab. See this module's "
            "docstring and harness.run()'s own docstring."
        )


class TestSchedulerRefreshesMainData:
    def test_run_full_captures_and_pushes_fresh_pipelines(self):
        src = _read("scheduler.py")
        full_run_body = src.split("def _run_full(")[1].split("\ndef ")[0]
        assert "harness.run(sport" in full_run_body
        assert "main._data[\"pipelines\"][sport] = " in full_run_body, (
            "scheduler.py's _run_full() no longer pushes harness.run()'s fresh pipeline back into "
            "main._data['pipelines'] -- this is the exact real incident (2026-09-14) that froze "
            "the Ratings tab. See this module's docstring."
        )
        # The capture must happen for the SAME sport as the run that just
        # produced it, inside the per-sport loop -- not hoisted somewhere
        # it'd silently apply to the wrong sport or never fire at all.
        assert full_run_body.index("harness.run(sport") < full_run_body.index('main._data["pipelines"][sport] = ')

    def test_nfl_also_refreshes_the_flattened_legacy_data_view(self):
        # NFL-only routes (matchup scoring, backtest summary, history
        # browser) read main._data directly (main.py's own "legacy flat
        # access" comment), not through _data["pipelines"]["NFL"] -- both
        # must refresh together or those routes silently stay stale even
        # after Ratings is fixed.
        src = _read("scheduler.py")
        full_run_body = src.split("def _run_full(")[1].split("\ndef ")[0]
        assert 'main._data.update(fresh_pipeline)' in full_run_body

    def test_nfl_also_refreshes_history_opportunities(self):
        # Same staleness class as the flattened _data view above, same
        # real incident class (app owner: "is there an efficient way of
        # doing this cheap?" -- measured, yes, ~0.16s for the whole NFL
        # history, see main.py's compute_history_opportunities).
        src = _read("scheduler.py")
        full_run_body = src.split("def _run_full(")[1].split("\ndef ")[0]
        assert 'main._data["history_opportunities"] = main.compute_history_opportunities(fresh_pipeline)' in full_run_body
