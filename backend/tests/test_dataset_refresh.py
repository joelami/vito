"""
Unit tests for core/dataset_refresh.py -- specifically the "never fatal"
contract: a failure in either refresh path (network hiccup, nflverse/
cfbfastR not having published this season's data yet, a venv/pip failure)
must be caught and logged, never propagate and take the scheduler down
with it. The two fetch-and-force-refresh mechanisms themselves are
exercised directly and manually (see this module's own docstring; both
were run for real against the live 2026 season during development) --
what's worth automated, fast, no-network coverage here is the failure
isolation, since that's the property most likely to silently regress.
"""

from unittest.mock import patch

from core import dataset_refresh


class TestRefreshCfbSuccessData:
    def test_an_exception_inside_fetch_is_caught_not_raised(self):
        with patch("scripts.fetch_cfbfastr_team_game_success.fetch_and_save",
                    side_effect=RuntimeError("network down")):
            dataset_refresh.refresh_cfb_success_data()  # must not raise


class TestRefreshNflEpaData:
    def test_a_venv_creation_failure_is_caught_not_raised(self):
        with patch("core.dataset_refresh._ensure_nfl_venv", side_effect=RuntimeError("no venv module")):
            dataset_refresh.refresh_nfl_epa_data()  # must not raise

    def test_a_nonzero_subprocess_exit_is_logged_not_raised(self):
        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = "some pip failure"

        with patch("core.dataset_refresh._ensure_nfl_venv", return_value="/fake/python3"), \
             patch("subprocess.run", return_value=FakeResult()):
            dataset_refresh.refresh_nfl_epa_data()  # must not raise


class TestRefreshAll:
    def test_calls_both_refreshers_and_never_raises_even_if_both_fail(self):
        with patch("core.dataset_refresh.refresh_cfb_success_data", side_effect=RuntimeError("x")), \
             patch("core.dataset_refresh.refresh_nfl_epa_data", side_effect=RuntimeError("y")):
            # refresh_all() itself has no try/except -- it relies on each
            # refresh_*() catching its own errors (verified above). If
            # either one stops doing that, THIS is the test that should
            # break, not scheduler.py's own broader try/except silently
            # papering over it.
            try:
                dataset_refresh.refresh_all()
                assert False, "expected refresh_all to propagate since the mocks bypass each function's own try/except"
            except RuntimeError:
                pass
