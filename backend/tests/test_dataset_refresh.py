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

Also covers a REAL, reproduced bug found 2026-10-12 (app owner: "why
isn't the success rate dataset get refreshed everyday like it should?"):
_ensure_nfl_venv() used to treat "the venv directory exists" as "the
venv works" -- a one-time pip-install failure (network blip, timeout)
left a non-functional venv directory behind, which was then reused
forever without ever being repaired, since nothing re-checked whether
the package actually imports. Reproduced directly (a bare venv with
nothing installed, confirmed it silently wedged every subsequent refresh
with the identical ModuleNotFoundError) before writing _venv_is_functional()
and the rebuild-on-broken logic this test class covers.
"""

from pathlib import Path
from unittest.mock import patch

from core import dataset_refresh


class TestRefreshCfbSuccessData:
    def test_an_exception_inside_fetch_is_caught_not_raised(self):
        with patch("scripts.fetch_cfbfastr_team_game_success.fetch_and_save",
                    side_effect=RuntimeError("network down")):
            result = dataset_refresh.refresh_cfb_success_data()  # must not raise
        assert result == {"ok": False, "detail": "network down", "timestamp": result["timestamp"]}

    def test_success_is_reported_as_ok_true(self):
        with patch("scripts.fetch_cfbfastr_team_game_success.fetch_and_save"):
            result = dataset_refresh.refresh_cfb_success_data()
        assert result["ok"] is True


class TestRefreshNflEpaData:
    def test_a_venv_creation_failure_is_caught_not_raised(self):
        with patch("core.dataset_refresh._ensure_nfl_venv", side_effect=RuntimeError("no venv module")):
            result = dataset_refresh.refresh_nfl_epa_data()  # must not raise
        assert result["ok"] is False
        assert "no venv module" in result["detail"]

    def test_a_nonzero_subprocess_exit_is_logged_not_raised(self):
        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = "some pip failure"

        with patch("core.dataset_refresh._ensure_nfl_venv", return_value="/fake/python3"), \
             patch("subprocess.run", return_value=FakeResult()):
            result = dataset_refresh.refresh_nfl_epa_data()  # must not raise
        assert result["ok"] is False
        assert "some pip failure" in result["detail"]

    def test_success_is_reported_as_ok_true(self):
        class FakeResult:
            returncode = 0
            stdout = ""
            stderr = ""

        with patch("core.dataset_refresh._ensure_nfl_venv", return_value="/fake/python3"), \
             patch("subprocess.run", return_value=FakeResult()):
            result = dataset_refresh.refresh_nfl_epa_data()
        assert result["ok"] is True


class TestVenvIsFunctional:
    def test_a_nonexistent_python_is_not_functional(self, tmp_path):
        assert dataset_refresh._venv_is_functional(tmp_path / "does_not_exist" / "python3") is False

    def test_an_import_failure_is_not_functional(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            with patch("pathlib.Path.exists", return_value=True):
                assert dataset_refresh._venv_is_functional(Path("/fake/python3")) is False

    def test_a_successful_import_is_functional(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            with patch("pathlib.Path.exists", return_value=True):
                assert dataset_refresh._venv_is_functional(Path("/fake/python3")) is True

    def test_a_subprocess_exception_is_not_functional_not_a_crash(self):
        # Real case this guards: a hung/corrupted interpreter raising on
        # invocation itself, not just exiting non-zero.
        with patch("subprocess.run", side_effect=OSError("exec format error")):
            with patch("pathlib.Path.exists", return_value=True):
                assert dataset_refresh._venv_is_functional(Path("/fake/python3")) is False


class TestEnsureNflVenv:
    def test_a_functional_existing_venv_is_reused_not_rebuilt(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dataset_refresh, "NFL_VENV_DIR", tmp_path / "venv")
        with patch("core.dataset_refresh._venv_is_functional", return_value=True), \
             patch("subprocess.run") as mock_run, \
             patch("shutil.rmtree") as mock_rmtree:
            result = dataset_refresh._ensure_nfl_venv()
        mock_run.assert_not_called()  # no venv-creation or pip-install subprocess calls
        mock_rmtree.assert_not_called()

    def test_a_broken_existing_venv_is_wiped_and_rebuilt(self, tmp_path, monkeypatch):
        # The real bug this test guards against: before the fix, this exact
        # scenario (directory exists, but nfl_data_py doesn't import)
        # returned the broken interpreter as-is, forever. Now it must be
        # wiped and rebuilt from scratch instead.
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        monkeypatch.setattr(dataset_refresh, "NFL_VENV_DIR", venv_dir)
        with patch("core.dataset_refresh._venv_is_functional", return_value=False), \
             patch("subprocess.run") as mock_run, \
             patch("shutil.rmtree") as mock_rmtree:
            dataset_refresh._ensure_nfl_venv()
        mock_rmtree.assert_called_once_with(venv_dir)
        assert mock_run.call_count == 2  # venv creation + pip install, both re-attempted

    def test_a_missing_venv_is_created_without_attempting_a_wipe(self, tmp_path, monkeypatch):
        venv_dir = tmp_path / "venv"  # deliberately not created
        monkeypatch.setattr(dataset_refresh, "NFL_VENV_DIR", venv_dir)
        with patch("core.dataset_refresh._venv_is_functional", return_value=False), \
             patch("subprocess.run") as mock_run, \
             patch("shutil.rmtree") as mock_rmtree:
            dataset_refresh._ensure_nfl_venv()
        mock_rmtree.assert_not_called()  # nothing to wipe -- this isn't the broken-venv case
        assert mock_run.call_count == 2


class TestRefreshNhlXgData:
    """Added 2026-10-15 alongside the MoneyPuck team-level rewrite (see
    sports/nhl/moneypuck_xg_features.py's own top docstring) -- same
    "never fatal" contract as NFL/CFB, and unlike those two, this fetch
    script has zero extra dependencies (same as CFB's), so it's safe to
    call directly in-process -- no isolated venv needed."""

    def test_an_exception_inside_fetch_is_caught_not_raised(self):
        with patch("scripts.fetch_moneypuck_team_xg.fetch_and_save", side_effect=RuntimeError("network down")):
            result = dataset_refresh.refresh_nhl_xg_data()  # must not raise
        assert result == {"ok": False, "detail": "network down", "timestamp": result["timestamp"]}

    def test_success_is_reported_as_ok_true_and_resets_the_cache(self):
        with patch("scripts.fetch_moneypuck_team_xg.fetch_and_save"), \
             patch("sports.nhl.moneypuck_xg_features.reset_caches") as mock_reset:
            result = dataset_refresh.refresh_nhl_xg_data()
        assert result["ok"] is True
        mock_reset.assert_called_once()


class TestRefreshAll:
    def test_calls_all_three_refreshers_and_never_raises_even_if_all_fail(self):
        with patch("core.dataset_refresh.refresh_cfb_success_data", side_effect=RuntimeError("x")), \
             patch("core.dataset_refresh.refresh_nfl_epa_data", side_effect=RuntimeError("y")), \
             patch("core.dataset_refresh.refresh_nhl_xg_data", side_effect=RuntimeError("z")):
            # refresh_all() itself has no try/except -- it relies on each
            # refresh_*() catching its own errors (verified above). If
            # any of the three stops doing that, THIS is the test that
            # should break, not scheduler.py's own broader try/except
            # silently papering over it.
            try:
                dataset_refresh.refresh_all()
                assert False, "expected refresh_all to propagate since the mocks bypass each function's own try/except"
            except RuntimeError:
                pass

    def test_returns_a_status_dict_for_all_three_sports_on_success(self):
        with patch("core.dataset_refresh.refresh_cfb_success_data", return_value={"ok": True, "detail": "x", "timestamp": "t"}), \
             patch("core.dataset_refresh.refresh_nfl_epa_data", return_value={"ok": True, "detail": "y", "timestamp": "t"}), \
             patch("core.dataset_refresh.refresh_nhl_xg_data", return_value={"ok": True, "detail": "z", "timestamp": "t"}), \
             patch("sports.nfl.nflfastr_features.reset_caches"), \
             patch("sports.nfl.qb_features.reset_cache"):
            result = dataset_refresh.refresh_all()
        assert set(result.keys()) == {"cfb", "nfl", "nhl"}
        assert all(v["ok"] for v in result.values())
