"""
Unit tests for core/watchlist_recheck.py -- turns core/watchlist.py's
stale-item nudge (deliberately non-auto-promoting, see that module's own
docstring) into a one-command manual re-check. These tests fake out the
actual research module imports/execution (each real one takes minutes and
needs the full historical dataset) -- what's under test is the REGISTRY
LOOKUP and RESULT-REPORTING logic, not the research itself.
"""

from unittest.mock import MagicMock

import pytest

from core import watchlist, watchlist_recheck


def stale_entry(name, sport="NBA", market="spread"):
    return {"name": name, "sport": sport, "market": market, "timestamp": "2026-01-01T00:00:00"}


class TestRunStaleRechecks:
    def test_registered_item_gets_imported_and_main_called(self, monkeypatch):
        fake_module = MagicMock()
        monkeypatch.setattr(watchlist, "stale_items", lambda max_age_days=30: [stale_entry("nba_spread_favorite_underdog_subgroup")])
        monkeypatch.setattr(watchlist_recheck.importlib, "import_module", lambda path: fake_module)

        results = watchlist_recheck.run_stale_rechecks()

        fake_module.main.assert_called_once()
        assert results == [{"name": "nba_spread_favorite_underdog_subgroup", "status": "rechecked", "error": None}]

    def test_unregistered_item_reports_no_registered_recheck_and_never_imports(self, monkeypatch):
        monkeypatch.setattr(watchlist, "stale_items", lambda max_age_days=30: [stale_entry("some_new_effect_nobody_registered")])
        import_calls = []
        monkeypatch.setattr(watchlist_recheck.importlib, "import_module", lambda path: import_calls.append(path))

        results = watchlist_recheck.run_stale_rechecks()

        assert import_calls == []   # never even tried to import
        assert results == [{"name": "some_new_effect_nobody_registered", "status": "no_registered_recheck", "error": None}]

    def test_a_failing_recheck_is_reported_not_raised_and_others_still_run(self, monkeypatch):
        ok_module = MagicMock()
        broken_module = MagicMock()
        broken_module.main.side_effect = RuntimeError("dataset unavailable")

        monkeypatch.setattr(watchlist, "stale_items", lambda max_age_days=30: [
            stale_entry("nba_spread_favorite_underdog_subgroup"),
            stale_entry("nhl_starting_goalie_individual_save_pct_total_market", sport="NHL", market="total"),
        ])

        def fake_import(path):
            return broken_module if "nba" in path else ok_module

        monkeypatch.setattr(watchlist_recheck.importlib, "import_module", fake_import)

        results = watchlist_recheck.run_stale_rechecks()

        by_name = {r["name"]: r for r in results}
        assert by_name["nba_spread_favorite_underdog_subgroup"]["status"] == "failed"
        assert "dataset unavailable" in by_name["nba_spread_favorite_underdog_subgroup"]["error"]
        assert by_name["nhl_starting_goalie_individual_save_pct_total_market"]["status"] == "rechecked"
        ok_module.main.assert_called_once()

    def test_no_stale_items_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(watchlist, "stale_items", lambda max_age_days=30: [])
        assert watchlist_recheck.run_stale_rechecks() == []

    def test_registry_entries_are_real_importable_module_paths(self):
        # A light sanity check with no mocking at all: every module path in
        # the registry must actually resolve to a real module with a
        # callable main() -- catches a typo'd path before it ever gets
        # invoked for real via the admin endpoint.
        import importlib
        for name, module_path in watchlist_recheck.RECHECK_REGISTRY.items():
            module = importlib.import_module(module_path)
            assert callable(getattr(module, "main", None)), f"{module_path} (registered for {name!r}) has no main()"
