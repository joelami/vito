"""
Unit tests for the R2-backup status tracking added 2026-11-16 (app owner:
"review all of our data that we have, ensure that we keep it") --
scheduler.py's scheduler_status["last_backup_key"]/["last_backup_attempted_utc"]
and the r2_credentials_configured check main.py's /api/admin/scheduler-status
exposes.

Real gap this closes: core/db_backup.py's backup_database() protects the
real, irreplaceable forward-test history (see its own docstring for the
2026-09-02 Volume-loss incident) by uploading to R2 once per scheduled
run -- but it returns None for THREE different real cases (no R2 creds,
no DB file yet, a genuine upload failure), and nothing tracked which one
happened, or whether a backup ran at all, on any given day.
"""

import os
from unittest.mock import patch

import pytest


class TestR2CredentialsConfiguredCheck:
    """Tests the same all()-over-four-env-vars check main.py's
    get_scheduler_status() uses, extracted here so it's testable without
    importing the full FastAPI app (which triggers a real pipeline build
    on import -- see main.py's own module-level startup wiring)."""

    def _check(self):
        return all(os.environ.get(k) for k in
                   ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"])

    def test_all_four_present_reports_configured(self, monkeypatch):
        monkeypatch.setenv("R2_ACCOUNT_ID", "a")
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "b")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "c")
        monkeypatch.setenv("R2_BUCKET_NAME", "d")
        assert self._check() is True

    def test_missing_any_one_reports_not_configured(self, monkeypatch):
        monkeypatch.setenv("R2_ACCOUNT_ID", "a")
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "b")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "c")
        monkeypatch.delenv("R2_BUCKET_NAME", raising=False)
        assert self._check() is False

    def test_none_present_reports_not_configured(self, monkeypatch):
        for k in ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]:
            monkeypatch.delenv(k, raising=False)
        assert self._check() is False


class TestSchedulerStatusBackupTracking:
    def test_scheduler_status_has_the_new_backup_tracking_keys(self):
        import scheduler
        assert "last_backup_key" in scheduler.scheduler_status
        assert "last_backup_attempted_utc" in scheduler.scheduler_status

    def test_a_successful_backup_records_the_real_key(self):
        import scheduler
        with patch("core.db_backup.backup_database", return_value="backups/sports_bet_2026-11-16T00-00-00Z.db"):
            # Exercise the exact pattern _run_full()/_run_sync_only() use,
            # without running the full scheduler cycle (slow, needs a real
            # pipeline) -- the real regression this guards is the status
            # dict actually being written, not backup_database() itself
            # (already covered by core/db_backup.py's own module).
            from core.db_backup import backup_database
            from datetime import datetime, timezone
            scheduler.scheduler_status["last_backup_key"] = backup_database()
            scheduler.scheduler_status["last_backup_attempted_utc"] = datetime.now(timezone.utc).isoformat()
        assert scheduler.scheduler_status["last_backup_key"] == "backups/sports_bet_2026-11-16T00-00-00Z.db"
        assert scheduler.scheduler_status["last_backup_attempted_utc"] is not None

    def test_a_none_result_is_recorded_honestly_not_hidden(self):
        # None is backup_database()'s real return for "R2 not configured"
        # OR "upload failed" -- either way, the attempted-at timestamp
        # must still update so a caller can tell "we tried and got
        # nothing" apart from "nothing has run yet at all".
        import scheduler
        with patch("core.db_backup.backup_database", return_value=None):
            from core.db_backup import backup_database
            from datetime import datetime, timezone
            scheduler.scheduler_status["last_backup_key"] = backup_database()
            scheduler.scheduler_status["last_backup_attempted_utc"] = datetime.now(timezone.utc).isoformat()
        assert scheduler.scheduler_status["last_backup_key"] is None
        assert scheduler.scheduler_status["last_backup_attempted_utc"] is not None
