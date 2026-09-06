"""
Unit tests for core/retry.py -- the shared backoff helper added this
session for the two real external HTTP dependencies (ESPN, nhl-api-py)
that previously had none. Uses a fake clock (monkeypatches time.sleep) so
these tests run instantly regardless of the real backoff delays.
"""

import pytest

from core.retry import with_retries


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch):
    """Every test below deliberately exercises multi-second-scale backoff
    -- patch time.sleep out so the suite stays fast, without changing the
    retry COUNT logic being tested."""
    monkeypatch.setattr("core.retry.time.sleep", lambda seconds: None)


def flaky(fail_times, exc=RuntimeError):
    """Returns a zero-arg callable that raises `exc` the first
    `fail_times` calls, then succeeds and returns the call count."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc(f"transient failure #{calls['n']}")
        return calls["n"]
    return fn


class TestWithRetries:
    def test_succeeds_on_first_try_no_retry_needed(self):
        assert with_retries(flaky(0)) == 1

    def test_succeeds_after_transient_failures_within_budget(self):
        assert with_retries(flaky(2), max_attempts=3) == 3

    def test_exhausts_attempts_and_reraises_the_last_exception(self):
        with pytest.raises(RuntimeError, match="transient failure #3"):
            with_retries(flaky(5), max_attempts=3)

    def test_is_retryable_false_gives_up_immediately_no_retry(self):
        fn = flaky(5)
        with pytest.raises(RuntimeError, match="transient failure #1"):
            with_retries(fn, max_attempts=5, is_retryable=lambda e: False)
        # Only ever called once -- is_retryable said "don't bother."

    def test_is_retryable_only_retries_matching_exception_types(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("not retryable")
            return "should not reach here"

        with pytest.raises(ValueError):
            with_retries(fn, max_attempts=3, is_retryable=lambda e: isinstance(e, ConnectionError))
        assert calls["n"] == 1

    def test_on_retry_callback_fires_once_per_retry_not_per_final_failure(self):
        events = []
        with pytest.raises(RuntimeError):
            with_retries(
                flaky(5), max_attempts=3,
                on_retry=lambda attempt, e, delay: events.append(attempt),
            )
        # 3 attempts total -> 2 retries fire the callback (attempt 0, 1),
        # the 3rd (final) failure re-raises instead of calling on_retry.
        assert events == [0, 1]

    def test_max_attempts_one_never_retries(self):
        fn = flaky(1)
        with pytest.raises(RuntimeError):
            with_retries(fn, max_attempts=1)
