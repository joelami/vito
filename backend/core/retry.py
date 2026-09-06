"""
Shared retry-with-backoff helper for this project's two real external HTTP
dependencies (core/espn_client.py, sports/nhl/nhl_api_client.py) -- both
undocumented, unauthenticated, unversioned feeds this project explicitly
treats as unstable (see each module's own docstring), yet neither had any
retry logic: a single transient timeout/connection blip/momentary 429
meant that ENTIRE sport's sync silently skipped for the rest of that day's
harness cycle (full runs happen once, sync-only once more -- there is no
"try again in five minutes" without this).

Deliberately NOT a third-party dependency (no `tenacity`/`backoff`) --
this project's own convention elsewhere (odds_math avoiding scipy,
research.py's own inverse-normal-CDF avoiding a rational-approximation
formula) is to hand-write the small, directly-verifiable version of
something like this rather than pull in a library for it.
"""

import random
import time


def with_retries(fn, *, max_attempts: int = 3, base_delay_s: float = 0.5,
                  is_retryable=lambda e: True, on_retry=None):
    """
    Calls `fn()` (no args -- callers wrap their own call in a zero-arg
    lambda/closure), retrying on failure up to `max_attempts` total tries.

    `is_retryable(exception) -> bool` lets each caller draw its own line
    between "worth retrying" (a timeout, a connection error, a 429/5xx) and
    "will never succeed no matter how many times you ask" (a 404, a
    malformed request) -- retrying the latter just wastes time and delays
    the honest "no data" result callers already handle. Defaults to
    retrying everything, since a caller that doesn't care to distinguish
    is better served by at least some retry than none.

    Exponential backoff with jitter (base_delay_s * 2**attempt, +/-25%
    random jitter so many callers hitting the same transient outage don't
    all retry in lockstep) -- attempt 0 waits ~0.5s, attempt 1 ~1s, attempt
    2 ~2s by default. Re-raises the final exception after the last attempt
    exhausts -- this function does not decide what "give up" means for the
    caller (return {}, log and skip, etc.); it only decides whether to try
    again.

    `on_retry(attempt, exception, delay_s)`, if given, is called before
    each sleep -- callers use this to log what's actually happening rather
    than retrying silently.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == max_attempts - 1 or not is_retryable(e):
                raise
            delay = base_delay_s * (2 ** attempt) * (1.0 + random.uniform(-0.25, 0.25))
            if on_retry:
                on_retry(attempt, e, delay)
            time.sleep(delay)
    raise last_exc  # unreachable in practice (the loop always returns or raises), kept for clarity
