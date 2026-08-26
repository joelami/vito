"""
Runs the harness (see harness.py) on a schedule, in-process, inside the same
container/service as the API server -- instead of as a separate Railway Cron
Job service. Real reason for that choice, not just simplicity: Railway
Volumes are scoped to a single service, and the whole point of running the
harness at all is to write real picks into the SAME database the API reads
from (DB_PATH, on a persistent Volume). A separate Cron Job service would
either need its own disconnected volume -- silently wrong, the web app would
never see what the harness logged, exactly the kind of failure that looks
like success -- or depend on cross-service volume sharing this app has no
business assuming works without verifying it first. Running in-process
guarantees "same container, same filesystem, same DB_PATH" by construction.

Opt-in only (ENABLE_SCHEDULER=1): local dev keeps using launchd
(scripts/run_harness.sh + the .plist jobs) unchanged, so running `python3
main.py` locally never double-runs the harness against your own machine.
"""

import os
import threading
import time
from datetime import datetime, timezone

import harness
from core.dispatch import LIVE_SPORTS

# UTC "HH:MM" -- mirrors the two launchd jobs this replaces on Railway (see
# docs/DEPLOYMENT.md): one full run, one lighter sync-only pass later in the
# day. Overridable via env if Railway's UTC scheduling ever needs shifting.
FULL_RUN_AT = os.environ.get("SCHEDULER_FULL_RUN_UTC", "09:00")
SYNC_ONLY_AT = os.environ.get("SCHEDULER_SYNC_ONLY_UTC", "21:00")

_CHECK_INTERVAL_S = 60


def _run_parlays(snapshot_new: bool):
    """
    Real gap this closes: harness.py's own snapshot_new_parlays()/
    settle_finished_parlays() only ever ran from its __main__ block (the
    standalone `python3 harness.py` CLI path) -- which nothing on Railway
    actually invokes. Both this function's callers below now mirror
    __main__'s exact behavior: settle on every pass (a leg can finish
    between runs same as a straight pick can), snapshot new parlays only on
    a full pass (matches harness.py's `python3 harness.py --sync-only`
    skipping it too). Confirmed missing directly: a live production run via
    the admin trigger logged real straight picks but zero parlays.
    """
    try:
        settled = harness.settle_finished_parlays()
        if settled:
            print(f"[scheduler] settled {settled} parlays")
        if snapshot_new:
            logged = harness.snapshot_new_parlays()
            print(f"[scheduler] logged {logged} new parlay suggestions")
    except Exception as e:
        print(f"[scheduler] parlay snapshot/settle FAILED: {e}")


def _run_full(pipelines: dict = None):
    """
    `pipelines`, if given, is passed through to `harness.run()` so it
    reuses an already-built pipeline instead of rebuilding one from
    scratch (see `harness.run()`'s own docstring for the OOM this fixes).
    Only ever safe to pass on the ONE-TIME immediate boot-time call in
    `_loop()` below, where the pipeline was built moments ago by main.py's
    own startup and is guaranteed current — every SCHEDULED call (the
    actual 09:00/21:00 UTC timer, not the boot-time kickoff) must keep
    rebuilding fresh, or the model would silently stop retraining on newly
    synced results after the very first boot, which would be a far worse
    bug than the memory this reuse saves. Callers pass `pipelines=None`
    for every case except that one boot-time call.
    """
    for sport in LIVE_SPORTS:
        try:
            harness.run(sport, pipeline=(pipelines or {}).get(sport))
        except Exception as e:
            print(f"[scheduler] {sport} full run FAILED: {e}")
    _run_parlays(snapshot_new=True)


def _run_sync_only():
    for sport in LIVE_SPORTS:
        try:
            result = harness.resync_and_settle(sport)
            print(f"[scheduler] {sport} sync-only: {result}")
        except Exception as e:
            print(f"[scheduler] {sport} sync-only FAILED: {e}")
    _run_parlays(snapshot_new=False)


def _loop(boot_pipelines: dict = None):
    print(f"[scheduler] in-process scheduler started -- full run at {FULL_RUN_AT} UTC, "
          f"sync-only at {SYNC_ONLY_AT} UTC")

    # Run once immediately on boot -- otherwise a fresh database (a new
    # Volume, or the very first deploy) sits empty until the next scheduled
    # time, which could be most of a day away. Idempotent either way
    # (snapshot_new_picks/settle_finished_picks are both safe to re-run --
    # see harness.py), so there's no real cost to an extra boot-time pass
    # beyond the CPU it uses.
    #
    # This ONE call reuses main.py's already-built pipelines
    # (boot_pipelines) instead of asking harness.run() to rebuild every
    # sport from scratch a second time -- real fix for a real production
    # OOM crash, confirmed right after CFB (a 5th sport) pushed peak
    # post-boot memory over the edge (two full copies of every sport's
    # model transiently resident: main.py's fresh ones plus a second set
    # harness.run() used to build independently). Every OTHER call to
    # _run_full() below (the actual scheduled 09:00/21:00 UTC timer) still
    # rebuilds fresh -- see _run_full()'s own docstring for why that's
    # required, not optional.
    today = datetime.now(timezone.utc).date()
    print("[scheduler] running an immediate full pass on boot" +
          (" (reusing main.py's already-built pipelines)" if boot_pipelines else ""))
    _run_full(pipelines=boot_pipelines)
    last_full_run_date = today
    last_sync_run_date = None

    while True:
        now = datetime.now(timezone.utc)
        hhmm = now.strftime("%H:%M")
        today = now.date()
        if hhmm == FULL_RUN_AT and last_full_run_date != today:
            last_full_run_date = today
            print("[scheduler] triggering scheduled full run")
            _run_full()
        if hhmm == SYNC_ONLY_AT and last_sync_run_date != today:
            last_sync_run_date = today
            print("[scheduler] triggering scheduled sync-only pass")
            _run_sync_only()
        time.sleep(_CHECK_INTERVAL_S)


def start_background_scheduler(boot_pipelines: dict = None):
    """
    Call once from main.py's startup, passing `_data["pipelines"]` (the
    dict already built by main.py's own startup, keyed by sport) so the
    immediate boot-time harness pass (see `_loop()`) can reuse it instead
    of rebuilding every sport from scratch a second time -- see `_loop()`'s
    own docstring for the real OOM crash this fixes. No-op unless
    ENABLE_SCHEDULER=1 -- see this module's docstring for why that's
    opt-in rather than automatic.
    """
    if os.environ.get("ENABLE_SCHEDULER") != "1":
        print("[scheduler] ENABLE_SCHEDULER not set to 1 -- in-process harness scheduling disabled "
              "(expected for local dev, which uses launchd instead; see docs/DEPLOYMENT.md).")
        return
    thread = threading.Thread(target=_loop, args=(boot_pipelines,), daemon=True, name="harness-scheduler")
    thread.start()
