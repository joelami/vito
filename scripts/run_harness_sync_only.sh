#!/bin/bash
# Second, deliberately cheap daily pass — refreshes odds and re-settles
# completed games WITHOUT rebuilding any model (no walk-forward retraining,
# no new picks snapshotted). Exists to catch a real closing price on games
# that start and finish between the morning's full run() and the next one —
# see harness.py's resync_and_settle() docstring for why that gap otherwise
# means a lot of picks never get a real captured close at all. Scheduled
# later in the day (see com.sportsbet.harness.syncOnly.plist) specifically
# so more of the day's games are near or past their start time.
set -e
cd "$(dirname "$0")/.."
python3 backend/harness.py --sync-only
