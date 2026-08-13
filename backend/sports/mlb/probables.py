"""
Probable-starting-pitcher context flag — the MLB equivalent of
`sports.nfl.injuries`'s QB-status flag, extending player-availability
surfacing beyond NFL/QB (see docs/METHODOLOGY.md's Phase 3 notes).

Different in shape from the QB flag on purpose: NFL's flag answers "is the
normal starter unavailable" (an injury question), because who starts at QB
is usually not in doubt — availability is. MLB's rotation means WHO starts
is itself the live question (every game has a different starter by design),
so the useful flag here is "who is ESPN's book announcing as the probable
starter, and how have they pitched this season" (name + ERA/W-L), not an
injury designation.

Also unlike the QB flag: no separate roster/depth-chart calls needed at
all. ESPN's scoreboard embeds each competitor's probable starter directly
(`competitor.probables[0]`) — the exact same scoreboard call
`harness.sync_espn_games` already makes for schedule/odds. This is
genuinely a superset read of data already being fetched, not a new source
of load on ESPN's API.

Scope, stated the same way as the QB flag: this is a CONTEXT FLAG shown
next to a pick, not a trained model input. The historical dataset this
model was built on has real starting-pitcher rolling-ER data (see
`starting_pitcher.py`, adopted into the live feature set via a proper
hypothesis test), but that's retrospective, walk-forward-safe attribution
from completed games — a different thing from "ESPN's live probable-starter
announcement for a game that hasn't been played yet," which has no
historical analog to have trained against.
"""

from core import espn_client


def fetch_probable_pitchers(dates: str = None) -> dict:
    """Returns {espn_event_id: {"home": {...} | None, "away": {...} | None}}
    for every game in the scoreboard window. `dates` follows the same
    ESPN YYYYMMDD or YYYYMMDD-YYYYMMDD convention as `fetch_scoreboard`;
    defaults to ESPN's own default window (today) if not given."""
    raw = espn_client.fetch_scoreboard("MLB", dates=dates)
    return espn_client.parse_probable_pitchers(raw)
