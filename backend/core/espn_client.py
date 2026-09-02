"""
Client for ESPN's public scoreboard endpoint (`site.api.espn.com`). This is
NOT a documented/contracted API — it's the same unauthenticated JSON feed
ESPN's own site and apps consume. No signup, no key, no cost. Useful beyond
just schedules: ESPN embeds real sportsbook odds (DraftKings, as of this
writing) directly on each event, including an open AND a close snapshot for
moneyline/spread/total — the same shape as the historical dataset.

Because this endpoint is undocumented, treat it as unstable: every field
access below is defensive (`.get()` chains, never a bare `[...]`), and a
malformed/missing response should degrade to "no data for this game" rather
than crash the whole sync. `harness.py` and `main.py`'s manual-game entry
form are both there specifically as fallbacks if this endpoint ever changes
shape or starts rate-limiting.

Personal/individual use only, matching the terms the historical dataset
itself was already restricted to — this pulls a small number of requests a
day, never redistributed.
"""

import json
import urllib.request
import urllib.error

from . import odds_math

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

# sport -> ESPN's path segment, for reuse across future sports
SPORT_PATHS = {
    "NFL": "football/nfl",
    "CFB": "football/college-football",
    "NBA": "basketball/nba",
    "NHL": "hockey/nhl",
    "MLB": "baseball/mlb",
}


def _get_json(url: str, timeout: float = 10.0) -> dict:
    """Shared fetch helper — every ESPN call in this module goes through
    here so the WAF workaround (see note below) only needs to live in one
    place. Returns {} on any network/parse failure rather than raising —
    callers should treat that as "skipped this run," not a fatal error."""
    try:
        # NOTE: ESPN's edge WAF 403s ANY custom User-Agent on these endpoints (tested:
        # a browser-style UA, a plain custom string — both blocked) but allows the
        # unmodified default UAs of common HTTP clients (confirmed: curl's own
        # default, urllib's own default). Deliberately sending no headers at all so
        # urllib's default "Python-urllib/x.y" goes out unmodified — do not add a
        # User-Agent override here without re-testing against the live endpoint.
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[espn_client] fetch failed for {url}: {e}")
        return {}


def fetch_scoreboard(sport: str, dates: str = None, timeout: float = 10.0) -> dict:
    """
    `dates` is either a single `YYYYMMDD` or a range `YYYYMMDD-YYYYMMDD`.
    Omit for "today" (ESPN's default).
    """
    path = SPORT_PATHS.get(sport)
    if not path:
        raise ValueError(f"unsupported sport {sport!r}, known: {list(SPORT_PATHS)}")
    url = f"{BASE_URL}/{path}/scoreboard"
    if dates:
        url += f"?dates={dates}"
    return _get_json(url, timeout)


def fetch_team_roster(sport: str, team_id: str, timeout: float = 10.0) -> dict:
    """Raw roster JSON for one team — includes each athlete's position AND
    their current `status`/`injuries` inline, no separate per-player lookup
    needed (verified: the injuries-list endpoint only returns unresolved
    $ref pointers, one extra HTTP call per injury; the roster endpoint has
    everything needed for a starter-availability check in a single call)."""
    path = SPORT_PATHS.get(sport)
    if not path:
        raise ValueError(f"unsupported sport {sport!r}, known: {list(SPORT_PATHS)}")
    url = f"{BASE_URL}/{path}/teams/{team_id}/roster"
    return _get_json(url, timeout)


def fetch_team_depth_chart(sport: str, team_id: str, timeout: float = 10.0) -> dict:
    """Raw depth-chart JSON for one team. Found and preferred over roster-listing
    order for identifying the actual starter — verified directly against a real
    case (KC's roster lists a backup QB first; the depth chart correctly has the
    real starter at the top) rather than assumed reliable."""
    path = SPORT_PATHS.get(sport)
    if not path:
        raise ValueError(f"unsupported sport {sport!r}, known: {list(SPORT_PATHS)}")
    url = f"{BASE_URL}/{path}/teams/{team_id}/depthcharts"
    return _get_json(url, timeout)


def depth_chart_starter_id(depth_chart_json: dict, position_abbr: str = "qb") -> str:
    """Returns the top-listed athlete's id at `position_abbr` (lowercase,
    e.g. "qb") across every formation group in the depth chart, or None if
    not found. The depth chart is genuinely depth-ordered (verified), unlike
    roster listing order which is not."""
    for formation in depth_chart_json.get("depthchart", []) or []:
        pos_data = (formation.get("positions") or {}).get(position_abbr.lower())
        if pos_data and pos_data.get("athletes"):
            return pos_data["athletes"][0].get("id")
    return None


def parse_key_position_status(roster_json: dict, position_abbrs: list) -> list:
    """
    Scans a roster for players at the given position(s) (e.g. `["QB"]`) and
    returns their name + roster status + any injuries, in roster-listing
    order (ESPN generally lists the starter first within a position group,
    but this is an observed convention, not a documented guarantee — treat
    the first match as "likely the starter," not certain).
    """
    out = []
    for group in roster_json.get("athletes", []) or []:
        for player in group.get("items", []) or []:
            pos = (player.get("position") or {}).get("abbreviation")
            if pos not in position_abbrs:
                continue
            status = player.get("status") or {}
            out.append({
                "id": player.get("id"),
                "name": player.get("displayName"),
                "position": pos,
                "status": status.get("name"),          # e.g. "Active", "Injured Reserve", "Out"
                "status_abbr": status.get("abbreviation"),
                "injuries": [
                    {"status": inj.get("status"), "date": inj.get("date"),
                     "short_comment": inj.get("shortComment")}
                    for inj in (player.get("injuries") or [])
                ],
            })
    return out


# Roster-level statuses that mean "not on the active roster right now" —
# distinct from a per-injury status (see OUT_INJURY_STATUSES below). A
# player can show roster status "Active" while still carrying a real
# game-day injury designation (confirmed: ESPN's roster `status` field
# tracks IR/PUP/suspension, not weekly game-status — a "Questionable"
# player still shows roster status "Active").
CONCERNING_ROSTER_STATUSES = {"Injured Reserve", "Physically Unable to Perform", "Suspension"}

# Per-injury status values that mean "probably not playing this week."
# "Questionable" is deliberately NOT in this set — it typically means a
# real chance of playing, not "likely out" — but is still surfaced via
# `questionable` on the returned dict so it isn't hidden either.
OUT_INJURY_STATUSES = {"Out", "Doubtful"}
QUESTIONABLE_INJURY_STATUSES = {"Questionable"}


def starter_out(roster_json: dict, position_abbr: str = "QB", depth_chart_json: dict = None) -> dict:
    """Returns the likely starter at `position_abbr` plus availability
    flags, or None if the position wasn't found at all. `likely_out` errs
    toward NOT flagging a merely-questionable player, to avoid crying wolf
    on the common case of a probable/questionable tag that rarely means a
    backup actually starts.

    Prefers `depth_chart_json` (real depth order) to identify WHO the
    starter is, falling back to roster-listing order only if no depth
    chart was supplied — roster order is NOT reliable for this (verified
    directly: a real case had a backup QB listed before the actual
    starter in roster order, while the depth chart had it right)."""
    players = parse_key_position_status(roster_json, [position_abbr])
    if not players:
        return None

    starter = players[0]
    if depth_chart_json:
        starter_id = depth_chart_starter_id(depth_chart_json, position_abbr.lower())
        if starter_id:
            match = next((p for p in players if p["id"] == starter_id), None)
            if match:
                starter = match

    injury_statuses = {inj["status"] for inj in starter["injuries"] if inj.get("status")}
    likely_out = (
        starter["status"] in CONCERNING_ROSTER_STATUSES
        or bool(injury_statuses & OUT_INJURY_STATUSES)
    )
    questionable = bool(injury_statuses & QUESTIONABLE_INJURY_STATUSES)
    return {**starter, "likely_out": likely_out, "questionable": questionable}


def _price(node: dict) -> float:
    """American odds string (e.g. '-142') -> decimal, or None if absent/unparseable."""
    if not node:
        return None
    raw = node.get("odds")
    if raw is None:
        return None
    try:
        return odds_math.american_to_decimal(float(raw))
    except (ValueError, TypeError):
        return None


def _line(node: dict, strip_prefix: bool = False) -> float:
    """Spread/total line string (e.g. '-7', 'o37.5') -> float, or None."""
    if not node:
        return None
    raw = node.get("line")
    if raw is None:
        return None
    if strip_prefix and raw and raw[0] in "ou":
        raw = raw[1:]
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _parse_odds_block(odds_list) -> dict:
    """
    Extracts the primary provider's odds into the same Open/Close naming
    convention `edge_finder.evaluate_game()` expects, so a synced game can
    be scored with zero translation. Missing pieces come back as None and
    `edge_finder` already skips any market with a None price.
    """
    empty = {
        "Home Odds Open": None, "Home Odds Close": None,
        "Away Odds Open": None, "Away Odds Close": None,
        "Home Line Open": None, "Home Line Close": None,
        "Home Line Odds Open": None, "Home Line Odds Close": None,
        "Away Line Odds Open": None, "Away Line Odds Close": None,
        "Total Score Open": None, "Total Score Close": None,
        "Total Score Over Open": None, "Total Score Over Close": None,
        "Total Score Under Open": None, "Total Score Under Close": None,
    }
    if not odds_list:
        return empty

    o = odds_list[0]  # ESPN sorts by provider priority; [0] is the primary book
    ml = o.get("moneyline") or {}
    sp = o.get("pointSpread") or {}
    tot = o.get("total") or {}

    return {
        "Home Odds Open": _price((ml.get("home") or {}).get("open")),
        "Home Odds Close": _price((ml.get("home") or {}).get("close")),
        "Away Odds Open": _price((ml.get("away") or {}).get("open")),
        "Away Odds Close": _price((ml.get("away") or {}).get("close")),
        "Home Line Open": _line((sp.get("home") or {}).get("open")),
        "Home Line Close": _line((sp.get("home") or {}).get("close")),
        "Home Line Odds Open": _price((sp.get("home") or {}).get("open")),
        "Home Line Odds Close": _price((sp.get("home") or {}).get("close")),
        "Away Line Odds Open": _price((sp.get("away") or {}).get("open")),
        "Away Line Odds Close": _price((sp.get("away") or {}).get("close")),
        "Total Score Open": _line((tot.get("over") or {}).get("open"), strip_prefix=True),
        "Total Score Close": _line((tot.get("over") or {}).get("close"), strip_prefix=True),
        "Total Score Over Open": _price((tot.get("over") or {}).get("open")),
        "Total Score Over Close": _price((tot.get("over") or {}).get("close")),
        "Total Score Under Open": _price((tot.get("under") or {}).get("open")),
        "Total Score Under Close": _price((tot.get("under") or {}).get("close")),
    }


def parse_events(raw: dict) -> list:
    """Returns one dict per event: identity/result fields + the Open/Close odds block."""
    out = []
    for e in raw.get("events", []) or []:
        comps = e.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        status = (comp.get("status") or {}).get("type") or {}

        def _score(c):
            v = c.get("score")
            try:
                return int(v) if v not in (None, "") else None
            except (ValueError, TypeError):
                return None

        # ESPN season.type: 1=preseason, 2=regular season, 3=postseason. The
        # historical dataset this model was trained on contains ONLY regular-season
        # and postseason games (confirmed via season row counts) — preseason games
        # (backups, different incentives, no historical analog) must never be
        # scored by this model. Callers should filter on this before scoring.
        season_type = ((e.get("season") or {}).get("type"))

        row = {
            "espn_event_id": e.get("id"),
            "date": e.get("date"),
            "season_type": season_type,
            "is_playoff": season_type == 3,
            "home_team": (home.get("team") or {}).get("displayName"),
            "away_team": (away.get("team") or {}).get("displayName"),
            "home_score": _score(home),
            "away_score": _score(away),
            "completed": bool(status.get("completed")),
            "status_state": status.get("state"),
            "is_neutral_venue": bool(comp.get("neutralSite")),
            # Real, confirmed-present field (checked directly against a live
            # CFB scoreboard call before relying on it) -- currently only
            # consumed by CFB's season_week_adj live-scoring hook (see
            # sports/cfb/features.py's extra_matchup_features()). None for
            # any event where ESPN doesn't supply it rather than a fabricated
            # default, so a missing value is never mistaken for week 0.
            "week": ((e.get("week") or {}).get("number")),
        }
        row.update(_parse_odds_block(comp.get("odds")))
        out.append(row)
    return out


def parse_probable_pitchers(raw: dict) -> dict:
    """
    MLB-specific: ESPN's scoreboard embeds each competitor's announced
    probable starting pitcher directly (`competitor.probables[0]`) —
    genuinely the same call `parse_events` already makes, no extra
    roster/depth-chart requests needed at all (unlike the QB flag, which
    needs two calls per team). This is a CONTEXT FLAG, same scope rule as
    `sports.nfl.injuries`: who's announced to start and their season ERA/
    W-L is useful to see next to a pick, but it was never in the historical
    training data, so it's surfaced, not fed into the model.

    Returns {espn_event_id: {"home": {...} | None, "away": {...} | None}}.
    A None side means ESPN hasn't posted a probable for that team yet
    (common more than a day or two out) — callers should treat that as
    "not yet announced," not an error.
    """
    out = {}
    for e in raw.get("events", []) or []:
        comps = e.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        sides = {}
        for c in comp.get("competitors", []) or []:
            home_away = c.get("homeAway")
            probables = c.get("probables") or []
            if not probables:
                sides[home_away] = None
                continue
            p = probables[0]
            athlete = p.get("athlete") or {}
            stats = {s.get("abbreviation"): s.get("displayValue") for s in (p.get("statistics") or [])}
            sides[home_away] = {
                "name": athlete.get("fullName"),
                "position": athlete.get("position"),
                "era": stats.get("ERA"),
                "wins": stats.get("W"),
                "losses": stats.get("L"),
            }
        out[e.get("id")] = {"home": sides.get("home"), "away": sides.get("away")}
    return out
