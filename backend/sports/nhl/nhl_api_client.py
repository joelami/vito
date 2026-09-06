"""
Thin wrapper around the `nhl-api-py` package (github.com/coreyjs/nhl-api-py),
which itself wraps NHL's real, undocumented-but-public api-web.nhle.com JSON
endpoints -- the same feed NHL.com's own site consumes. No auth, no key, no
cost (`from nhlpy import NHLClient; NHLClient()` needs nothing else).

Isolates the external dependency the same way core/espn_client.py isolates
ESPN's scoreboard feed -- so feature code never imports `nhlpy` directly, and
this module is the one place that has to change if the library's shape does.
Every call is defensive: a missing/malformed response degrades to `{}`/`[]`
rather than crashing the caller, since this is (like ESPN's feed) a real but
UNDOCUMENTED, unversioned API -- treat it as unstable.

Two genuinely different data families are wrapped here, verified separately:

1. BOXSCORE (`game_center.boxscore`) -- confirmed real and populated back to
   at least the 1995-96 season (`starter: True/False` present and correctly
   set on every goalie row tested from 1995020001 through 2025020740 --
   real, recognizable goalies, real save%/TOI). This is NOT gated by any
   Edge-era limitation; it is ordinary box-score data NHL has always
   published, just never exposed through a clean per-goalie `starter` flag
   before this endpoint. Use this for anything that only needs "who
   started" or per-goalie box-score lines -- it covers this project's
   entire historical dataset range (2004-2026).

2. EDGE (`edge.*`) -- NHL's real player/puck-tracking platform (skating
   speed/distance, shot speed/location, zone time), analogous to NFL Next
   Gen Stats / MLB Statcast. CONFIRMED REAL COVERAGE, directly verified via
   `edge.goalie_landing()`'s own self-reported `seasonsWithEdgeStats` list
   (the authoritative source, not inferred from one player's history):
   ONLY seasons 20212022, 20222023, 20232024, 20242025, 20252026 (5 seasons)
   -- i.e. the 2021-22 season onward. Every season string tried before that
   (20032004 through 20202021) returns a real `ResourceNotFoundException`
   for both skater and goalie detail calls, not an empty-but-present
   response -- Edge genuinely does not exist before 2021-22, full stop.
   This is a HARD ceiling against this project's ~2004-2026 historical
   dataset: Edge can cover at most the most recent ~5 of ~21 season labels
   (~24%), the rest must fall back to a league-average/neutral value, same
   convention as every other trailing feature's cold-start handling in this
   codebase (see sports/nhl/research_starting_goalie_save_pct.py).

   A SECOND, per-player gate also exists on top of the season-level one:
   `edge.goalie_detail(player_id, season)` 404s for a real goalie who did
   play that season but below Edge's own `minimumGamesPlayed` reporting
   threshold (verified: Marc-Andre Fleury, a real MIN goaltender in
   2022-23/2023-24, 404s in both seasons despite `goalie_landing` confirming
   Edge data exists league-wide for both -- while Connor Hellebuyck, a
   full-time starter every one of those seasons, returns real data for all
   five Edge seasons). `goalie_landing(season)["minimumGamesPlayed"]` is the
   real, self-reported cutoff (25 games, confirmed for 2022-23) -- treat a
   404 on an in-Edge-era player as "below the games-played bar," not proof
   Edge itself has no data that season.

EDGE_SEASONS below is a snapshot of that live-verified list (2026-09-04) --
kept as a fast, no-network default for code that just needs "is Edge
plausible for this season," but `edge_seasons_with_data()` hits the real
endpoint and should be preferred by anything that can afford one live call,
since the league adds one new season to this list every year.
"""

from nhlpy import NHLClient
from nhlpy.http_client import ResourceNotFoundException

_client = None


def _get_client() -> NHLClient:
    global _client
    if _client is None:
        _client = NHLClient()
    return _client


# Live-verified 2026-09-04 via edge.goalie_landing()["seasonsWithEdgeStats"].
# NHL Edge simply does not exist before the 2021-22 season -- every season
# string tried before this (back through 20032004) returned a real
# ResourceNotFoundException, not an empty response, for both skater and
# goalie detail calls.
EDGE_SEASONS = frozenset({"20212022", "20222023", "20232024", "20242025", "20252026"})


def edge_seasons_with_data(force_refresh: bool = False) -> frozenset:
    """Live-checks Edge's own self-reported season coverage via
    `goalie_landing()["seasonsWithEdgeStats"]` -- the authoritative source
    (each entry also carries which `gameTypes` -- 2=regular, 3=playoffs --
    are covered, not surfaced here since both are covered for every season
    seen so far). Falls back to the `EDGE_SEASONS` snapshot above on any
    network failure rather than crashing -- this is a nice-to-have refresh,
    not something callers should ever hard-depend on being live."""
    if not force_refresh:
        return EDGE_SEASONS
    try:
        raw = _get_client().edge.goalie_landing()
        seasons = raw.get("seasonsWithEdgeStats") or []
        found = {str(s["id"]) for s in seasons if "id" in s}
        return frozenset(found) if found else EDGE_SEASONS
    except Exception as e:
        print(f"[nhl_api_client] edge_seasons_with_data() live check failed, using snapshot: {e}")
        return EDGE_SEASONS


def get_boxscore(game_id) -> dict:
    """Real NHL game_id (the standard 10-digit SSSSTTNNNN format -- e.g.
    2025020740 -- NOT this project's own synthetic `nhl_<row index>` game_id
    or the raw historical CSV's internal `game_id` column; MoneyPuck's own
    per-goalie log already uses this real id directly, see
    research_starting_goalie_save_pct.py) -> the raw boxscore dict, or {} on
    any failure (unknown/future/malformed game_id) rather than raising."""
    try:
        return _get_client().game_center.boxscore(str(game_id)) or {}
    except (ResourceNotFoundException, Exception) as e:
        print(f"[nhl_api_client] boxscore fetch failed for game_id={game_id}: {e}")
        return {}


def goalie_rows_from_boxscore(boxscore: dict) -> list:
    """
    Flattens `boxscore["playerByGameStats"].{awayTeam,homeTeam}.goalies`
    into one dict per goalie who appeared, each carrying the real `starter`
    boolean plus box-score line -- the cleaner, more direct signal this
    module exists to expose (see research_starting_goalie_save_pct.py for
    the hypothesis this replaces a same-spirit icetime-max PROXY with).
    Returns [] for a missing/malformed boxscore rather than raising.
    """
    pbs = boxscore.get("playerByGameStats") or {}
    out = []
    for side, team_key in (("away", "awayTeam"), ("home", "homeTeam")):
        team_abbrev = (boxscore.get(team_key) or {}).get("abbrev")
        for g in (pbs.get(team_key) or {}).get("goalies", []) or []:
            name = g.get("name")
            out.append({
                "game_id": boxscore.get("id"),
                "side": side,
                "team_abbrev": team_abbrev,
                "player_id": g.get("playerId"),
                "name": name.get("default") if isinstance(name, dict) else name,
                "starter": g.get("starter"),
                "toi": g.get("toi"),
                "save_pctg": g.get("savePctg"),
                "shots_against": g.get("shotsAgainst"),
                "goals_against": g.get("goalsAgainst"),
                "decision": g.get("decision"),
            })
    return out


def starting_goalie(boxscore: dict, team_abbrev: str):
    """Returns the goalie row (see `goalie_rows_from_boxscore`) flagged
    `starter: True` for the given team abbreviation in this boxscore, or
    None if not found (malformed boxscore, or -- observed on some very old
    games -- no goalie flagged starter at all)."""
    for row in goalie_rows_from_boxscore(boxscore):
        if row["team_abbrev"] == team_abbrev and row["starter"] is True:
            return row
    return None


def edge_goalie_detail(player_id, season: str, game_type: int = 2) -> dict:
    """Real per-goalie Edge shot-location/save% detail for one season. `{}`
    on any failure -- including a real goalie who played that season but
    fell below Edge's own `minimumGamesPlayed` reporting threshold (see
    module docstring; this is NOT proof Edge has no data for the season
    itself, only for this specific player)."""
    try:
        return _get_client().edge.goalie_detail(str(player_id), season=season, game_type=game_type) or {}
    except (ResourceNotFoundException, Exception) as e:
        print(f"[nhl_api_client] edge_goalie_detail failed for player_id={player_id} season={season}: {e}")
        return {}


def edge_skater_skating_speed_detail(player_id, season: str, game_type: int = 2) -> dict:
    """Real per-skater Edge top-speed detail for one season. `{}` on any
    failure (unknown player, pre-2021-22 season, below games-played bar)."""
    try:
        return _get_client().edge.skater_skating_speed_detail(str(player_id), season=season, game_type=game_type) or {}
    except (ResourceNotFoundException, Exception) as e:
        print(f"[nhl_api_client] edge_skater_skating_speed_detail failed for player_id={player_id} season={season}: {e}")
        return {}


def edge_skater_skating_distance_detail(player_id, season: str, game_type: int = 2) -> dict:
    """Real per-skater Edge total-distance-skated detail for one season. `{}`
    on any failure (see `edge_skater_skating_speed_detail`)."""
    try:
        return _get_client().edge.skater_skating_distance_detail(str(player_id), season=season, game_type=game_type) or {}
    except (ResourceNotFoundException, Exception) as e:
        print(f"[nhl_api_client] edge_skater_skating_distance_detail failed for player_id={player_id} season={season}: {e}")
        return {}


def edge_goalie_shot_location_detail(player_id, season: str, game_type: int = 2) -> dict:
    """Real per-goalie Edge shot-location-adjusted save% detail (save% broken
    out by ice-area, e.g. "Crease"/"High Slot"/"L Circle") for one season.
    `{}` on any failure."""
    try:
        return _get_client().edge.goalie_shot_location_detail(str(player_id), season=season, game_type=game_type) or {}
    except (ResourceNotFoundException, Exception) as e:
        print(f"[nhl_api_client] edge_goalie_shot_location_detail failed for player_id={player_id} season={season}: {e}")
        return {}


def daily_schedule(date: str = None) -> dict:
    """`date` is `YYYY-MM-DD`, omit for today. Returns the raw schedule dict
    (`{"games": [...], "numberOfGames": N, ...}`) or `{}` on failure. Used
    for live pre-game confirmation checks once real games are scheduled --
    NOT verifiable right now (confirmed 2026-09-04: `numberOfGames: 0`,
    NHL is in its off-season, `nextStartDate` is 2026-09-18 preseason)."""
    try:
        return _get_client().schedule.daily_schedule(date) or {}
    except Exception as e:
        print(f"[nhl_api_client] daily_schedule fetch failed for date={date}: {e}")
        return {}
