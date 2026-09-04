"""
ESPN numeric team_id -> canonical franchise name, so ESPN roster/depth-chart
calls (keyed by their numeric IDs) can be joined back onto this app's own
franchise identifiers (config.canonical_franchise / ESPN_NAME_TO_ABBREV).
Fetched once from `site.api.espn.com/apis/site/v2/sports/basketball/nba/teams`
and hardcoded here since it's essentially static -- same precedent as
sports/nfl/espn_teams.py. Verified directly: mapping every one of the 30
returned display names through `config.canonical_franchise` produced zero
fallback misses (no unrecognized name silently passed through unchanged).
"""

from . import config

ESPN_TEAM_ID_TO_DISPLAY_NAME = {
    "1": "Atlanta Hawks", "2": "Boston Celtics", "3": "New Orleans Pelicans",
    "4": "Chicago Bulls", "5": "Cleveland Cavaliers", "6": "Dallas Mavericks",
    "7": "Denver Nuggets", "8": "Detroit Pistons", "9": "Golden State Warriors",
    "10": "Houston Rockets", "11": "Indiana Pacers", "12": "LA Clippers",
    "13": "Los Angeles Lakers", "14": "Miami Heat", "15": "Milwaukee Bucks",
    "16": "Minnesota Timberwolves", "17": "Brooklyn Nets", "18": "New York Knicks",
    "19": "Orlando Magic", "20": "Philadelphia 76ers", "21": "Phoenix Suns",
    "22": "Portland Trail Blazers", "23": "Sacramento Kings", "24": "San Antonio Spurs",
    "25": "Oklahoma City Thunder", "26": "Utah Jazz", "27": "Washington Wizards",
    "28": "Toronto Raptors", "29": "Memphis Grizzlies", "30": "Charlotte Hornets",
}

ESPN_TEAM_ID_TO_FRANCHISE = {
    tid: config.canonical_franchise(name) for tid, name in ESPN_TEAM_ID_TO_DISPLAY_NAME.items()
}
FRANCHISE_TO_ESPN_TEAM_ID = {fr: tid for tid, fr in ESPN_TEAM_ID_TO_FRANCHISE.items()}
