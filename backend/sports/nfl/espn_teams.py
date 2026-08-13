"""
ESPN numeric team_id -> canonical franchise name, so ESPN roster/injury
calls (keyed by their numeric IDs) can be joined back onto this app's own
franchise identifiers (see config.FRANCHISE_CANONICAL). Fetched once from
`site.api.espn.com/apis/site/v2/sports/football/nfl/teams` and hardcoded
here since it's essentially static (ESPN's team IDs don't change even when
a franchise relocates/renames, confirmed the same way team continuity was
verified for every other sport module this session).
"""

from . import config

ESPN_TEAM_ID_TO_DISPLAY_NAME = {
    "1": "Atlanta Falcons", "2": "Buffalo Bills", "3": "Chicago Bears",
    "4": "Cincinnati Bengals", "5": "Cleveland Browns", "6": "Dallas Cowboys",
    "7": "Denver Broncos", "8": "Detroit Lions", "9": "Green Bay Packers",
    "10": "Tennessee Titans", "11": "Indianapolis Colts", "12": "Kansas City Chiefs",
    "13": "Las Vegas Raiders", "14": "Los Angeles Rams", "15": "Miami Dolphins",
    "16": "Minnesota Vikings", "17": "New England Patriots", "18": "New Orleans Saints",
    "19": "New York Giants", "20": "New York Jets", "21": "Philadelphia Eagles",
    "22": "Arizona Cardinals", "23": "Pittsburgh Steelers", "24": "Los Angeles Chargers",
    "25": "San Francisco 49ers", "26": "Seattle Seahawks", "27": "Tampa Bay Buccaneers",
    "28": "Washington Commanders", "29": "Carolina Panthers", "30": "Jacksonville Jaguars",
    "33": "Baltimore Ravens", "34": "Houston Texans",
}

ESPN_TEAM_ID_TO_FRANCHISE = {
    tid: config.canonical_franchise(name) for tid, name in ESPN_TEAM_ID_TO_DISPLAY_NAME.items()
}
FRANCHISE_TO_ESPN_TEAM_ID = {fr: tid for tid, fr in ESPN_TEAM_ID_TO_FRANCHISE.items()}
