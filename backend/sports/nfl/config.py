"""
NFL-specific constants. Every constant that's "tuned" rather than derived
lives here so it's obvious what's a modeling assumption vs. a fact pulled
from data.
"""

from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent.parent / "Datasets" / "NFL" / "nfl.xlsx"

# ---------- Power rating engine ----------
ELO_K_FACTOR = 20.0           # how fast ratings move per game
ELO_START_RATING = 1500.0     # every new franchise starts here
HOME_FIELD_ADV_ELO = 48.0     # elo points added to home team's rating pre-game (~2.4 pts on scoreboard)
SEASON_REGRESSION = 0.33      # fraction a team's rating reverts toward 1500 at each new season boundary
MOV_MULT_BASE = 2.2           # margin-of-victory multiplier base (standard "Elo MOV" formula)
MOV_MULT_DIVISOR = 2.2

# Points-per-elo-point conversion, used to turn a rating differential into a
# predicted scoring margin. ~25 elo points ~= 1 point of scoring margin is a
# commonly used approximation for pro football; refined by residual-fit at
# runtime, this is just the starting scale.
ELO_POINTS_PER_MARGIN = 25.0

# ---------- Season boundaries ----------
# NFL seasons run Sept -> Feb. A game in Jan/Feb belongs to the season that
# started the previous autumn (e.g. Feb 2026 Super Bowl = "2025 season").
def season_for_date(dt) -> int:
    return dt.year if dt.month >= 7 else dt.year - 1


# ---------- Franchise continuity across relocations/rebrands ----------
# Maps every historical display name to a canonical franchise key so power
# ratings and rolling form carry through a move/rebrand (same front office,
# same roster continuity) even though the display name shown in the UI stays
# whatever that historical row actually said.
FRANCHISE_CANONICAL = {
    "Oakland Raiders": "Raiders",
    "Las Vegas Raiders": "Raiders",
    "San Diego Chargers": "Chargers",
    "Los Angeles Chargers": "Chargers",
    "St. Louis Rams": "Rams",
    "Los Angeles Rams": "Rams",
    "Washington Redskins": "Washington",
    "Washington Football Team": "Washington",
    "Washington Commanders": "Washington",
}


def canonical_franchise(display_name: str) -> str:
    return FRANCHISE_CANONICAL.get(display_name, display_name)


# ---------- Divisions (stable since 2002 realignment; data starts 2006) ----------
DIVISIONS = {
    "AFC East": ["Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets"],
    "AFC North": ["Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers"],
    "AFC South": ["Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans"],
    "AFC West": ["Denver Broncos", "Kansas City Chiefs", "Raiders", "Chargers"],
    "NFC East": ["Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington"],
    "NFC North": ["Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings"],
    "NFC South": ["Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers"],
    "NFC West": ["Arizona Cardinals", "San Francisco 49ers", "Seattle Seahawks", "Rams"],
}

TEAM_DIVISION = {}
for _div, _teams in DIVISIONS.items():
    for _t in _teams:
        TEAM_DIVISION[_t] = _div


def division_for(canonical_name: str):
    return TEAM_DIVISION.get(canonical_name)


def is_divisional_game(home_canonical: str, away_canonical: str) -> bool:
    dh = division_for(home_canonical)
    da = division_for(away_canonical)
    return dh is not None and dh == da


# ---------- Bookmaker source by date (informational; devig handles vig per-row already) ----------
def book_source_for_date(dt) -> str:
    if dt.year > 2025 or (dt.year == 2025 and dt.month >= 9):
        return "Betr"
    if dt.year > 2018 or (dt.year == 2018 and dt.month >= 9):
        return "bet365"
    if dt.year >= 2014:
        return "Pinnacle"
    return "Historical"
