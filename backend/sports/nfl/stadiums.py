"""
Home-stadium reference data: location + roof type, keyed by canonical
franchise (see `config.FRANCHISE_CANONICAL`) and season. Used to join
historical weather onto each game (`features.py`) — a team-strength model
alone can't see whether a game was played in a blizzard or a dome.

Approximation, documented: every franchise uses its CURRENT stadium's
location/roof type for its entire history EXCEPT the four teams below that
had a genuinely different roof situation at some point in the 2006-2025
window (`STADIUM_OVERRIDES`). Getting roof type wrong is a real error (it
would feed real cold/wind data into what was actually a climate-controlled
game, or vice versa) so these are handled explicitly rather than glossed
over; getting the exact lat/long a few miles off for a team that moved
stadiums within the same metro area is not (weather is regional), so that
level of precision isn't chased everywhere.

`is_dome=True` also covers retractable-roof stadiums — teams overwhelmingly
close the roof specifically to avoid bad weather, so treating "retractable"
as "presumed closed when it matters" is the safer default; the alternative
(feeding real outdoor weather into a game actually played with the roof
shut) would inject wrong signal far more often than the reverse.
"""

# franchise (canonical) -> (lat, lon, is_dome) — current stadium, the default
CURRENT_STADIUM = {
    "Buffalo Bills": (42.7738, -78.7870, False),
    "Miami Dolphins": (25.9580, -80.2389, False),
    "New England Patriots": (42.0909, -71.2643, False),
    "New York Jets": (40.8135, -74.0745, False),
    "Baltimore Ravens": (39.2780, -76.6227, False),
    "Cincinnati Bengals": (39.0955, -84.5160, False),
    "Cleveland Browns": (41.5061, -81.6995, False),
    "Pittsburgh Steelers": (40.4468, -80.0158, False),
    "Houston Texans": (29.6847, -95.4107, True),   # retractable
    "Indianapolis Colts": (39.7601, -86.1639, True),  # retractable
    "Jacksonville Jaguars": (30.3239, -81.6373, False),
    "Tennessee Titans": (36.1665, -86.7713, False),
    "Denver Broncos": (39.7439, -105.0201, False),
    "Kansas City Chiefs": (39.0489, -94.4839, False),
    "Las Vegas Raiders": (36.0909, -115.1833, True),
    "Los Angeles Chargers": (33.9535, -118.3392, True),
    "Dallas Cowboys": (32.7473, -97.0945, True),   # retractable
    "New York Giants": (40.8135, -74.0745, False),
    "Philadelphia Eagles": (39.9008, -75.1675, False),
    "Washington": (38.9077, -76.8645, False),
    "Chicago Bears": (41.8623, -87.6167, False),
    "Detroit Lions": (42.3400, -83.0456, True),
    "Green Bay Packers": (44.5013, -88.0622, False),
    "Minnesota Vikings": (44.9735, -93.2575, True),
    "Atlanta Falcons": (33.7554, -84.4008, True),  # retractable
    "Carolina Panthers": (35.2258, -80.8528, False),
    "New Orleans Saints": (29.9511, -90.0812, True),
    "Tampa Bay Buccaneers": (27.9759, -82.5033, False),
    "Arizona Cardinals": (33.5276, -112.2626, True),  # retractable
    "Rams": (33.9535, -118.3392, True),
    "San Francisco 49ers": (37.4032, -121.9698, False),
    "Seattle Seahawks": (47.5952, -122.3316, False),
}

# franchise -> [(min_season, max_season_inclusive, lat, lon, is_dome), ...]
# only teams whose roof situation genuinely changed within 2006-2025
STADIUM_OVERRIDES = {
    "Minnesota Vikings": [
        (0, 2013, 44.9738, -93.2581, True),    # Metrodome
        (2014, 2015, 44.9760, -93.2166, False),  # TCF Bank Stadium — outdoor, cold Minneapolis games
        (2016, 9999, 44.9735, -93.2575, True),  # US Bank Stadium
    ],
    "Rams": [
        (0, 2015, 38.6328, -90.1885, True),     # Edward Jones Dome, St. Louis
        (2016, 2019, 34.0141, -118.2879, False),  # LA Memorial Coliseum — outdoor
        (2020, 9999, 33.9535, -118.3392, True),  # SoFi Stadium
    ],
    "Chargers": [
        (0, 2016, 32.7831, -117.1196, False),   # Qualcomm Stadium, San Diego — outdoor
        (2017, 2019, 33.8644, -118.2611, False),  # Dignity Health Sports Park — outdoor
        (2020, 9999, 33.9535, -118.3392, True),  # SoFi Stadium
    ],
    "Raiders": [
        (0, 2019, 37.7516, -122.2005, False),   # Oakland Coliseum — outdoor
        (2020, 9999, 36.0909, -115.1833, True),  # Allegiant Stadium
    ],
}


def stadium_for(franchise: str, season: int):
    """Returns (lat, lon, is_dome) for a franchise in a given season."""
    overrides = STADIUM_OVERRIDES.get(franchise)
    if overrides:
        for lo, hi, lat, lon, is_dome in overrides:
            if lo <= season <= hi:
                return lat, lon, is_dome
    return CURRENT_STADIUM.get(franchise, (39.8283, -98.5795, False))  # geographic-center-of-US fallback
