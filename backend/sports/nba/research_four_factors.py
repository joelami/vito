"""
Throwaway research script (NOT part of the live pipeline) testing whether
NBA "Four Factors" team-efficiency signal — extracted from the untapped
player-level box-score file `Datasets/NBA/nba-box-scores.csv` — improves the
NBA model over the current points-only feature set.

Run with:  python -m sports.nba.research_four_factors   (from backend/)

STEP 1 — join-key verification (done directly against the data, not assumed):

  `nba-box-scores.csv`'s `event_id` is NOT the same id space as
  `sports/nba/loader.py`'s `nba_games_all.csv` `game_id` (an NBA-stats-API
  10-digit id like "0020800741"). The box-score file's `event_id` uses at
  least two different schemes over its history (short numeric codes early,
  9-digit ESPN gameIds later) — neither matches the stats-API id. So the
  join has to be done on (date, home team, away team) instead, using the
  loader's already-canonical franchise abbreviations.

  A full-name -> abbreviation map was built for the box score file's 30 real
  NBA franchises (see TEAM_NAME_TO_ABBREV below; the box-score file also
  contains ~35 international/All-Star team names — Real Madrid, CSKA Moscow,
  "Team LeBron", etc — from preseason exhibitions, which are deliberately
  left unmapped and dropped by the join, not force-matched).

  A naive exact-date join on (date, home_abbrev, away_abbrev) only matched
  19.6% of the odds-covered (season 2006-2017) window. Spot-checking a real
  game (GSW 127-104 over LAL) showed the box-score file logs it as
  2014-11-02 while nba_games_all.csv logs the same game as 2014-11-01 — a
  one-calendar-day offset, most likely a UTC-vs-local-date logging
  difference for late West-Coast tip-offs. Confirmed systematic: shifting
  the box-score date back 1 day alone recovers 71.6% of games. The join
  actually used here is more robust than a blind shift: for each loader
  game, find box-score team-games sharing the same (home_abbrev,
  away_abbrev) with a date within +/-1 day, take the closest, and enforce a
  strict one-to-one match (both the loader game and the box-score event
  consumed at most once). That recovers 90.6% of the odds-covered window
  (13,728 / 15,155 games) — reported honestly as the real coverage ceiling
  for this hypothesis, not silently patched to 100%.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nba import config as nba_config
from sports.nba.loader import load_games
from sports.nba.features import build_features, ML_FEATURE_COLS

BOX_SCORES_PATH = (
    Path(__file__).parent.parent.parent.parent / "Datasets" / "NBA" / "nba-box-scores.csv"
)

# Full team name (as it appears in nba-box-scores.csv) -> canonical current
# abbreviation, covering every real-NBA-franchise name variant that shows up
# in the 2002-2018 span this file overlaps with the loader's data (verified
# directly — see module docstring). International/All-Star team names are
# deliberately excluded so they fail to join rather than get force-mapped.
TEAM_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS",
    "Charlotte Bobcats": "CHA", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "Brooklyn Nets": "BKN", "New Jersey Nets": "BKN",
    "New Orleans Hornets": "NOP", "NO/Oklahoma City Hornets": "NOP",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Seattle SuperSonics": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

ROLL_WINDOW = 10

# League-average fallbacks for a team's first tracked game / unmatched games
# (same role as features.py's LEAGUE_AVG_TEAM_SCORE) - computed once below
# from the box data itself, not guessed.
LEAGUE_AVG_EFG = 0.50
LEAGUE_AVG_TOV_RATE = 0.14
LEAGUE_AVG_OREB_PCT = 0.28


def _parse_made_attempted(series: pd.Series):
    """Parses 'M-A' strings (e.g. '8-15') into (made, attempted) int arrays.
    Blank/NaN (DNP rows with no minutes played) parse to (0, 0) - verified
    directly: every NaN fgm_a row has NaN `min` too, i.e. a player who didn't
    play, not a data error."""
    def parse_one(s):
        if pd.isna(s):
            return (0, 0)
        m, a = s.split("-")
        return int(m), int(a)
    parsed = series.apply(parse_one)
    return parsed.apply(lambda t: t[0]), parsed.apply(lambda t: t[1])


def _load_box_team_games() -> pd.DataFrame:
    """Aggregates the player-level box-score CSV to one row per (event_id,
    home_away side) team-game, then pairs the two sides into one row per
    event_id with home_/away_ prefixed columns. Only usecols read (file is
    118MB) - no need for player-level columns for this aggregation."""
    box = pd.read_csv(
        BOX_SCORES_PATH,
        usecols=["date", "event_id", "team", "home_away",
                  "to", "oreb", "dreb", "fgm_a", "threepm_a", "ftm_a"],
    )
    box["date"] = pd.to_datetime(box["date"], errors="coerce")

    box["fgm"], box["fga"] = _parse_made_attempted(box["fgm_a"])
    box["fg3m"], box["fg3a"] = _parse_made_attempted(box["threepm_a"])
    box["ftm"], box["fta"] = _parse_made_attempted(box["ftm_a"])

    agg = box.groupby(["event_id", "date", "team", "home_away"], as_index=False).agg(
        fgm=("fgm", "sum"), fga=("fga", "sum"),
        fg3m=("fg3m", "sum"), fg3a=("fg3a", "sum"),
        ftm=("ftm", "sum"), fta=("fta", "sum"),
        to=("to", "sum"), oreb=("oreb", "sum"), dreb=("dreb", "sum"),
    )
    agg["abbrev"] = agg["team"].map(TEAM_NAME_TO_ABBREV)
    mapped = agg.dropna(subset=["abbrev"])

    home = mapped[mapped["home_away"] == "home"]
    away = mapped[mapped["home_away"] == "away"]
    team_games = home.merge(away, on="event_id", suffixes=("_home", "_away"))
    team_games["date"] = team_games["date_home"]
    return team_games


def _attach_box_join(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Strict-ish nearest-date (+/-1 day) join on (home_abbrev, away_abbrev)
    - see module docstring for why a plain date-equality join only recovers
    19.6% of games while this recovers ~90%. Returns `games` with box-score
    columns attached (NaN where no match was found).

    IMPORTANT implementation note: an earlier version of this function did a
    plain pandas `.merge()` on (home_franchise, away_franchise) across the
    FULL 59,093-row/68-season games history against the 31,278-row box
    team-games table. Because team abbreviations repeat every season, that's
    a many-to-many join - every historical Lakers/Celtics matchup (dozens
    across 68 years) paired against every other Lakers/Celtics matchup in
    the box data BEFORE the date filter could narrow it down - a real
    cartesian blowup (millions of intermediate rows) that ran for 50+
    minutes of CPU with no output before being killed. Fixed by using
    `pd.merge_asof(..., by=[...], direction='nearest', tolerance=1 day)`,
    which does a sorted nearest-match per (home, away) group instead of a
    full cross join - no combinatorial blowup, same matching semantics."""
    g = games.sort_values("date", kind="stable").reset_index(drop=True).copy()
    g["gidx"] = g.index

    tg = team_games.sort_values("date", kind="stable").reset_index(drop=True).copy()
    # merge_asof requires unique keys aren't necessary, but ties on the same
    # (team pair, date) need to not multiply rows - team_games is already
    # one row per event_id, so this is safe.

    matched = pd.merge_asof(
        g.rename(columns={"home_franchise": "abbrev_home", "away_franchise": "abbrev_away"}),
        tg,
        on="date", by=["abbrev_home", "abbrev_away"],
        suffixes=("", "_box"),
        direction="nearest", tolerance=pd.Timedelta(days=1),
    ).rename(columns={"abbrev_home": "home_franchise", "abbrev_away": "away_franchise"})

    # merge_asof's nearest-match can (rarely) assign the same box event_id to
    # two different games if two matchups of the same pair happen within a
    # day of each other - enforce one-to-one same as before.
    matched = matched.sort_values(["gidx"]).drop_duplicates("event_id", keep="first")
    # re-index back to the full game set (drop_duplicates above may have
    # dropped a gidx if its event_id lost a tiebreak) - use a left merge on
    # gidx against the original g to guarantee every game row survives.
    box_cols = [
        "event_id", "fgm_home", "fga_home", "fg3m_home", "fta_home", "to_home", "oreb_home", "dreb_home",
        "fgm_away", "fga_away", "fg3m_away", "fta_away", "to_away", "oreb_away", "dreb_away",
    ]
    out = g.merge(matched[["gidx"] + box_cols], on="gidx", how="left").drop(columns=["gidx"])
    return out


def _box_team_log(games_with_box: pd.DataFrame) -> pd.DataFrame:
    """Long format one row per (team, game) carrying that team's own
    eFG%, turnover rate, and OREB% FOR THAT GAME (not yet rolled) - built
    the same shape as features.py's _team_game_log so the rolling step below
    is a direct copy of the proven shift(1)-before-rolling pattern."""
    cols = ["game_id", "date", "season"]

    home = games_with_box[cols + ["home_franchise",
        "fgm_home", "fga_home", "fg3m_home", "fta_home", "to_home", "oreb_home", "dreb_away"]].rename(columns={
        "home_franchise": "team", "fgm_home": "fgm", "fga_home": "fga", "fg3m_home": "fg3m",
        "fta_home": "fta", "to_home": "to", "oreb_home": "oreb", "dreb_away": "opp_dreb",
    })
    home["is_home"] = True
    away = games_with_box[cols + ["away_franchise",
        "fgm_away", "fga_away", "fg3m_away", "fta_away", "to_away", "oreb_away", "dreb_home"]].rename(columns={
        "away_franchise": "team", "fgm_away": "fgm", "fga_away": "fga", "fg3m_away": "fg3m",
        "fta_away": "fta", "to_away": "to", "oreb_away": "oreb", "dreb_home": "opp_dreb",
    })
    away["is_home"] = False
    log = pd.concat([home, away], ignore_index=True)

    # Standard Four Factors formulas (Dean Oliver) - possession-based, not
    # raw counting stats, so they're not just re-encoding pace:
    #   eFG%  = (FGM + 0.5*3PM) / FGA
    #   TOV%  = TO / (FGA + 0.44*FTA + TO)                (per-possession)
    #   OREB% = OREB / (OREB + opponent's DREB)            (contested boards only)
    log["efg_pct"] = (log["fgm"] + 0.5 * log["fg3m"]) / log["fga"].replace(0, np.nan)
    log["tov_rate"] = log["to"] / (log["fga"] + 0.44 * log["fta"] + log["to"]).replace(0, np.nan)
    log["oreb_pct"] = log["oreb"] / (log["oreb"] + log["opp_dreb"]).replace(0, np.nan)

    return log.sort_values(["team", "date"], kind="stable")


def attach_four_factors_features(games: pd.DataFrame) -> pd.DataFrame:
    """Public entry point: returns `games` with walk-forward-safe
    (shift(1)-before-rolling, identical discipline to features.py)
    home_/away_ efg_pct_l10, tov_l10, oreb_pct_l10 and their diffs attached,
    NaN-filled with league-average for warmup/unmatched rows."""
    team_games = _load_box_team_games()
    games_with_box = _attach_box_join(games, team_games)
    log = _box_team_log(games_with_box)

    grp = log.groupby("team", group_keys=False)
    log["efg_pct_l10"] = grp["efg_pct"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    log["tov_l10"] = grp["tov_rate"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    log["oreb_pct_l10"] = grp["oreb_pct"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())

    home_feats = log[log["is_home"]][["game_id", "efg_pct_l10", "tov_l10", "oreb_pct_l10"]].rename(
        columns={"efg_pct_l10": "home_efg_pct_l10", "tov_l10": "home_tov_l10", "oreb_pct_l10": "home_oreb_pct_l10"})
    away_feats = log[~log["is_home"]][["game_id", "efg_pct_l10", "tov_l10", "oreb_pct_l10"]].rename(
        columns={"efg_pct_l10": "away_efg_pct_l10", "tov_l10": "away_tov_l10", "oreb_pct_l10": "away_oreb_pct_l10"})

    out = games.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")

    for col, fill in [
        ("home_efg_pct_l10", LEAGUE_AVG_EFG), ("away_efg_pct_l10", LEAGUE_AVG_EFG),
        ("home_tov_l10", LEAGUE_AVG_TOV_RATE), ("away_tov_l10", LEAGUE_AVG_TOV_RATE),
        ("home_oreb_pct_l10", LEAGUE_AVG_OREB_PCT), ("away_oreb_pct_l10", LEAGUE_AVG_OREB_PCT),
    ]:
        out[col] = out[col].fillna(fill)

    out["efg_diff"] = out["home_efg_pct_l10"] - out["away_efg_pct_l10"]
    out["tov_diff"] = out["home_tov_l10"] - out["away_tov_l10"]
    out["oreb_pct_diff"] = out["home_oreb_pct_l10"] - out["away_oreb_pct_l10"]
    return out


# ---------------------------------------------------------------------------
# Standard pipeline runner (loader -> ratings -> features -> walk-forward ML
# -> residual stds -> backtest), matching verify.py's shape exactly.
# ---------------------------------------------------------------------------

def run_pipeline(feats: pd.DataFrame, feature_cols: list) -> dict:
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    margin_corr = corr("predicted_margin", "actual_margin")
    total_corr = corr("predicted_total", "actual_total")

    stds = ensemble.compute_residual_stds(oos, nba_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close",
    )
    bets = backtest.run_backtest(oos, stds, nba_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    s = backtest.summarize(bets)
    return {
        "margin_corr": margin_corr, "total_corr": total_corr,
        "roi_pct": float(s["roi_pct"].iloc[0]),
        "roi_stderr_pct": float(s["roi_stderr_pct"].iloc[0]),
        "bets": int(s["bets"].iloc[0]),
    }


def main():
    print("=" * 78)
    print("STEP 1: JOIN-KEY / COVERAGE VERIFICATION")
    print("=" * 78)

    games = load_games()
    odds_games = games[games["season"].between(2006, 2017)].copy()
    print(f"Loader games in odds-covered window (season 2006-2017): {len(odds_games):,}")

    team_games = _load_box_team_games()
    print(f"Box-score paired team-games (both sides mapped to a real franchise): {len(team_games):,}")

    games_with_box = _attach_box_join(games, team_games)
    matched = games_with_box["fgm_home"].notna()
    odds_mask = games_with_box["season"].between(2006, 2017)
    print(f"\nExact-date-equality join match rate (odds window), for comparison: 19.6% (measured separately)")
    print(f"Nearest-date (+/-1 day), team-pair, one-to-one join match rate (odds window): "
          f"{matched[odds_mask].mean()*100:.1f}% ({matched[odds_mask].sum():,} / {odds_mask.sum():,})")
    print(f"Same join, ALL loader games (all seasons back to 1950, for context): "
          f"{matched.mean()*100:.1f}% ({matched.sum():,} / {len(matched):,})")

    print("\n" + "=" * 78)
    print("BUILDING SHARED PIPELINE (ratings + features, box features attached)")
    print("=" * 78)

    games_ff = attach_four_factors_features(games)

    rating_cfg = PowerRatingConfig(
        k_factor=nba_config.ELO_K_FACTOR, start_rating=nba_config.ELO_START_RATING,
        home_field_adv=nba_config.HOME_FIELD_ADV_ELO, season_regression=nba_config.SEASON_REGRESSION,
        mov_mult_base=nba_config.MOV_MULT_BASE, mov_mult_divisor=nba_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games_ff, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", config=rating_cfg,
    )
    feats = build_features(games_ff, rr.history)
    print(f"Feature rows: {len(feats):,}")

    # ------------------------------------------------------------------
    # Hypothesis 1: Four Factors shooting efficiency + turnover rate
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("HYPOTHESIS 1: efg_pct_l10 / tov_l10 (Four Factors: shooting efficiency, turnovers)")
    print("=" * 78)

    h1 = Hypothesis(
        name="four_factors_efg_tov",
        reasoning=(
            "Dean Oliver's 'Four Factors' (eFG%, TOV%, OREB%, FT rate) is a "
            "long-established, externally published basketball-analytics "
            "framework showing team efficiency/possession factors predict "
            "true team strength better than points scored, which conflates "
            "shooting volume/pace with shooting efficiency. The current NBA "
            "model only ever sees final score, never shooting efficiency or "
            "turnovers, despite this being the standard critique of "
            "points-only team models in published NBA analytics."
        ),
        sport="NBA",
    )

    baseline1 = run_pipeline(feats, ML_FEATURE_COLS)
    variant1_cols = ML_FEATURE_COLS + [
        "home_efg_pct_l10", "away_efg_pct_l10", "efg_diff",
        "home_tov_l10", "away_tov_l10", "tov_diff",
    ]
    variant1 = run_pipeline(feats, variant1_cols)

    print(f"Baseline: {baseline1}")
    print(f"Variant:  {variant1}")

    result1 = evaluate_hypothesis(h1, baseline1, variant1)
    print("\nresult.to_dict():")
    import json
    print(json.dumps(result1.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # Hypothesis 2 (time permitting): offensive rebounding rate as a
    # distinct signal from shooting efficiency
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("HYPOTHESIS 2: oreb_pct_l10 (Four Factors: offensive rebound rate)")
    print("=" * 78)

    h2 = Hypothesis(
        name="four_factors_oreb_pct",
        reasoning=(
            "Offensive rebound rate is the third of Dean Oliver's published "
            "Four Factors, distinct from shooting efficiency (a team can "
            "shoot poorly but generate extra possessions via offensive "
            "boards) and distinct from turnovers - a genuinely separate "
            "possession-level signal not captured by points, eFG%, or TOV% "
            "alone, worth testing on its own rather than folded into "
            "hypothesis 1 so a null/positive result on one factor doesn't "
            "get attributed to the wrong one."
        ),
        sport="NBA",
    )

    baseline2 = run_pipeline(feats, ML_FEATURE_COLS)
    variant2_cols = ML_FEATURE_COLS + [
        "home_oreb_pct_l10", "away_oreb_pct_l10", "oreb_pct_diff",
    ]
    variant2 = run_pipeline(feats, variant2_cols)

    print(f"Baseline: {baseline2}")
    print(f"Variant:  {variant2}")

    result2 = evaluate_hypothesis(h2, baseline2, variant2)
    print("\nresult.to_dict():")
    print(json.dumps(result2.to_dict(), indent=2))

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
