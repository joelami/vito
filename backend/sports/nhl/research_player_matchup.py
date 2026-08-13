"""
THROWAWAY research script — NOT part of the production pipeline. Prompted
directly by the app owner: "do certain players perform better against
certain teams and actually have an advantage?" Tests whether a team's
primary skater's own historical scoring history AGAINST THE SPECIFIC
OPPONENT being faced predicts that TEAM's game-level performance, beyond
what the existing rating-diff/rolling-form/special-teams feature set already
captures. Same spirit and structure as sports/nba/research_star_venue_split.py
(explore at a coarser grain first, THEN formalize via evaluate_hypothesis()
regardless of what exploration shows, per this project's standing practice).

DATA SOURCE, AND A REAL CORRECTION FOUND DURING THIS PASS: config.py's own
docstring flagged NHL's MoneyPuck player files as "multiple gigabytes with
their own unverified game-id join" as the reason NBA (not NHL) was used for
the earlier player-venue research. That join problem is solved directly here
(STEP 1) - but the FIRST file tried (`2008_to_2024 copy 3.csv` / `2025-2
copy.csv`, the smaller "simple" per-player-per-game format) turned out to be
a GOALIE-only MoneyPuck export, not a general player file - verified
directly, `position` is "G" for 100.0% of its 217,710 (2008-2024) rows (its
column set - unblocked_shot_attempts, freeze, playStopped - are goalie
box-score stats in hindsight, not skater ones). The real skater data
(positions C/L/R/D, confirmed directly) lives in the "detailed" file family
instead - much larger on disk (2.6GB for 2008-2024) but loads in ~6s with
`usecols` restricting to just the columns needed here, so file size alone
wasn't actually the obstacle once the right file was identified.
`Datasets/NHL/2008_to_2024 copy 2.csv` + `2025 copy.csv` are used below.

STEP 1: JOIN. MoneyPuck's `gameId` (real NHL API game ids, e.g. 2008020008)
does NOT match this project's `nhl_data_plus.csv`-derived `source_game_id`
(a different, non-NHL-API numbering scheme) - no shared key exists. Joined
instead via (home_team_id, away_team_id, date) with a merge_asof nearest-date
tolerance of 1 day per team-pair (the same "nearest-date/team-pair join"
family of technique NBA's research scripts already use), after mapping
MoneyPuck's 3-letter team codes to this project's numeric team_id via a
small, verified dict (ARI/UTA both -> Utah's canonical 129764, ATL/WPG both
-> 28, matching config.py's own franchise-continuity handling).

STEP 2: STAR IDENTIFICATION. Same walk-forward-safe definition NBA's research
used: a team's "star" for season S = the skater with the most total ice time
for that team in season S-1 (the prior season) - fixed before season S
starts, so identifying the star never uses same-season data being tested.

STEP 3: MATCHUP FEATURE. For the identified star's own career game log
(independent of which team currently employs them), an expanding (not
rolling-10 — meetings with one specific opponent are too sparse for a fixed
window) mean of individual points (goals+assists, MoneyPuck's `I_F_points`)
in ALL prior games against THIS SPECIFIC opponent, shift(1)'d so the game
being scored is never included in its own average. Requires at least one
prior meeting against that exact opponent to be non-missing; neutral-filled
(league-average points/game) otherwise, same treatment NBA's script used.

Run with:  python -m sports.nhl.research_player_matchup   (from backend/)
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
from sports.nhl import config as nhl_config
from sports.nhl.loader import load_games
from sports.nhl.features import build_features, ML_FEATURE_COLS

DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "Datasets" / "NHL"
MP_FILE_OLD = DATASETS_DIR / "2008_to_2024 copy 2.csv"   # 2008-2024, skaters, detailed per-situation format
MP_FILE_NEW = DATASETS_DIR / "2025 copy.csv"             # 2025, same format

# MoneyPuck 3-letter team code -> this project's canonical numeric team_id.
# Verified directly against the full unique playerTeam list in the raw file
# (34 codes) and sports/nhl/loader.py's cleaned team_id/team_name pairs.
# ARI and UTA both point at the same canonical id as config.py's own
# FRANCHISE_CANONICAL (Phoenix/Arizona -> Utah); ATL and WPG both point at
# the Atlanta Thrashers/Winnipeg Jets' single continuous team_id (28) the
# same way config.py's own docstring already documents that franchise.
MONEYPUCK_CODE_TO_TEAM_ID = {
    "ANA": 25, "ARI": 129764, "ATL": 28, "BOS": 1, "BUF": 2, "CAR": 7, "CBJ": 29,
    "CGY": 3, "CHI": 4, "COL": 17, "DAL": 9, "DET": 5, "EDM": 6, "FLA": 26,
    "LAK": 8, "MIN": 30, "MTL": 10, "NJD": 11, "NSH": 27, "NYI": 12, "NYR": 13,
    "OTT": 14, "PHI": 15, "PIT": 16, "SEA": 124292, "SJS": 18, "STL": 19,
    "TBL": 20, "TOR": 21, "UTA": 129764, "VAN": 22, "VGK": 37, "WPG": 28, "WSH": 23,
}


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# STEP 1: load MoneyPuck player-game rows, map team codes
# ---------------------------------------------------------------------------
def load_player_games() -> pd.DataFrame:
    cols = ["playerId", "name", "gameId", "season", "playerTeam", "opposingTeam",
            "home_or_away", "gameDate", "position", "situation", "icetime", "I_F_points"]
    old = pd.read_csv(MP_FILE_OLD, usecols=cols)
    new = pd.read_csv(MP_FILE_NEW, usecols=cols)
    raw = pd.concat([old, new], ignore_index=True)
    raw = raw[raw["situation"] == "all"].copy()   # one row per player per game (no double counting across strengths)
    raw = raw.rename(columns={"I_F_points": "points"})   # individual points = goals + assists
    raw["team_id"] = raw["playerTeam"].map(MONEYPUCK_CODE_TO_TEAM_ID)
    raw["opp_team_id"] = raw["opposingTeam"].map(MONEYPUCK_CODE_TO_TEAM_ID)
    n_unmapped = raw["team_id"].isna().sum() + raw["opp_team_id"].isna().sum()
    n_goalies = (raw["position"] == "G").sum()
    print(f"MoneyPuck player-game rows (situation=='all'): {len(raw):,}; unmapped team codes: {n_unmapped}; "
          f"goalie rows (should be 0 - this is the skater file): {n_goalies}")
    raw["date"] = pd.to_datetime(raw["gameDate"], format="%Y%m%d")
    raw = raw.dropna(subset=["team_id", "opp_team_id"]).copy()
    return raw


def build_gameid_map(mp_games: pd.DataFrame, our_games: pd.DataFrame) -> pd.DataFrame:
    """
    One row per MoneyPuck gameId -> our game_id, joined on (home_team_id,
    away_team_id, nearest date within 1 day). No shared key exists between
    the two datasets (see module docstring) - verified directly here by
    reporting the match rate rather than assuming it.
    """
    home_rows = mp_games[mp_games["home_or_away"] == "HOME"][
        ["gameId", "team_id", "opp_team_id", "date"]
    ].rename(columns={"team_id": "home_team_id", "opp_team_id": "away_team_id"})
    home_rows = home_rows.drop_duplicates(subset=["gameId"]).sort_values("date")

    ours = our_games[["game_id", "source_game_id", "home_team_id", "away_team_id", "date"]].copy()
    ours["date_only"] = ours["date"].dt.tz_localize(None).dt.normalize()
    home_rows["date_only"] = home_rows["date"].dt.normalize()
    ours = ours.sort_values("date_only")
    home_rows = home_rows.sort_values("date_only")

    merged = pd.merge_asof(
        home_rows, ours, on="date_only", by=["home_team_id", "away_team_id"],
        direction="nearest", tolerance=pd.Timedelta("1D"),
    )
    match_rate = merged["game_id"].notna().mean()
    print(f"MoneyPuck gameId -> our game_id join: {merged['game_id'].notna().sum():,}/{len(merged):,} "
          f"matched ({match_rate*100:.1f}%) via (home_team_id, away_team_id, nearest date, 1-day tolerance)")
    return merged[["gameId", "game_id"]].dropna()


# ---------------------------------------------------------------------------
# STEP 2: star identification - prior season's TOI leader per team
# ---------------------------------------------------------------------------
def identify_stars(player_games: pd.DataFrame) -> dict:
    skaters = player_games[player_games["position"] != "G"]   # defensive filter; this file has no goalie rows anyway
    toi_by_team_season = skaters.groupby(["team_id", "season", "playerId"])["icetime"].sum().reset_index()
    idx = toi_by_team_season.groupby(["team_id", "season"])["icetime"].idxmax()
    leaders = toi_by_team_season.loc[idx].set_index(["team_id", "season"])["playerId"]
    # star for (team_id, season S) = leader of season S-1
    stars = {}
    for (team_id, season), player_id in leaders.items():
        stars[(team_id, season + 1)] = player_id
    print(f"Identified {len(stars):,} (team_id, season) -> star-player assignments "
          f"(prior-season TOI leader, walk-forward safe)")
    return stars


# ---------------------------------------------------------------------------
# STEP 3: each star's own career-long, opponent-specific trailing points average
# ---------------------------------------------------------------------------
def build_star_matchup_trail(player_games: pd.DataFrame, star_player_ids: set) -> pd.DataFrame:
    log = player_games[player_games["playerId"].isin(star_player_ids)].copy()
    log = log.sort_values(["playerId", "opp_team_id", "date"], kind="stable")
    grp = log.groupby(["playerId", "opp_team_id"], group_keys=False)
    log["points_vs_opp_trail"] = grp["points"].apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
    log["n_prior_meetings_vs_opp"] = grp["points"].apply(lambda s: s.shift(1).expanding(min_periods=1).count())
    return log[["playerId", "gameId", "team_id", "opp_team_id", "date", "points",
                "points_vs_opp_trail", "n_prior_meetings_vs_opp"]]


def main():
    hr("STEP 1: LOAD + JOIN")
    our_games = load_games()
    mp_games = load_player_games()
    gid_map = build_gameid_map(mp_games, our_games)

    hr("STEP 2: STAR IDENTIFICATION")
    stars = identify_stars(mp_games)
    star_player_ids = set(stars.values())
    print(f"Unique star players across all team-seasons: {len(star_player_ids):,}")

    hr("STEP 3: BUILD OPPONENT-SPECIFIC TRAILING MATCHUP FEATURE")
    trail = build_star_matchup_trail(mp_games, star_player_ids)
    league_avg_points = float(mp_games[mp_games["position"] != "G"]["points"].mean())
    print(f"League-average skater points/game (fallback fill value): {league_avg_points:.4f}")

    # ------------------------------------------------------------------
    # EXPLORATION: direct correlation check at game level BEFORE formalizing,
    # same discipline sports/nba/research_star_venue_split.py used - if the
    # friendliest, most information-rich version of this check finds
    # nothing, a walk-forward game-level feature is unlikely to either.
    # ------------------------------------------------------------------
    hr("EXPLORATION: does a star's trailing points-vs-this-opponent correlate with the STAR'S OWN points this game?")
    has_prior = trail["n_prior_meetings_vs_opp"] >= 1
    sub = trail[has_prior]
    r_self = float(np.corrcoef(sub["points_vs_opp_trail"], sub["points"])[0, 1])
    print(f"n={len(sub):,} star-games with >=1 prior meeting vs this exact opponent")
    print(f"corr(star's trailing points-vs-opponent, star's ACTUAL points this game): {r_self:.4f}")
    print("(sanity check - if a player's own personal matchup history doesn't even predict their OWN next-game")
    print(" output against that opponent, it's very unlikely to predict anything at the TEAM level.)")

    hr("BUILDING BASE PIPELINE (loader -> power ratings -> features, current production state)")
    rating_cfg = PowerRatingConfig(
        k_factor=nhl_config.ELO_K_FACTOR, start_rating=nhl_config.ELO_START_RATING,
        home_field_adv=nhl_config.HOME_FIELD_ADV_ELO, season_regression=nhl_config.SEASON_REGRESSION,
        mov_mult_base=nhl_config.MOV_MULT_BASE, mov_mult_divisor=nhl_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        our_games, home_col="home_team_id", away_col="away_team_id",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", config=rating_cfg,
    )
    feats_base = build_features(our_games, rr.history)
    print(f"Base feature rows: {len(feats_base):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)} "
          f"(current production state, already includes adopted shot_diff_l10/pp_pct_l10/pk_pct_l10)")

    hr("STEP 4: ATTACH TEAM-LEVEL MATCHUP FEATURE VIA gameId -> game_id MAP")
    trail_mapped = trail.merge(gid_map, on="gameId", how="inner")
    print(f"Star matchup-trail rows successfully mapped to a real game_id: {len(trail_mapped):,}/{len(trail):,}")

    # For each of OUR games, look up: is the home/away team's identified star
    # (per config.py's own team_id, that team's season) present in this
    # mapped game's rows, and what's their trailing points-vs-this-opponent?
    home_side = our_games[["game_id", "season", "home_team_id"]].copy()
    home_side["star_player_id"] = home_side.apply(
        lambda r: stars.get((r["home_team_id"], r["season"])), axis=1)
    away_side = our_games[["game_id", "season", "away_team_id"]].copy()
    away_side["star_player_id"] = away_side.apply(
        lambda r: stars.get((r["away_team_id"], r["season"])), axis=1)

    # merge_asof's nearest-within-1-day join can (rarely) map two different
    # MoneyPuck gameIds onto the same our-side game_id (verified: 215/20,661,
    # ~1% - almost certainly isolated date-recording glitches on one side or
    # the other, not systematic mis-joins) - collapsed via groupby-mean
    # (nearly identical values in practice, same player/opponent on
    # consecutive real dates) so the lookup below always returns a scalar,
    # never a Series (which would otherwise silently propagate as NaN once
    # cast to a clean numeric dtype - caught by inspecting dtype directly).
    trail_by_game_player = trail_mapped.groupby(["game_id", "playerId"])["points_vs_opp_trail"].mean()

    def _lookup(row):
        pid = row["star_player_id"]
        if pd.isna(pid):
            return np.nan
        key = (row["game_id"], pid)
        return trail_by_game_player.get(key, np.nan)

    # .apply(axis=1) over a dict-lookup can leave the result dtype='object'
    # (a mix of Python float('nan') and numpy.float64) even when every value
    # is numeric - pd.to_numeric forces a clean float64 column so downstream
    # np.corrcoef/sklearn calls don't choke on the object dtype.
    home_side["home_star_vs_opp_pts_trail"] = pd.to_numeric(home_side.apply(_lookup, axis=1), errors="coerce")
    away_side["away_star_vs_opp_pts_trail"] = pd.to_numeric(away_side.apply(_lookup, axis=1), errors="coerce")

    coverage_home = home_side["home_star_vs_opp_pts_trail"].notna().mean()
    coverage_away = away_side["away_star_vs_opp_pts_trail"].notna().mean()
    print(f"Coverage (non-neutral-filled): home {coverage_home*100:.1f}%, away {coverage_away*100:.1f}% of games")

    feats = feats_base.merge(home_side[["game_id", "home_star_vs_opp_pts_trail"]], on="game_id", how="left")
    feats = feats.merge(away_side[["game_id", "away_star_vs_opp_pts_trail"]], on="game_id", how="left")
    feats["home_star_vs_opp_pts_trail"] = feats["home_star_vs_opp_pts_trail"].fillna(league_avg_points)
    feats["away_star_vs_opp_pts_trail"] = feats["away_star_vs_opp_pts_trail"].fillna(league_avg_points)

    hr("STEP 5: GAME-LEVEL EXPLORATION - does this feature correlate with actual TEAM margin?")
    has_home_cov = feats["game_id"].isin(home_side.loc[home_side["home_star_vs_opp_pts_trail"].notna(), "game_id"])
    r_margin = float(np.corrcoef(
        feats.loc[has_home_cov, "home_star_vs_opp_pts_trail"], feats.loc[has_home_cov, "actual_margin"])[0, 1])
    print(f"corr(home_star_vs_opp_pts_trail, actual_margin), n={has_home_cov.sum():,}: {r_margin:.4f}")

    import json

    def run_pipeline(feature_cols):
        wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
        oos = feats.set_index("game_id").join(wf.predictions, how="inner")
        margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
        total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])
        stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
        ens_cfg = ensemble.EnsembleConfig()
        bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
        bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
        if bets.empty:
            return {"margin_corr": margin_corr, "total_corr": total_corr,
                    "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "n_bets": 0}
        summary = backtest.summarize(bets)
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float(summary["roi_pct"].iloc[0]),
                "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
                "n_bets": int(summary["bets"].iloc[0])}

    hr("BASELINE RUN (current production ML_FEATURE_COLS)")
    baseline = run_pipeline(ML_FEATURE_COLS)
    print(baseline)

    hr("HYPOTHESIS: adding star's trailing points-vs-this-specific-opponent")
    feature_cols_matchup = ML_FEATURE_COLS + ["home_star_vs_opp_pts_trail", "away_star_vs_opp_pts_trail"]
    variant = run_pipeline(feature_cols_matchup)
    print(variant)

    h = Hypothesis(
        name="nhl_star_player_opponent_matchup",
        reasoning=(
            "Direct app-owner question: do certain players perform better against certain teams, in a way "
            "that carries real signal for the TEAM-level model? A player having a personal scoring history "
            "against a specific opponent (a 'David vs. Goliath' matchup narrative, or a real goaltender-"
            "specific shooting familiarity effect) is a commonly cited hockey narrative worth testing "
            "empirically rather than assuming either way. Operationalized the same walk-forward-safe way "
            "sports/nba/research_star_venue_split.py's star-player research did: 'star' = each team's prior-"
            "season ice-time leader (fixed before the season starts, non-circular), and this specific feature "
            "is that star's own career-to-date, opponent-specific expanding average individual points "
            "(goals+assists, shift(1)'d - the game being scored is never included in its own average) - "
            "distinct from every existing feature in ML_FEATURE_COLS, none of which are player-level or "
            "opponent-specific."
        ),
        sport="NHL",
    )
    result = evaluate_hypothesis(h, baseline, variant)
    hr("HYPOTHESIS RESULT (evaluate_hypothesis().to_dict())")
    print(json.dumps(result.to_dict(), indent=2))

    hr("DONE")


if __name__ == "__main__":
    main()
