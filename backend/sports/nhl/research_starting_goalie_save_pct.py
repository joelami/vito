"""
THROWAWAY research script -- NOT part of the production pipeline, not
imported by anything else. Tests a hypothesis explicitly distinct from
`sports/nhl/research_goalie_pk.py`'s already-tested-and-rejected
`nhl_trailing_save_pct` (see decision_log.jsonl, 2026-08-12,
"adopt_cautiously" -- fit moved less than CORR_NOISE_FLOOR in both
directions, a clean null, NOT wired into features.py per that module's
docstring).

That prior test used TEAM-level rolling save percentage -- a blend of
whichever two goalies happened to play for a team over its last 10 games,
with no individual identity at all (`sports/nhl/loader.py`'s box-score
source, `Datasets/NHL/archive/nhl_data_plus.csv`, carries only team-level
shots/goals-against, never which specific goalie was between the pipes).
Its own reasoning text says as much: "hockey analytics treats 'hot/cold
goaltending' ... as one of the sport's largest sources of short-term
variance ... in the absence of goalie-specific start data" -- i.e. that
test was a deliberate, acknowledged proxy for a signal nobody had actually
built the real version of yet.

THIS test is the real version: tonight's SPECIFIC starting goalie's own
individually-identified trailing quality, tested specifically against the
TOTAL market (not moneyline/spread) -- a distinct hypothesis from team-level
form, motivated by the same "hot/cold goaltending" mechanism but now
isolating the actual individual signal hockey analysts mean when they say it
(a team can carry two goalies of very different true quality; blending them
together, as the team-level test did, dilutes exactly the signal this
re-test isolates).

-----------------------------------------------------------------------
STEP 1 FINDING -- per-goalie historical data DOES exist, but not in the
file `sports/nhl/loader.py` reads. It lives in a separate MoneyPuck-style
per-player-per-game shot log:
  Datasets/NHL/2008_to_2024 copy 3.csv   (2008-10-04 .. 2024, goalie rows only)
  Datasets/NHL/2025-2 copy.csv           (2025-10-07 .. 2026-04-16, goalie rows only)
(identical content also duplicated under "Datasets/NHL/Game by Game Data
copy/2008_to_2024.csv" and ".../2025.csv" -- verified byte-identical via
DataFrame.equals(), so reading either location is equivalent; the two
"copy N" files above were used here since they're pre-filtered to
position=="G" already, at 1/5th the read cost of the un-filtered duplicates
and full-impact-metric siblings sitting alongside them, which carry
skater-only or season-aggregate shapes with no gameId and were not used).

Each row is one (playerId, gameId, situation) combination with `position`
("G" isolates goalies), `icetime`, `goals` (goals allowed), `ongoal` (shots
faced) -- i.e. real, individually-attributed shot-stopping data, NOT a
mirror of the team box score. Verified directly on a real game (ANA 5, LAK
4, 2025-11-28, gameId 2025020384): team-level goals_against split into
Husso (ANA) allowing 4 (matches ANA's box-score goals-against exactly) and
Kuemper (LAK) allowing 4 -- one goal LESS than LAK's team goals-against (5),
because the 5th goal was an empn-net goal scored after LAK pulled Kuemper
for an extra attacker, so it was never charged to him individually. That
mismatch is exactly the proof this is real individual attribution, not a
recomputation of the same team number research_goalie_pk.py already used.

No "started" flag exists in the raw data, so the starting goalie for a
given (gameId, team) is inferred as the goalie with the MAX `icetime` among
that team's goalie rows for that game (situation=="all") -- a standard,
defensible proxy (a starter who isn't pulled early plays the large majority
of a game; a reliever who comes in after a pull necessarily has less ice
time than the starter had already accumulated in the modal case). Verified
unambiguous: grouping by (gameId, playerTeam) produces exactly one row per
group after `idxmax` with zero groups needing a tie-break.

JOIN MECHANICS (real, non-trivial, documented since it silently failed at
first): `sports/nhl/loader.py`'s `games["date"]` is UTC and pushes evening
North American games into the next UTC calendar day (e.g. a 7pm ET game
lands at ~00:00 UTC); MoneyPuck's `gameDate` is the LOCAL schedule date.
Naively joining on (games["date"].dt.date, team abbreviation) hits only
33% of games. Shifting the UTC date back one day whenever the UTC hour is
< 12 (covers every US/Canada evening start across all time zones and DST
states) raises the match rate to 93.2% for games on/after 2008-10-04 (the
earliest MoneyPuck goalie-log date) -- the remaining ~6.8% misses are
98.7% playoff games specifically (MoneyPuck's per-game goalie log has much
thinner playoff coverage than regular season), a clean, understood gap
rather than a join bug, filled with the league-average save% like every
other trailing feature's cold-start rows. Team-name -> 3-letter-abbreviation
mapping (`NAME_TO_ABBREV` below) is built directly from the 34 distinct
(team_id, team_name) pairs in `sports/nhl/config.py`'s REAL_TEAM_IDS
universe, keyed on the RAW per-season display name (not the canonicalized
numeric team_id) so the Coyotes->Mammoth and Thrashers->Jets relocations
resolve to the correct MoneyPuck abbreviation (ARI vs UTA, ATL vs WPG)
automatically, without extra date-range logic, since MoneyPuck itself
switches abbreviation at the same real-world rename boundary.

LEAKAGE: `goals`/`ongoal` on a goalie's OWN game row are that game's own
outcome (same leakage family as every other box-score column in this
project -- see research_goalie_pk.py's leakage_check). The per-goalie
trailing save_pct_l10 below is shift(1)'d on the GOALIE's own chronological
appearance sequence (not the team's) before rolling, so a game's feature
value only reflects that specific goalie's OWN games strictly before this
one -- the individual-identity analogue of every shift(1)-before-rolling
convention already in features.py.

Rolled over the goalie's last 10 appearances of ANY role (start or relief)
with icetime>0 and ongoal>0 -- a deliberate, documented choice: restricting
to starts-only would need the same max-icetime proxy applied retroactively
to every one of a goalie's past games too, adding a second layer of
inference on top of the first for a real but second-order refinement; using
all appearances is the simpler, more defensible baseline and still measures
the same underlying "how well has this specific human been stopping pucks
lately" signal a broadcast graphic means by "hot goaltending."

Run with:  python -m sports.nhl.research_starting_goalie_save_pct   (from backend/)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest, research
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nhl import config as nhl_config
from sports.nhl.loader import load_games
from sports.nhl.features import build_features, ML_FEATURE_COLS

ROLL_WINDOW = 10

# Raw per-season team_name (as loader.py preserves it, pre-canonicalization)
# -> MoneyPuck's 3-letter team abbreviation. Built directly from the 34
# distinct (team_id, team_name) pairs enumerated in sports/nhl/config.py's
# REAL_TEAM_IDS universe (verified 1:1 against the 34 distinct abbreviations
# actually present in the MoneyPuck goalie files -- no fallback needed).
NAME_TO_ABBREV = {
    "Boston Bruins": "BOS", "Buffalo Sabres": "BUF", "Calgary Flames": "CGY",
    "Chicago Blackhawks": "CHI", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
    "Carolina Hurricanes": "CAR", "Los Angeles Kings": "LAK", "Dallas Stars": "DAL",
    "Montreal Canadiens": "MTL", "New Jersey Devils": "NJD", "New York Islanders": "NYI",
    "New York Rangers": "NYR", "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "Colorado Avalanche": "COL", "San Jose Sharks": "SJS",
    "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN", "Washington Capitals": "WSH",
    "Arizona Coyotes": "ARI", "Phoenix Coyotes": "ARI",
    "Anaheim Ducks": "ANA", "Anaheim Mighty Ducks": "ANA",
    "Florida Panthers": "FLA", "Nashville Predators": "NSH",
    "Atlanta Thrashers": "ATL", "Winnipeg Jets": "WPG",
    "Columbus Blue Jackets": "CBJ", "Minnesota Wild": "MIN",
    "Vegas Golden Knights": "VGK", "Seattle Kraken": "SEA",
    "Utah Hockey Club": "UTA", "Utah Mammoth": "UTA",
}

# The two goalie-only, per-game MoneyPuck files (see module docstring for
# provenance/equivalence-to-duplicates note).
DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "Datasets" / "NHL"
GOALIE_LOG_PATHS = [
    DATASETS_DIR / "2008_to_2024 copy 3.csv",
    DATASETS_DIR / "2025-2 copy.csv",
]


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def leakage_check(mp: pd.DataFrame):
    hr("STEP 1: LEAKAGE CHECK ON individual goalie goals/ongoal (same-game)")
    valid = mp["ongoal"] > 0
    save_pct_this_game = 1 - mp.loc[valid, "goals"] / mp.loc[valid, "ongoal"]
    corr = float(np.corrcoef(save_pct_this_game, mp.loc[valid, "goals"])[0, 1])
    print(f"corr(this-game individual save%, this-game individual goals allowed): {corr:.4f} "
          "(strongly negative, as expected -- confirms these are per-goalie GAME OUTCOMES, "
          "must never be used same-game, shift(1)-then-rolled trailing history only.)")
    print(f"League-wide individual save% sanity check: "
          f"{1 - mp['goals'].sum() / mp['ongoal'].sum():.4f} (realistic NHL range ~0.895-0.910).")


def load_goalie_starter_table() -> tuple:
    """
    Returns (starters_df, league_avg_save_pct). `starters_df` has one row
    per (gameDate, playerTeam) -- the inferred starting goalie for that
    team's game -- with `save_pct_l10` already shift(1)-then-rolled on that
    specific goalie's own prior appearances (walk-forward-safe: this value
    reflects only games that goalie played strictly before this one).
    """
    frames = [pd.read_csv(p) for p in GOALIE_LOG_PATHS]
    mp = pd.concat(frames, ignore_index=True)
    mp = mp[(mp["position"] == "G") & (mp["situation"] == "all")].copy()
    mp = mp[(mp["icetime"] > 0) & (mp["ongoal"] > 0)].copy()

    leakage_check(mp)
    league_avg_save_pct = 1.0 - float(mp["goals"].sum() / mp["ongoal"].sum())

    # Per-goalie chronological trailing save% -- shift(1) on THIS GOALIE's
    # own appearance sequence (own identity, not team), sum-of-ratio over
    # the trailing window (same convention as pp_pct_l10/pk_pct_l10).
    mp = mp.sort_values(["playerId", "gameDate"], kind="stable")
    grp = mp.groupby("playerId", group_keys=False)
    goals_l10sum = grp["goals"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    ongoal_l10sum = grp["ongoal"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    mp["save_pct_l10"] = 1.0 - goals_l10sum / ongoal_l10sum

    # Starter = max icetime among a team's goalie rows for that game (see
    # module docstring -- verified 1:1, no ties needing a tie-break).
    idx = mp.groupby(["gameId", "playerTeam"])["icetime"].idxmax()
    starters = mp.loc[idx, ["gameDate", "playerTeam", "name", "playerId", "save_pct_l10"]].copy()
    return starters, league_avg_save_pct


def add_starter_save_pct_feature(feats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    starters, league_avg = load_goalie_starter_table()
    print(f"League-average individual starter save% (fallback fill value): {league_avg:.4f}")
    print(f"Distinct goalies identified as a starter at least once: {starters['playerId'].nunique():,}")

    g = games[["game_id", "date", "home_team_name", "away_team_name"]].copy()
    g["home_abbrev"] = g["home_team_name"].map(NAME_TO_ABBREV)
    g["away_abbrev"] = g["away_team_name"].map(NAME_TO_ABBREV)

    # UTC -> local schedule-date fix (see module docstring's JOIN MECHANICS
    # section for the measured 33% -> 93.2% match-rate improvement this gets).
    local_date = pd.to_datetime(g["date"]).dt.tz_localize(None).dt.normalize()
    shift_back = pd.to_datetime(g["date"]).dt.hour < 12
    local_date = local_date.where(~shift_back, local_date - pd.Timedelta(days=1))
    g["date_int"] = local_date.dt.strftime("%Y%m%d").astype(int)

    home_starters = starters.rename(columns={
        "playerTeam": "home_abbrev", "save_pct_l10": "home_starter_save_pct_l10",
        "name": "home_starter_name",
    })[["gameDate", "home_abbrev", "home_starter_save_pct_l10", "home_starter_name"]]
    away_starters = starters.rename(columns={
        "playerTeam": "away_abbrev", "save_pct_l10": "away_starter_save_pct_l10",
        "name": "away_starter_name",
    })[["gameDate", "away_abbrev", "away_starter_save_pct_l10", "away_starter_name"]]

    g = g.merge(home_starters, left_on=["date_int", "home_abbrev"], right_on=["gameDate", "home_abbrev"], how="left")
    g = g.merge(away_starters, left_on=["date_int", "away_abbrev"], right_on=["gameDate", "away_abbrev"], how="left")

    join_hit = g["home_starter_save_pct_l10"].notna() & g["away_starter_save_pct_l10"].notna()
    eligible = g["date_int"] >= 20081004
    print(f"Starter-identity join coverage (games on/after 2008-10-04, the earliest "
          f"MoneyPuck goalie-log date): {join_hit[eligible].mean()*100:.1f}% "
          f"({join_hit[eligible].sum():,}/{eligible.sum():,}) -- remainder (mostly playoff "
          f"games, see module docstring) filled with league-average.")

    g["home_starter_save_pct_l10"] = g["home_starter_save_pct_l10"].fillna(league_avg)
    g["away_starter_save_pct_l10"] = g["away_starter_save_pct_l10"].fillna(league_avg)

    out = feats.merge(
        g[["game_id", "home_starter_save_pct_l10", "away_starter_save_pct_l10"]],
        on="game_id", how="left",
    )
    out["home_starter_save_pct_l10"] = out["home_starter_save_pct_l10"].fillna(league_avg)
    out["away_starter_save_pct_l10"] = out["away_starter_save_pct_l10"].fillna(league_avg)
    return out


def run_pipeline(feats: pd.DataFrame, feature_cols: list) -> dict:
    """Same shape as research_goalie_pk.py's run_pipeline, with one
    deliberate change: ROI is computed on TOTAL-MARKET BETS ONLY (this
    hypothesis is specifically about the total market, per the task/
    reasoning below -- mixing in moneyline/spread bets, as the team-level
    save_pct test did, would dilute exactly the market this feature is
    theorized to move)."""
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0,
        allowed_confidence=("Medium", "High"),
        price_point="Close",
    )
    bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    total_bets = bets[bets["market"] == "total"] if not bets.empty else bets
    if total_bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}

    summary = backtest.summarize(total_bets)
    return {
        "margin_corr": margin_corr,
        "total_corr": total_corr,
        "roi_pct": float(summary["roi_pct"].iloc[0]),
        "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
        "bets": int(summary["bets"].iloc[0]),
    }, total_bets


def split_half_total_roi(total_bets: pd.DataFrame) -> tuple:
    """(delta_first_half, delta_second_half) is not meaningful for a single
    whole-population feature the way it is for a subgroup comparison -- kept
    here only as a season-stability read: ROI computed independently on each
    half of the total-market bet log's seasons, reported alongside the main
    result so a directionally-unstable effect is visible even before any
    subgroup framing is considered."""
    seasons = sorted(total_bets["season"].dropna().unique())
    if len(seasons) < 2:
        return None, None
    mid = seasons[len(seasons) // 2]
    first = total_bets[total_bets["season"] < mid]
    second = total_bets[total_bets["season"] >= mid]

    def roi(df):
        if df.empty:
            return float("nan")
        return float(df["flat_profit"].sum() / len(df) * 100.0)

    return roi(first), roi(second)


def main():
    hr("BUILDING BASE PIPELINE (loader -> power ratings -> features, current production state)")
    games = load_games()

    rating_cfg = PowerRatingConfig(
        k_factor=nhl_config.ELO_K_FACTOR,
        start_rating=nhl_config.ELO_START_RATING,
        home_field_adv=nhl_config.HOME_FIELD_ADV_ELO,
        season_regression=nhl_config.SEASON_REGRESSION,
        mov_mult_base=nhl_config.MOV_MULT_BASE,
        mov_mult_divisor=nhl_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_team_id", away_col="away_team_id",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        config=rating_cfg,
    )
    feats_base = build_features(games, rr.history)
    print(f"Base feature rows: {len(feats_base):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")

    hr("BASELINE RUN (current production ML_FEATURE_COLS, TOTAL-market bets only)")
    baseline, baseline_total_bets = run_pipeline(feats_base, ML_FEATURE_COLS)
    print(baseline)
    b1, b2 = split_half_total_roi(baseline_total_bets)
    print(f"Baseline TOTAL-market ROI by season half: first {b1:+.2f}%, second {b2:+.2f}%"
          if b1 is not None else "Baseline TOTAL-market ROI: not enough seasons to split")

    hr("HYPOTHESIS: adding STARTING GOALIE'S OWN individually-identified trailing save%")
    feats_sv = add_starter_save_pct_feature(feats_base, games)
    feature_cols_sv = ML_FEATURE_COLS + ["home_starter_save_pct_l10", "away_starter_save_pct_l10"]
    variant, variant_total_bets = run_pipeline(feats_sv, feature_cols_sv)
    print(variant)
    v1, v2 = split_half_total_roi(variant_total_bets)
    print(f"Variant TOTAL-market ROI by season half: first {v1:+.2f}%, second {v2:+.2f}%"
          if v1 is not None else "Variant TOTAL-market ROI: not enough seasons to split")

    hyp = Hypothesis(
        name="nhl_starting_goalie_individual_save_pct_total_market",
        reasoning=(
            "Distinct from the already-tested-and-rejected team-level trailing save% "
            "(decision_log.jsonl, nhl_trailing_save_pct, adopt_cautiously/null on 2026-08-12) -- "
            "that test blended whichever two goalies a team happened to use over its last 10 games, "
            "diluting exactly the individual shot-stopping signal hockey analysts mean by 'hot/cold "
            "goaltending.' This test isolates the SPECIFIC identified starting goalie (inferred via "
            "max icetime per team-game from a real per-player-per-game MoneyPuck log, joined onto the "
            "existing games table) and its own trailing individual save percentage, tested specifically "
            "against the TOTAL market (not moneyline/spread) since a goalie's shot-stopping quality most "
            "directly acts on how many goals get scored in the game overall, not which team wins by how "
            "much against the spread. A specific individual's own recent form is a materially different, "
            "more precisely targeted signal than a team-wide blend, and is worth testing on its own merits "
            "rather than assuming the prior team-level null already settled the question."
        ),
        sport="NHL",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT (evaluate_hypothesis().to_dict())")
    print(json.dumps(result.to_dict(), indent=2))

    # If the whole-population result isn't a clean adopt/reject -- borderline
    # but directionally real -- follow the exact worked pattern in
    # sports/nba/research_spread_favorite_underdog_shrinkage.py: define a
    # genuine SUBGROUP (games where the two starters' own quality gap is
    # large -- the cases where this signal should matter most, if it matters
    # at all) vs the REST of TOTAL-market bets, and let core/shrinkage.py
    # give an honest partial-trust answer instead of forcing a binary call.
    if result.recommendation in ("adopt_cautiously", "inconclusive"):
        hr("BORDERLINE RESULT -- running subgroup split (large starter save% gap vs. rest)")
        vb = variant_total_bets.copy()
        gap_lookup = feats_sv.set_index("game_id")[
            ["home_starter_save_pct_l10", "away_starter_save_pct_l10"]
        ]
        vb = vb.join(gap_lookup, on="game_id")
        vb["starter_gap"] = (vb["home_starter_save_pct_l10"] - vb["away_starter_save_pct_l10"]).abs()
        median_gap = vb["starter_gap"].median()
        large_gap = vb[vb["starter_gap"] >= median_gap]
        rest = vb[vb["starter_gap"] < median_gap]

        def roi_stats(df):
            n = len(df)
            if n == 0:
                return {"bets": 0, "roi_pct": float("nan"), "roi_stderr_pct": float("nan")}
            roi = df["flat_profit"].sum() / n * 100.0
            stderr = df["flat_profit"].std(ddof=1) / (n ** 0.5) * 100.0 if n > 1 else float("nan")
            return {"bets": n, "roi_pct": roi, "roi_stderr_pct": stderr}

        sub = roi_stats(large_gap)
        rst = roi_stats(rest)
        print(f"Large starter-gap subgroup: n={sub['bets']}, ROI={sub['roi_pct']:+.2f}% "
              f"+/-{sub['roi_stderr_pct']:.2f}pp (median gap cutoff {median_gap:.4f})")
        print(f"Rest of TOTAL-market population: n={rst['bets']}, ROI={rst['roi_pct']:+.2f}% "
              f"+/-{rst['roi_stderr_pct']:.2f}pp")

        seasons = sorted(vb["season"].dropna().unique())
        if len(seasons) >= 2:
            mid = seasons[len(seasons) // 2]

            def delta(df):
                f = roi_stats(df[df["starter_gap"] >= median_gap])
                r = roi_stats(df[df["starter_gap"] < median_gap])
                if f["bets"] == 0 or r["bets"] == 0:
                    return float("nan")
                return f["roi_pct"] - r["roi_pct"]

            d1 = delta(vb[vb["season"] < mid])
            d2 = delta(vb[vb["season"] >= mid])
            print(f"Split-half delta (large-gap - rest): first half {d1:+.2f}pp, second half {d2:+.2f}pp")
            split_deltas = (d1, d2) if not (np.isnan(d1) or np.isnan(d2)) else None
        else:
            split_deltas = None

        sub_result = research.evaluate_subgroup_hypothesis(
            hyp, market="total", subgroup_metrics=sub, rest_metrics=rst,
            split_half_deltas=split_deltas, min_bets=100,
        )
        hr("SUBGROUP RESULT")
        print(f"z={sub_result.z_score:+.2f} (bar: {sub_result.stderr_multiplier:.2f}), "
              f"stable_direction={sub_result.stable_direction}, recommendation={sub_result.recommendation}")
        print(f"Shrinkage: subgroup_weight={sub_result.shrinkage.subgroup_weight:.3f}, "
              f"shrunk_effect={sub_result.shrinkage.shrunk_effect:+.2f}pp")

    hr("DONE")


if __name__ == "__main__":
    main()
