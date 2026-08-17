"""
Walk-forward-safe feature engineering for MLB games. Structurally identical
to sports/cfb/features.py - every rolling/trailing stat here is computed
with the current game excluded (via shift(1) before any rolling window), so
a game's feature row only ever reflects information that existed strictly
before that game's first pitch.

Differences from the CFB/NFL versions, all driven by the data rather than
stylistic: there is no `ats_pct_l10` at all (no odds data exists to compute
an ATS result from - see loader.py/config.py's module docstrings), no
`is_bowl`/`is_neutral_venue` (MLB's regular season, which is all this
dataset contains, has no bowl/playoff/neutral-site games), and no divisional
flag (MLB has real divisions, but nothing here currently uses them). In
their place, two cheap, real, non-fabricated context flags are carried
straight from the loader: `is_interleague` (AL team @ NL team or vice versa)
and `is_night` (night game). `LEAGUE_AVG_TEAM_SCORE` is MLB's actual scale
(~4.6 runs/team) rather than football's ~22-27 points/team.

STARTING PITCHER QUALITY: `home_sp_er_lN` / `away_sp_er_lN` (a starter's
rolling earned-runs-allowed PER START, not innings-normalized ERA - see
sports/mlb/starting_pitcher.py's module docstring for the exact scoping
and why) must already be columns on the `games` DataFrame passed into
`build_features` - call `starting_pitcher.attach_starter_quality(games)`
before this, same pattern as sports/nfl/weather.py's attach_weather()
being called before NFL's build_features. `sp_er_diff_lN` is computed here,
the same "diff" convention as `rest_diff`/`pyth_pct_diff` elsewhere in this
project. Adopted (adopt_cautiously) via a core.research hypothesis test -
see research_starting_pitcher.py and decision_log.jsonl.
"""

import pandas as pd

ROLL_WINDOW = 10  # trailing games considered for form stats - same window as NFL/CFB for structural consistency, see module docstring for the tradeoff this implies given MLB's 162-game season

# League-average runs scored per team in this dataset (completed games,
# 1990-2025, non-tie): actual_total mean ~9.17 => ~4.58/team, rounded.
# Used only to fill a team's very first game in the data, where no trailing
# history exists yet.
LEAGUE_AVG_TEAM_SCORE = 4.6

# Typical days between a team's games (MLB teams play almost daily with
# occasional off days) - used only to fill a team's very first game, where
# no prior game date exists to diff against.
LEAGUE_AVG_REST_DAYS = 1.0


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (team, game) so rolling stats can be computed per-team."""
    cols = ["game_id", "date", "season", "home_win", "home_score", "away_score"]

    home = games[cols + ["home_franchise", "away_franchise"]].rename(columns={
        "home_franchise": "team", "away_franchise": "opponent",
        "home_win": "won", "home_score": "points_for", "away_score": "points_against",
    })
    home["is_home"] = True

    away = games[cols + ["away_franchise", "home_franchise"]].rename(columns={
        "away_franchise": "team", "home_franchise": "opponent",
        "home_win": "won", "home_score": "points_against", "away_score": "points_for",
    })
    away["won"] = 1 - away["won"]
    away["is_home"] = False

    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["team", "date"], kind="stable")


def _rolling_form(log: pd.DataFrame) -> pd.DataFrame:
    grp = log.groupby("team", group_keys=False)

    # shift(1) first so the current game is never included in its own rolling window
    log["win_pct_l10"] = grp["won"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    log["pf_l10"] = grp["points_for"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    log["pa_l10"] = grp["points_against"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    log["rest_days"] = grp["date"].apply(lambda s: s.diff().dt.days)

    # signed current streak (positive = winning streak, negative = losing streak), pre-game
    def _streak(sub: pd.DataFrame) -> pd.Series:
        prior = sub["won"].shift(1)
        streaks = []
        cur = 0
        for w in prior:
            if pd.isna(w):
                streaks.append(0)
                cur = 0
                continue
            if w == 1:
                cur = cur + 1 if cur >= 0 else 1
            else:
                cur = cur - 1 if cur <= 0 else -1
            streaks.append(cur)
        return pd.Series(streaks, index=sub.index)

    log["streak"] = grp.apply(_streak).reset_index(level=0, drop=True)
    return log


def build_features(games: pd.DataFrame, rating_history: pd.DataFrame) -> pd.DataFrame:
    """
    `games` is the loader's cleaned DataFrame; `rating_history` is the
    `RatingResult.history` from power_ratings (indexed by game_id, holding
    pre-game ratings). Returns `games` joined with engineered, walk-forward
    -safe feature columns for both teams.
    """
    log = _team_game_log(games)
    log = _rolling_form(log)

    home_feats = log[log["is_home"]][
        ["game_id", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak"]
    ].rename(columns={
        "win_pct_l10": "home_win_pct_l10",
        "pf_l10": "home_pf_l10", "pa_l10": "home_pa_l10",
        "rest_days": "home_rest_days", "streak": "home_streak",
    })
    away_feats = log[~log["is_home"]][
        ["game_id", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak"]
    ].rename(columns={
        "win_pct_l10": "away_win_pct_l10",
        "pf_l10": "away_pf_l10", "pa_l10": "away_pa_l10",
        "rest_days": "away_rest_days", "streak": "away_streak",
    })

    out = games.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out = out.merge(rating_history[["home_rating_pre", "away_rating_pre", "rating_diff_pre",
                                     "elo_home_win_prob"]], left_on="game_id", right_index=True, how="left")

    # first game of a team's history has no prior rest date -> fill with a neutral typical-rest value
    out["home_rest_days"] = out["home_rest_days"].fillna(LEAGUE_AVG_REST_DAYS)
    out["away_rest_days"] = out["away_rest_days"].fillna(LEAGUE_AVG_REST_DAYS)
    out["home_win_pct_l10"] = out["home_win_pct_l10"].fillna(0.5)
    out["away_win_pct_l10"] = out["away_win_pct_l10"].fillna(0.5)
    out["home_streak"] = out["home_streak"].fillna(0)
    out["away_streak"] = out["away_streak"].fillna(0)
    out["home_pf_l10"] = out["home_pf_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["home_pa_l10"] = out["home_pa_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["away_pf_l10"] = out["away_pf_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["away_pa_l10"] = out["away_pa_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)

    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]

    out["naive_total"], out["naive_margin"] = naive_score_features(
        out["home_pf_l10"], out["home_pa_l10"], out["away_pf_l10"], out["away_pa_l10"]
    )

    # Starting-pitcher rolling ER/start: home_sp_er_lN/away_sp_er_lN must
    # already be present on `games` (see module docstring - call
    # starting_pitcher.attach_starter_quality(games) beforehand). Positive
    # sp_er_diff_lN = home's starter has allowed FEWER earned runs per start
    # recently than away's -> favors home, same sign convention as rest_diff.
    if "home_sp_er_lN" in out.columns and "away_sp_er_lN" in out.columns:
        out["sp_er_diff_lN"] = out["away_sp_er_lN"] - out["home_sp_er_lN"]

    return out


def naive_score_features(home_pf, home_pa, away_pf, away_pa):
    """
    Naive "implied score" baseline: each team's expected runs = average of
    (its own scoring rate, its opponent's rate of allowing runs) - a
    standard, simple total/margin prior that gives the ML model a strong
    starting signal for scoring pace, which team strength alone (rating
    differential) doesn't capture. Identical formula to the NFL/CFB
    versions - sport-agnostic, just fed MLB-shaped (runs, not points)
    inputs. Shared by `build_features` (historical, vectorized over a
    DataFrame) and any future manually-entered-game caller so the formula
    only lives in one place.
    """
    implied_home_score = (home_pf + away_pa) / 2.0
    implied_away_score = (away_pf + home_pa) / 2.0
    return implied_home_score + implied_away_score, implied_home_score - implied_away_score


def current_form_snapshot(games: pd.DataFrame) -> pd.DataFrame:
    """
    Each team's rolling form as of RIGHT NOW (i.e. including their most
    recent played game, unlike the walk-forward columns in `build_features`
    which deliberately exclude the current row). Used only to score brand
    new manually-entered upcoming games - there's no leakage concern since
    every game in `games` is legitimately in the past relative to a new one.
    Indexed by canonical franchise. Not exercised by verify.py (out of scope
    for this backend-only pass) but kept for interface parity with
    sports/nfl/features.py and sports/cfb/features.py.
    """
    log = _team_game_log(games)
    grp = log.groupby("team", group_keys=False)
    log["win_pct_l10"] = grp["won"].apply(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    log["pf_l10"] = grp["points_for"].apply(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    log["pa_l10"] = grp["points_against"].apply(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())

    def _streak_now(sub: pd.DataFrame) -> pd.Series:
        cur = 0
        out = []
        for w in sub["won"]:
            cur = (cur + 1 if cur >= 0 else 1) if w == 1 else (cur - 1 if cur <= 0 else -1)
            out.append(cur)
        return pd.Series(out, index=sub.index)

    log["streak"] = grp.apply(_streak_now).reset_index(level=0, drop=True)

    latest = log.sort_values(["team", "date"]).groupby("team").tail(1).set_index("team")
    return latest[["win_pct_l10", "pf_l10", "pa_l10", "streak", "date"]].rename(
        columns={"date": "last_game_date"}
    )


ML_FEATURE_COLS = [
    "rating_diff_pre", "rest_diff", "home_rest_days", "away_rest_days",
    "home_win_pct_l10", "away_win_pct_l10",
    "home_pf_l10", "home_pa_l10", "away_pf_l10", "away_pa_l10",
    "naive_total", "naive_margin",
    "home_streak", "away_streak", "is_interleague", "is_night",
    "home_sp_er_lN", "away_sp_er_lN", "sp_er_diff_lN",
]


def extra_matchup_features(home_fr, away_fr, game_date, home_row, away_row) -> dict:
    """
    Live-scoring counterpart to build_features' loader-derived sp_er_lN
    columns above. core/matchup.py (the generic scorer every live MLB pick
    goes through) calls this hook for a brand-new, not-yet-played game --
    which has no loader row to pull home_sp_er_lN/away_sp_er_lN from, since
    that requires starting_pitcher.attach_starter_quality(games), a
    dataset-wide batch join that only runs once, offline, during
    build_pipeline(). Without this hook, core/matchup.py's generic
    fill-anything-missing-with-a-default loop was silently filling both
    columns with 0.0 for every single live MLB prediction -- and 0.0 isn't
    a neutral "no signal" value here, it's a real, wrong one: it reads as
    "this starter has allowed zero earned runs across their last 8 starts,"
    an extreme outlier that basically never happens in the training data
    (league average is ~2.6). Fed to a HistGradientBoostingRegressor (which
    branches on exact thresholds, unlike a linear model that would just
    shrug off an off-distribution constant), that's a real, systematic bias
    on every live MLB pick, not just missing information. Confirmed by
    tracing core/matchup.py's fallback logic against this module's own
    ML_FEATURE_COLS.

    NOT fixed by actually computing each team's real confirmed starter's
    current rolling form live (that needs today's probable-pitcher
    identity, already surfaced for DISPLAY only via sports/mlb/probables.py,
    joined against that specific pitcher's own recent starts -- a real
    future enhancement, not attempted here). This hook instead falls back
    to the same LEAGUE_AVG_SP_ER_PER_START neutral value the historical
    pipeline already uses for any game outside its own event-file coverage
    -- the honest "no live signal available" value the model was actually
    trained to see for a missing starter, not an invented one.
    """
    from .starting_pitcher import LEAGUE_AVG_SP_ER_PER_START
    return {
        "home_sp_er_lN": LEAGUE_AVG_SP_ER_PER_START,
        "away_sp_er_lN": LEAGUE_AVG_SP_ER_PER_START,
        "sp_er_diff_lN": 0.0,
    }
