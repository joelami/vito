"""
Walk-forward-safe feature engineering for NFL games. Every rolling/trailing
stat here is computed with the current game excluded (via shift(1) before
any rolling window), so a game's feature row only ever reflects information
that existed strictly before that game's kickoff.
"""

import pandas as pd

ROLL_WINDOW = 10  # trailing games considered for form stats


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (team, game) so rolling stats can be computed per-team."""
    cols = ["game_id", "date", "season", "home_covers_close", "home_win",
            "home_score", "away_score"]

    home = games[cols + ["home_franchise", "away_franchise"]].rename(columns={
        "home_franchise": "team", "away_franchise": "opponent", "home_covers_close": "covered",
        "home_win": "won", "home_score": "points_for", "away_score": "points_against",
    })
    home["is_home"] = True

    away = games[cols + ["away_franchise", "home_franchise"]].rename(columns={
        "away_franchise": "team", "home_franchise": "opponent", "home_covers_close": "covered",
        "home_win": "won", "home_score": "points_against", "away_score": "points_for",
    })
    # flip perspective for the away team: covered is {1, 0 (push), -1, NaN}; negate, push/NaN unchanged
    away["covered"] = away["covered"].apply(lambda v: -v if pd.notna(v) else v)
    away["won"] = 1 - away["won"]
    away["is_home"] = False

    log = pd.concat([home, away], ignore_index=True)
    # rescale covered from {-1, 0, 1} (loss/push/win) to a genuine 0-1 rate so a rolling
    # mean is an actual "cover percentage" rather than ranging -1..1
    log["covered"] = log["covered"].map({1: 1.0, 0: 0.5, -1: 0.0})
    return log.sort_values(["team", "date"], kind="stable")


def _rolling_form(log: pd.DataFrame) -> pd.DataFrame:
    grp = log.groupby("team", group_keys=False)

    # shift(1) first so the current game is never included in its own rolling window
    log["ats_pct_l10"] = grp["covered"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
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
        ["game_id", "ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak"]
    ].rename(columns={
        "ats_pct_l10": "home_ats_pct_l10", "win_pct_l10": "home_win_pct_l10",
        "pf_l10": "home_pf_l10", "pa_l10": "home_pa_l10",
        "rest_days": "home_rest_days", "streak": "home_streak",
    })
    away_feats = log[~log["is_home"]][
        ["game_id", "ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak"]
    ].rename(columns={
        "ats_pct_l10": "away_ats_pct_l10", "win_pct_l10": "away_win_pct_l10",
        "pf_l10": "away_pf_l10", "pa_l10": "away_pa_l10",
        "rest_days": "away_rest_days", "streak": "away_streak",
    })

    out = games.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out = out.merge(rating_history[["home_rating_pre", "away_rating_pre", "rating_diff_pre",
                                     "elo_home_win_prob"]], left_on="game_id", right_index=True, how="left")

    # first game of a team's history has no prior rest date -> fill with a neutral bye-week value
    out["home_rest_days"] = out["home_rest_days"].fillna(7)
    out["away_rest_days"] = out["away_rest_days"].fillna(7)
    out["home_ats_pct_l10"] = out["home_ats_pct_l10"].fillna(0.5)
    out["away_ats_pct_l10"] = out["away_ats_pct_l10"].fillna(0.5)
    out["home_win_pct_l10"] = out["home_win_pct_l10"].fillna(0.5)
    out["away_win_pct_l10"] = out["away_win_pct_l10"].fillna(0.5)
    out["home_streak"] = out["home_streak"].fillna(0)
    out["away_streak"] = out["away_streak"].fillna(0)
    # league-average NFL score (~22-23/team) is a reasonable neutral fill for a team's first game ever
    out["home_pf_l10"] = out["home_pf_l10"].fillna(22.0)
    out["home_pa_l10"] = out["home_pa_l10"].fillna(22.0)
    out["away_pf_l10"] = out["away_pf_l10"].fillna(22.0)
    out["away_pa_l10"] = out["away_pa_l10"].fillna(22.0)

    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]

    out["naive_total"], out["naive_margin"] = naive_score_features(
        out["home_pf_l10"], out["home_pa_l10"], out["away_pf_l10"], out["away_pa_l10"]
    )
    out["home_pyth_pct"] = pythagorean_win_pct(out["home_pf_l10"], out["home_pa_l10"])
    out["away_pyth_pct"] = pythagorean_win_pct(out["away_pf_l10"], out["away_pa_l10"])
    out["pyth_pct_diff"] = out["home_pyth_pct"] - out["away_pyth_pct"]
    return out


PYTHAGOREAN_EXPONENT = 2.37  # standard NFL exponent (Football Outsiders' "Pythagorean Expectation")


def pythagorean_win_pct(points_for, points_against, exponent: float = PYTHAGOREAN_EXPONENT):
    """
    Pythagorean expected win percentage: a nonlinear function of scoring
    differential that's been shown (originally in baseball, adapted for NFL
    by Football Outsiders) to predict a team's TRUE strength better than its
    actual win-loss record over a short season — exactly the "16 games is a
    small sample, record can be misleading" problem a single season of NFL
    data has. Built on the same walk-forward-safe rolling points-for/against
    already computed above, so this adds no new leakage risk.
    """
    pf_exp = points_for ** exponent
    pa_exp = points_against ** exponent
    return pf_exp / (pf_exp + pa_exp)


def naive_score_features(home_pf, home_pa, away_pf, away_pa):
    """
    Naive "implied score" baseline: each team's expected points = average of
    (its own scoring rate, its opponent's rate of allowing points) — a
    standard, simple total/margin prior that gives the ML model a strong
    starting signal for scoring pace, which team strength alone (rating
    differential) doesn't capture. Shared by `build_features` (historical,
    vectorized over a DataFrame) and `main.py` (a single manually-entered
    upcoming game) so the formula only lives in one place.
    """
    implied_home_score = (home_pf + away_pa) / 2.0
    implied_away_score = (away_pf + home_pa) / 2.0
    return implied_home_score + implied_away_score, implied_home_score - implied_away_score


def current_form_snapshot(games: pd.DataFrame) -> pd.DataFrame:
    """
    Each team's rolling form as of RIGHT NOW (i.e. including their most
    recent played game, unlike the walk-forward columns in `build_features`
    which deliberately exclude the current row). Used only to score brand
    new manually-entered upcoming games — there's no leakage concern since
    every game in `games` is legitimately in the past relative to a new one.
    Indexed by team (franchise-normalized), one row per team.
    """
    log = _team_game_log(games)
    grp = log.groupby("team", group_keys=False)
    log["ats_pct_l10"] = grp["covered"].apply(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
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
    return latest[["ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "streak", "date"]].rename(
        columns={"date": "last_game_date"}
    )


ML_FEATURE_COLS = [
    "rating_diff_pre", "rest_diff", "home_rest_days", "away_rest_days",
    "home_ats_pct_l10", "away_ats_pct_l10", "home_win_pct_l10", "away_win_pct_l10",
    "home_pf_l10", "home_pa_l10", "away_pf_l10", "away_pa_l10",
    "naive_total", "naive_margin", "home_pyth_pct", "away_pyth_pct", "pyth_pct_diff",
    "home_streak", "away_streak", "is_divisional", "is_playoff", "is_neutral_venue",
    "game_temp_f", "game_wind_mph", "game_precip_mm", "is_dome",
]
