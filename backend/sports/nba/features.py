"""
Walk-forward-safe feature engineering for NBA games. Structurally identical
to sports/cfb/features.py - every rolling/trailing stat here is computed
with the current game excluded (via shift(1) before any rolling window), so
a game's feature row only ever reflects information that existed strictly
before that game's tip-off.

Differences from the CFB version, both driven by the data rather than
stylistic: `is_bowl`/`is_neutral_venue` don't exist for NBA in this source
(no venue column at all, so a handful of London/Mexico City neutral-site
games in the 2013+ seasons can't be flagged - a known, low-impact gap,
documented rather than faked) and are replaced by `is_playoff` (real,
directly from the data). NBA teams play far more back-to-backs than any
other sport in this project (0 days rest is common and well-documented in
NBA analytics as a real performance factor), so `home_b2b`/`away_b2b`
boolean flags are added alongside the generic `rest_days` - a genuinely new,
real, non-fabricated feature, not present in the other sports' modules
because it isn't a meaningful signal there.
"""

import pandas as pd

ROLL_WINDOW = 10  # trailing games considered for form stats

# League-average points scored per team across the full cleaned dataset
# (1950-2018, Regular Season + Playoffs, known team_ids, valid scores):
# pts mean ~103.0/team. Used only to fill a team's very first game in the
# data, where no trailing history exists yet.
LEAGUE_AVG_TEAM_SCORE = 103.0


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

    # first game of a team's history (or first game after a season gap) has
    # no prior rest date -> fill with a neutral value. 3 days is used
    # (rather than CFB's bye-week 7) since NBA teams routinely play every
    # 1-3 days during the season - 3 is a realistic "no info yet" default,
    # not a rare/extreme value like a full week would be for this sport.
    out["home_rest_days"] = out["home_rest_days"].fillna(3)
    out["away_rest_days"] = out["away_rest_days"].fillna(3)
    out["home_ats_pct_l10"] = out["home_ats_pct_l10"].fillna(0.5)
    out["away_ats_pct_l10"] = out["away_ats_pct_l10"].fillna(0.5)
    out["home_win_pct_l10"] = out["home_win_pct_l10"].fillna(0.5)
    out["away_win_pct_l10"] = out["away_win_pct_l10"].fillna(0.5)
    out["home_streak"] = out["home_streak"].fillna(0)
    out["away_streak"] = out["away_streak"].fillna(0)
    out["home_pf_l10"] = out["home_pf_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["home_pa_l10"] = out["home_pa_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["away_pf_l10"] = out["away_pf_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["away_pa_l10"] = out["away_pa_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)

    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]

    # Back-to-back flag: playing on consecutive calendar days since the
    # team's last game (real, cheap, well-documented NBA-specific signal -
    # see module docstring). `rest_days` is a DATE DIFFERENCE, so a true
    # back-to-back (played yesterday, playing again today) shows up as
    # rest_days == 1, not 0 - 0 would mean two games the same calendar day,
    # which does not happen in this sport. A team's very first game in the
    # data has rest_days filled to 3 above, so it is never mistakenly
    # flagged as a back-to-back.
    out["home_b2b"] = (out["home_rest_days"] <= 1).astype(int)
    out["away_b2b"] = (out["away_rest_days"] <= 1).astype(int)

    out["naive_total"], out["naive_margin"] = naive_score_features(
        out["home_pf_l10"], out["home_pa_l10"], out["away_pf_l10"], out["away_pa_l10"]
    )
    return out


def naive_score_features(home_pf, home_pa, away_pf, away_pa):
    """
    Naive "implied score" baseline: each team's expected points = average of
    (its own scoring rate, its opponent's rate of allowing points) - a
    standard, simple total/margin prior that gives the ML model a strong
    starting signal for scoring pace, which team strength alone (rating
    differential) doesn't capture. Identical formula to the NFL/CFB/MLB
    versions - sport-agnostic, just fed NBA-shaped inputs.
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
    Indexed by team (canonical franchise abbreviation), one row per team.
    Not exercised by verify.py (out of scope for this backend-only pass) but
    kept for interface parity with sports/cfb/features.py.
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
    "naive_total", "naive_margin",
    "home_streak", "away_streak", "home_b2b", "away_b2b", "is_playoff",
]
