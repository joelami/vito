"""
Bridges real, ESPN-synced, COMPLETED games back into the power-rating
engine — the gap the app owner asked about directly: "are you able to
continue to build on a live Elo rating with the data we're pulling from
ESPN?" Answer before this module existed: no. `pipeline.py` recomputed the
entire Elo history from the static historical dataset file every time, with
zero awareness that `harness.py` had already synced and settled real games
this season via `espn_games`. A team's "current rating" was frozen at
wherever the historical file's last recorded season ended, then regressed
toward the mean for "next season" — never actually incremented by a real
result no matter how many real games had already been played and recorded
this session.

This module reads real completed games out of `espn_games` and reshapes
them into the exact schema `core.power_ratings.compute_power_ratings()`
already expects from a sport's own historical loader — so they can be
concatenated onto the historical `games` DataFrame and processed by the
SAME walk-forward Elo update loop, not a separate, parallel rating path.

Deliberately scoped to ratings/form only, not the ML/backtest leg: a live
game usually lacks the same odds/feature richness the historical, already
walk-forward-tested feature set assumes, and mixing it into
`build_features()`'s training data risks introducing schema drift into the
one part of this system with the deepest, most carefully verified test
history. Elo is comparatively simple (score + date + who played), and is
exactly what a "live rating" means in normal usage — this is a real, scoped
fix for that specific gap, not a wholesale pipeline rearchitecture.
"""

import importlib

import pandas as pd

import database


def fetch_completed_espn_results(sport: str, since_date=None) -> pd.DataFrame:
    """
    Returns a DataFrame shaped like a sport's own `loader.load_games()`
    output (home_franchise, away_franchise, home_score, away_score, date,
    season, is_neutral_venue, home_win, game_id) built from real, completed,
    ESPN-synced games — empty DataFrame (not None) if there's nothing to
    add, so callers can unconditionally `pd.concat` without a branch.

    `home_win`/`game_id` matter beyond the rating engine: `current_form_
    snapshot()`'s `_team_game_log()` helper requires both by name (see
    sports/*/features.py) to compute rolling win%/rest-days/streak — without
    them, concatenating this onto `games` still "works" (pandas just fills
    NaN for the missing columns on the appended rows) but silently, wrongly
    excludes every live-synced game from those rolling stats instead of
    raising anything. `home_win` is trivially derivable here (home_score >
    away_score); `game_id` gets a synthetic `live_<espn_event_id>` value,
    distinct from the historical loader's own `<sport>_<n>` scheme by
    construction so the two can never collide.

    `since_date` should be the historical dataset's own max date — only
    games strictly after it are included, which is what makes this safe to
    blindly concatenate: the historical file and the live ESPN sync can
    never overlap in what they cover, so there's no risk of double-counting
    the same real game once from each source.
    """
    sport = sport.upper()
    empty = pd.DataFrame(columns=["home_franchise", "away_franchise", "home_score",
                                   "away_score", "date", "season", "is_neutral_venue",
                                   "home_win", "game_id"])

    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM espn_games WHERE sport = ? AND completed = 1 "
            "AND home_score IS NOT NULL AND away_score IS NOT NULL",
            (sport,),
        ).fetchall()]

    if not rows:
        # Real bug caught running this for the first time: CFB goes through
        # this same generic `with_live_results()` call (harmless in theory —
        # it's never in LIVE_SPORTS, so harness.py never syncs it, so this
        # query is always empty for CFB) but CFB's config.py has no
        # `canonical_franchise()` at all (it was never built to be
        # live-synced — see its own `canonical_team()` docstring). The old
        # code required `canonical_franchise` to exist BEFORE checking
        # whether there was even any data to resolve, so CFB's pipeline
        # build crashed on an empty-no-op case. Checking for real rows
        # first — and only requiring canonical_franchise() once there's
        # actually something to resolve — makes the "nothing to do" case
        # genuinely harmless for every sport, not just the ones that happen
        # to have live sync wired up.
        return empty

    config = importlib.import_module(f"sports.{sport.lower()}.config")
    canonical = getattr(config, "canonical_franchise", None)
    if canonical is None:
        raise ValueError(f"sports.{sport.lower()}.config has no canonical_franchise() — "
                          f"live results can't be resolved to this sport's franchise identifiers")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    if since_date is not None:
        since_date = pd.to_datetime(since_date)
        if since_date.tzinfo is not None:
            since_date = since_date.tz_localize(None)
        df = df[df["date"] > since_date]
    if df.empty:
        return empty

    df["home_franchise"] = df["home_team"].apply(canonical)
    df["away_franchise"] = df["away_team"].apply(canonical)
    df["season"] = df["date"].apply(config.season_for_date)
    df["is_neutral_venue"] = df["is_neutral_venue"].astype(bool)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["game_id"] = "live_" + df["espn_event_id"].astype(str)

    return df[["home_franchise", "away_franchise", "home_score", "away_score",
               "date", "season", "is_neutral_venue", "home_win", "game_id"
               ]].sort_values("date", kind="stable").reset_index(drop=True)


def with_live_results(games: pd.DataFrame, sport: str, date_col: str = "date") -> pd.DataFrame:
    """
    `games` (a sport's full historical DataFrame, already loaded) with real
    completed ESPN results appended — safe to call unconditionally (a
    sport with nothing new synced yet, or entirely off-season, just gets
    `games` back unchanged). Only appends the columns the rating engine and
    `current_form_snapshot()` actually need (home/away franchise, scores,
    date, season, neutral flag); anything else those functions don't touch
    stays NaN on the appended rows, which is fine — they're never read.

    Real bug found and fixed 2026-10-15 (discovered verifying a real NHL
    pipeline rebuild end-to-end): fetch_completed_espn_results() always
    returns a tz-NAIVE `date` column (a deliberate choice matching NFL/
    CFB/MLB/NBA's own bulk loaders, which are ALL tz-naive -- confirmed
    directly). NHL's bulk loader is the one real exception -- its own
    `date` column is tz-AWARE (datetime64[ns, UTC], confirmed directly) --
    so concatenating tz-naive live rows onto it produced a column pandas
    can't keep as one real datetime64 dtype (silently downgrades to
    `object`, holding a genuine mix of tz-aware and tz-naive Timestamp
    objects), which then crashed compute_power_ratings()'s own
    sort_values() with a real, reproduced `TypeError: Cannot compare
    tz-naive and tz-aware timestamps` -- meaning NHL's Elo ratings were
    silently never actually seeing this season's real completed games at
    all (this call raises every time it's attempted, so the exception
    this project's own pipeline-building try/except catches would have
    hidden it as "NHL pipeline FAILED to build" rather than a specific,
    diagnosable tz mismatch). Fixed here, not in
    fetch_completed_espn_results() itself (whose own tz-naive contract is
    correct for 4 of 5 sports and shouldn't change) -- matches the live
    rows' tz-awareness to whatever `games[date_col]` itself actually is,
    checked directly rather than assumed, the same "don't guess, verify"
    discipline core/matchup.py's own rest_days() tz fix (2026-09-15, see
    decision_log.jsonl) already established for this exact NHL quirk.
    """
    live = fetch_completed_espn_results(sport, since_date=games[date_col].max())
    if live.empty:
        return games
    target_tz_aware = games[date_col].dt.tz is not None
    if target_tz_aware and live[date_col].dt.tz is None:
        live = live.copy()
        live[date_col] = live[date_col].dt.tz_localize("UTC")
    elif not target_tz_aware and live[date_col].dt.tz is not None:
        live = live.copy()
        live[date_col] = live[date_col].dt.tz_localize(None)
    return pd.concat([games, live], ignore_index=True, sort=False)
