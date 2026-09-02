"""
Builds the entire NFL model pipeline exactly once: loader -> features ->
walk-forward power ratings -> walk-forward ML -> ensemble residual stds ->
backtest -> final models (fit on all history) -> current ratings/form.

This is the SINGLE SOURCE OF TRUTH for "how the model gets built," used by
both the FastAPI app (`main.py`'s startup hook) and the standalone harness
script (`harness.py`, run via cron/launchd with no server involved). Before
this refactor the two would have needed to duplicate this sequence, which
is exactly how a live app and an offline job quietly drift out of sync with
each other — extracting it here removes that risk entirely.
"""

import importlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

import database
from core import ensemble, backtest, live_results
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict, fit_final_models
from sports.nfl import config as nfl_config
from sports.nfl.loader import load_games
from sports.nfl.features import build_features, current_form_snapshot, ML_FEATURE_COLS
from sports.nfl.weather import attach_weather


def records(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    return json.loads(df.replace([np.inf, -np.inf], np.nan).fillna(0).to_json(orient="records"))


def build_nfl_pipeline(persist_backtest: bool = True) -> dict:
    games = load_games()
    games = attach_weather(games)  # disk-cached — see sports/nfl/weather.py

    # Real, ESPN-synced, COMPLETED games this season get folded into the Elo
    # computation AND the current-form snapshot below (current_form_snapshot
    # (games_for_ratings), not the stale current_form_snapshot(games) this
    # used to be) — see core/live_results.py's module docstring for the gap
    # this closes. NOT used for the walk-forward ML training data, though:
    # `feats = build_features(games, ...)` a few lines down deliberately
    # keeps using the ORIGINAL `games` — live rows don't carry the full odds/
    # feature schema historical rows do, and mixing them into training risks
    # schema drift in the most heavily-tested part of this system. Safe by
    # construction: live rows are always strictly AFTER the historical
    # file's own max date, so nothing here can double-count or alter any
    # historical game's pre-game rating/form lookup.
    games_for_ratings = live_results.with_live_results(games, "NFL")
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", neutral_col="is_neutral_venue",
        config=PowerRatingConfig(
            k_factor=nfl_config.ELO_K_FACTOR, start_rating=nfl_config.ELO_START_RATING,
            home_field_adv=nfl_config.HOME_FIELD_ADV_ELO, season_regression=nfl_config.SEASON_REGRESSION,
            mov_mult_base=nfl_config.MOV_MULT_BASE, mov_mult_divisor=nfl_config.MOV_MULT_DIVISOR,
        ),
    )
    feats = build_features(games, rr.history)
    wf = walk_forward_predict(feats, ML_FEATURE_COLS)

    # full history (left join) for browsing; OOS-only (inner join) for backtest/residual stats
    history_df = feats.set_index("game_id").join(wf.predictions, how="left")
    oos_df = history_df.dropna(subset=["predicted_margin"])

    stds = ensemble.compute_residual_stds(oos_df, nfl_config.ELO_POINTS_PER_MARGIN)
    ecfg = ensemble.EnsembleConfig()
    bcfg = backtest.BacktestConfig(min_edge_pct=3.0)

    bets_df = backtest.run_backtest(oos_df, stds, nfl_config.ELO_POINTS_PER_MARGIN, ecfg, bcfg, sport="NFL")
    if not bets_df.empty:
        bets_df["edge_bucket"] = bets_df["edge_pct"].apply(backtest.edge_bucket)
    overall = backtest.summarize(bets_df)
    by_market = backtest.summarize(bets_df, ["market"])
    by_market_edge = backtest.summarize(bets_df, ["market", "edge_bucket"])
    by_season = backtest.summarize(bets_df, ["season"])
    curve = backtest.bankroll_curve(bets_df, bcfg)

    backtest_summary = {
        "config": {
            "min_edge_pct": bcfg.min_edge_pct, "allowed_confidence": list(bcfg.allowed_confidence),
            "kelly_frac": bcfg.kelly_frac, "price_point": bcfg.price_point,
        },
        "overall": records(overall)[0] if not overall.empty else {},
        "by_market": records(by_market),
        "by_market_edge_bucket": records(by_market_edge),
        "by_season": records(by_season),
        "bankroll_curve": records(curve.assign(date=curve["date"].astype(str))) if not curve.empty else [],
    }
    if persist_backtest:
        with database.get_db() as conn:
            conn.execute(
                "INSERT INTO backtest_results (sport, summary_json) VALUES (?, ?)",
                ("NFL", json.dumps(backtest_summary, default=str)),
            )

    # final models fit on ALL history (every historical game is legitimately "the
    # past" relative to a brand-new upcoming game), used only to score those.
    final_models = fit_final_models(feats, ML_FEATURE_COLS)
    current_season = nfl_config.season_for_date(datetime.now())
    current_ratings = rr.ratings_entering_season(current_season)
    # games_for_ratings (not the static-only `games`), same real fix as
    # `current_ratings` just above -- see core/live_results.py's module
    # docstring. Real, confirmed bug this closes: current_form_snapshot(games)
    # silently used ONLY the static historical file, which ends whenever last
    # season's dataset was captured -- so every live prediction's rest_days
    # was computed against a team's LAST HISTORICAL FILE game, not their real
    # last game. Verified directly: this had a real team's home_rest_days at
    # 324 for a live game (should be ~1-3), a wildly out-of-distribution value
    # fed straight into a HistGradientBoostingRegressor for every live pick.
    current_form = current_form_snapshot(games_for_ratings)

    # latest display name used per franchise (so relocated/rebranded teams show
    # their current name in the UI even though older rows kept their old name)
    long_names = pd.concat([
        games[["date", "home_franchise", "Home Team"]].rename(columns={"home_franchise": "franchise", "Home Team": "name"}),
        games[["date", "away_franchise", "Away Team"]].rename(columns={"away_franchise": "franchise", "Away Team": "name"}),
    ])
    latest_names = long_names.sort_values("date").groupby("franchise")["name"].last().to_dict()

    return dict(
        games=games, rating_result=rr, history_df=history_df, oos_df=oos_df,
        stds=stds, ensemble_cfg=ecfg, backtest_cfg=bcfg, backtest_summary=backtest_summary,
        final_models=final_models, current_ratings=current_ratings, current_form=current_form,
        latest_names=latest_names,
    )


# ---------------------------------------------------------------------------
# Generic multi-sport builder — NBA/MLB/NHL (CFB deliberately excluded, no
# live wiring planned per this session's scope decision). Each sport's data,
# config, ratings, and models stay fully separate — this function only
# dispatches to whichever single sport's own modules it's asked for, it
# never mixes state between sports. NFL keeps its own dedicated
# `build_nfl_pipeline()` above (weather, Pythagorean features, its own
# tested history) rather than being folded into this generic path, to avoid
# any regression risk on the one sport with the deepest track record.
# ---------------------------------------------------------------------------
def build_pipeline(sport: str, persist_backtest: bool = True) -> dict:
    sport = sport.lower()
    loader = importlib.import_module(f"sports.{sport}.loader")
    config = importlib.import_module(f"sports.{sport}.config")
    features = importlib.import_module(f"sports.{sport}.features")

    games = loader.load_games()

    # some sports keep the real-odds join in a separate module (e.g. MLB's
    # odds_loader.py, added after its base loader.py was already built
    # odds-free) — apply it if present so the backtest has real market data.
    try:
        odds_loader = importlib.import_module(f"sports.{sport}.odds_loader")
        games = odds_loader.attach_odds(games)
    except ModuleNotFoundError:
        pass

    # research-adopted features sometimes need a pre-`build_features` attach
    # step of their own (e.g. MLB's starting_pitcher.attach_starter_quality,
    # adopted via a core.research hypothesis test) — same optional-module
    # pattern as odds_loader above, generalized so a future adopted feature
    # doesn't need this dispatcher edited again as long as it follows the
    # same `attach_*(games) -> games` convention.
    for extra_module_name, attach_fn_name in [("starting_pitcher", "attach_starter_quality")]:
        try:
            extra_module = importlib.import_module(f"sports.{sport}.{extra_module_name}")
            games = getattr(extra_module, attach_fn_name)(games)
        except ModuleNotFoundError:
            pass

    # Real, ESPN-synced, COMPLETED games this season folded into the Elo
    # computation AND the current-form snapshot below — see core/live_results.py
    # and build_nfl_pipeline()'s matching comment above for the full
    # reasoning. Harmless no-op for CFB (never in LIVE_SPORTS, harness never
    # syncs it — with_live_results() queries an always-empty result and
    # returns `games` unchanged).
    games_for_ratings = live_results.with_live_results(games, sport.upper())
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        neutral_col="is_neutral_venue" if "is_neutral_venue" in games.columns else None,
        config=PowerRatingConfig(
            k_factor=config.ELO_K_FACTOR, start_rating=config.ELO_START_RATING,
            home_field_adv=config.HOME_FIELD_ADV_ELO, season_regression=config.SEASON_REGRESSION,
            mov_mult_base=config.MOV_MULT_BASE, mov_mult_divisor=config.MOV_MULT_DIVISOR,
        ),
    )
    feats = features.build_features(games, rr.history)
    wf = walk_forward_predict(feats, features.ML_FEATURE_COLS)

    history_df = feats.set_index("game_id").join(wf.predictions, how="left")
    oos_df = history_df.dropna(subset=["predicted_margin"])

    stds = ensemble.compute_residual_stds(oos_df, config.ELO_POINTS_PER_MARGIN)
    ecfg = ensemble.EnsembleConfig()
    # unlike NFL (real open-vs-close data), every other sport module built this session
    # has only a single odds snapshot — populated into the "*Close" columns by convention
    # (see each sport's odds loader/verify.py) — so "Close" is the only real price_point
    # that exists for any of them. Using the default "Open" here would silently find zero
    # opportunities every time (confirmed: this was a real bug caught while wiring this up).
    bcfg = backtest.BacktestConfig(min_edge_pct=3.0, price_point="Close")

    bets_df = backtest.run_backtest(oos_df, stds, config.ELO_POINTS_PER_MARGIN, ecfg, bcfg, sport=sport)
    if not bets_df.empty:
        bets_df["edge_bucket"] = bets_df["edge_pct"].apply(backtest.edge_bucket)
    overall = backtest.summarize(bets_df)
    by_market = backtest.summarize(bets_df, ["market"])
    by_market_edge = backtest.summarize(bets_df, ["market", "edge_bucket"])
    by_season = backtest.summarize(bets_df, ["season"])
    curve = backtest.bankroll_curve(bets_df, bcfg)

    backtest_summary = {
        "config": {
            "min_edge_pct": bcfg.min_edge_pct, "allowed_confidence": list(bcfg.allowed_confidence),
            "kelly_frac": bcfg.kelly_frac, "price_point": bcfg.price_point,
        },
        "overall": records(overall)[0] if not overall.empty else {},
        "by_market": records(by_market),
        "by_market_edge_bucket": records(by_market_edge),
        "by_season": records(by_season),
        "bankroll_curve": records(curve.assign(date=curve["date"].astype(str))) if not curve.empty else [],
    }
    if persist_backtest:
        with database.get_db() as conn:
            conn.execute(
                "INSERT INTO backtest_results (sport, summary_json) VALUES (?, ?)",
                (sport.upper(), json.dumps(backtest_summary, default=str)),
            )

    final_models = fit_final_models(feats, features.ML_FEATURE_COLS)
    current_season = config.season_for_date(datetime.now())
    current_ratings = rr.ratings_entering_season(current_season)
    # games_for_ratings, same fix and same reasoning as build_nfl_pipeline's
    # matching comment above -- see core/live_results.py's module docstring.
    current_form = features.current_form_snapshot(games_for_ratings)

    # display names: prefer each sport's own franchise_display_name() (MLB/NBA/NHL
    # all have one — see their config.py, built from the ESPN-name mapping the
    # live harness already needed) over trusting a raw `home_team` column as if it
    # were a display name. Real bug this replaced: MLB's own `home_team` column is
    # a Retrosheet CODE, not a display name (`home_franchise` is literally derived
    # FROM it via `canonical_team`) — the old column-presence check treated it as
    # one anyway, which is exactly why the Ratings tab was showing "NYA"/"LAN"
    # instead of "New York Yankees"/"Los Angeles Dodgers".
    display_fn = getattr(config, "franchise_display_name", None)
    all_franchises = pd.concat([games["home_franchise"], games["away_franchise"]]).unique()
    if display_fn:
        latest_names = {fr: display_fn(fr) for fr in all_franchises}
    elif "home_team" in games.columns and "away_team" in games.columns:
        long_names = pd.concat([
            games[["date", "home_franchise", "home_team"]].rename(
                columns={"home_franchise": "franchise", "home_team": "name"}),
            games[["date", "away_franchise", "away_team"]].rename(
                columns={"away_franchise": "franchise", "away_team": "name"}),
        ])
        latest_names = long_names.sort_values("date").groupby("franchise")["name"].last().to_dict()
    else:
        latest_names = {fr: fr for fr in all_franchises}

    return dict(
        sport=sport.upper(), games=games, rating_result=rr, history_df=history_df, oos_df=oos_df,
        stds=stds, ensemble_cfg=ecfg, backtest_cfg=bcfg, backtest_summary=backtest_summary,
        final_models=final_models, current_ratings=current_ratings, current_form=current_form,
        latest_names=latest_names,
    )
