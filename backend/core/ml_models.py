"""
Gradient-boosted margin & total predictors, trained EXPANDING-WINDOW by
season: to produce a prediction for season Y, the model is fit only on
seasons strictly before Y, then slid forward one season at a time. This is
what makes the backtest's ML leg walk-forward-honest — a season is never in
its own training set, and no future season ever leaks into an earlier
prediction.

`HistGradientBoostingRegressor` is used instead of xgboost/lightgbm to avoid
adding a heavy new dependency; it's scikit-learn-native, handles the small-
to-medium row counts here (thousands, not millions) comfortably, and copes
with mild nonlinearity/interactions out of the box.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def _make_model() -> HistGradientBoostingRegressor:
    # Deliberately shallow/regularized: a few thousand rows of noisy sports
    # outcomes overfits fast with a deeper/more flexible model.
    #
    # loss="absolute_error" (MAE) instead of the default squared-error: a
    # standard, well-justified statistical choice (blowout games shouldn't
    # dominate training the way squared error lets them) — NOT a
    # profit-optimized objective. Considered and rejected a genuinely
    # profit-aware custom loss (external prior art reviewed) specifically
    # because training directly on backtested profit would blur the safe
    # separation this project relies on between "predict the game"
    # (statistical objective) and "is there profit" (backtest evaluation) —
    # that separation is what keeps the model from curve-fitting to the
    # historical odds sample. Measured honestly: MAE vs squared-error moved
    # margin/total correlation and backtest ROI by less than their own
    # standard error — a real but marginal improvement, kept because it's
    # safe and well-motivated, not because it moved the needle.
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=3,
        max_iter=150,
        learning_rate=0.05,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame          # indexed by game_id: predicted_margin, predicted_total (OOS only)
    margin_residual_std: float
    total_residual_std: float
    seasons_predicted: list = field(default_factory=list)


def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: list,
    season_col: str = "season",
    margin_col: str = "actual_margin",
    total_col: str = "actual_total",
    game_id_col: str = "game_id",
    min_train_seasons: int = 3,
) -> WalkForwardResult:
    """
    Returns out-of-sample predictions for every season after the first
    `min_train_seasons` seasons in the data (those seasons are training-only
    — too little history exists to have produced a walk-forward prediction
    for them, so the backtest correctly excludes them rather than pretending
    they were "predicted").
    """
    seasons = sorted(df[season_col].unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError("Not enough seasons of data to walk-forward train.")

    X_all = df[feature_cols].astype(float)
    pred_rows = []
    seasons_predicted = []

    for i in range(min_train_seasons, len(seasons)):
        target_season = seasons[i]
        train_mask = df[season_col] < target_season
        test_mask = df[season_col] == target_season
        if train_mask.sum() < 50:
            continue

        margin_model = _make_model().fit(X_all[train_mask], df.loc[train_mask, margin_col])
        total_model = _make_model().fit(X_all[train_mask], df.loc[train_mask, total_col])

        pred_margin = margin_model.predict(X_all[test_mask])
        pred_total = total_model.predict(X_all[test_mask])

        chunk = pd.DataFrame({
            game_id_col: df.loc[test_mask, game_id_col].values,
            "predicted_margin": pred_margin,
            "predicted_total": pred_total,
        })
        pred_rows.append(chunk)
        seasons_predicted.append(target_season)

    predictions = pd.concat(pred_rows, ignore_index=True).set_index(game_id_col)

    merged = df.set_index(game_id_col).join(predictions, how="inner")
    margin_resid_std = float((merged[margin_col] - merged["predicted_margin"]).std())
    total_resid_std = float((merged[total_col] - merged["predicted_total"]).std())

    return WalkForwardResult(
        predictions=predictions,
        margin_residual_std=margin_resid_std,
        total_residual_std=total_resid_std,
        seasons_predicted=seasons_predicted,
    )


@dataclass
class FinalModels:
    """Models fit on the ENTIRE historical dataset — used only to score brand-new
    (manually entered) upcoming games, never to backtest, since every historical
    game is legitimately "the past" relative to a new game today."""
    margin_model: HistGradientBoostingRegressor
    total_model: HistGradientBoostingRegressor
    feature_cols: list

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        X = features_df[self.feature_cols].astype(float)
        return pd.DataFrame({
            "predicted_margin": self.margin_model.predict(X),
            "predicted_total": self.total_model.predict(X),
        }, index=features_df.index)


def fit_final_models(df: pd.DataFrame, feature_cols: list,
                      margin_col: str = "actual_margin", total_col: str = "actual_total") -> FinalModels:
    X = df[feature_cols].astype(float)
    margin_model = _make_model().fit(X, df[margin_col])
    total_model = _make_model().fit(X, df[total_col])
    return FinalModels(margin_model=margin_model, total_model=total_model, feature_cols=feature_cols)


def compute_feature_importance(df: pd.DataFrame, feature_cols: list,
                                margin_col: str = "actual_margin", total_col: str = "actual_total",
                                n_repeats: int = 10, random_state: int = 42) -> dict:
    """
    What the model is ACTUALLY relying on, not a guess about it — sklearn's
    permutation importance: shuffle one feature column at a time, measure
    how much worse the model's predictions get, average over `n_repeats`
    shuffles. Works on HistGradientBoostingRegressor (which has no built-in
    `.feature_importances_` the way a plain random forest does) with no new
    dependency, since `sklearn.inspection` ships with scikit-learn already.

    This is a diagnostic, not a hypothesis test — it doesn't go through
    `core.research.evaluate_hypothesis()` (there's no "baseline vs variant"
    here, it's introspection on the model as it already exists), but its
    real, honest purpose is the same as everything in `research.py`:
    telling you what to investigate next based on actual evidence instead
    of intuition. A feature with near-zero importance is either genuinely
    useless or redundant with something else the model already has; a
    feature with high importance validates that it was worth adding.

    Fits a single model on the full dataset (not walk-forward — this is
    about understanding the model's CURRENT reliance on each input, not
    producing an out-of-sample prediction) and returns both models' ranked
    importances.
    """
    from sklearn.inspection import permutation_importance

    X = df[feature_cols].astype(float)
    result = {}
    for target_name, target_col in [("margin", margin_col), ("total", total_col)]:
        model = _make_model().fit(X, df[target_col])
        perm = permutation_importance(model, X, df[target_col], n_repeats=n_repeats,
                                       random_state=random_state, scoring="r2")
        ranked = sorted(
            zip(feature_cols, perm.importances_mean, perm.importances_std),
            key=lambda t: t[1], reverse=True,
        )
        result[target_name] = [
            {"feature": f, "importance_mean": round(float(m), 5), "importance_std": round(float(s), 5)}
            for f, m, s in ranked
        ]
    return result
