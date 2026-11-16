"""
Unit tests for core/ml_models.py's `total_loss` parameter -- added
2026-11-07 (app owner: "take a look at this and make any necessary
changes to make the algo better") after finding a real, measured bias in
NHL's total model under this project's project-wide default MAE loss
(see sports/nhl/config.py's TOTAL_MODEL_LOSS and sports/nhl/
research_total_model_loss.py for the full real-data investigation).

These tests are deliberately narrow and fast (no real pipeline build,
small synthetic data) -- they check that the parameter is actually
threaded through to the fitted model's own `loss` attribute and that the
margin model is NEVER affected by it, not that squared-error "works
better" (that's what the research script + decision_log.jsonl entry are
for, on real data).
"""

import numpy as np
import pandas as pd

from core.ml_models import _make_model, fit_final_models, walk_forward_predict


def _synthetic_df(n_seasons=5, games_per_season=40, n_features=3):
    rows = []
    rng = np.random.RandomState(0)
    gid = 0
    for season in range(2020, 2020 + n_seasons):
        for _ in range(games_per_season):
            feats = rng.normal(size=n_features)
            rows.append({
                "game_id": f"g{gid}", "season": season,
                "actual_margin": float(feats[0] * 3 + rng.normal()),
                "actual_total": float(45 + feats[1] * 5 + rng.normal()),
                **{f"f{i}": float(feats[i]) for i in range(n_features)},
            })
            gid += 1
    return pd.DataFrame(rows)


FEATURE_COLS = ["f0", "f1", "f2"]


class TestMakeModelLossParameter:
    def test_default_loss_is_absolute_error(self):
        model = _make_model()
        assert model.loss == "absolute_error"

    def test_explicit_loss_is_respected(self):
        model = _make_model(loss="squared_error")
        assert model.loss == "squared_error"


class TestWalkForwardPredictTotalLoss:
    def test_default_total_loss_matches_prior_behavior(self):
        df = _synthetic_df()
        result = walk_forward_predict(df, FEATURE_COLS)
        assert len(result.predictions) > 0  # just confirms nothing broke with the new optional param

    def test_custom_total_loss_runs_without_error_and_changes_predictions(self):
        df = _synthetic_df()
        default_result = walk_forward_predict(df, FEATURE_COLS, total_loss="absolute_error")
        squared_result = walk_forward_predict(df, FEATURE_COLS, total_loss="squared_error")
        # Real regression check: a different loss on a skewed-enough
        # synthetic target should actually produce different predictions,
        # not silently reuse the same model (confirms the parameter is
        # genuinely wired through, not a no-op default argument).
        assert not np.allclose(
            default_result.predictions["predicted_total"].values,
            squared_result.predictions["predicted_total"].values,
        )

    def test_margin_predictions_are_identical_regardless_of_total_loss(self):
        # total_loss must ONLY affect the total model -- margin always
        # uses _make_model()'s own default, unconditionally.
        df = _synthetic_df()
        default_result = walk_forward_predict(df, FEATURE_COLS, total_loss="absolute_error")
        squared_result = walk_forward_predict(df, FEATURE_COLS, total_loss="squared_error")
        np.testing.assert_array_equal(
            default_result.predictions["predicted_margin"].values,
            squared_result.predictions["predicted_margin"].values,
        )


class TestFitFinalModelsTotalLoss:
    def test_total_loss_is_passed_through_to_the_fitted_total_model(self):
        df = _synthetic_df()
        models = fit_final_models(df, FEATURE_COLS, total_loss="squared_error")
        assert models.total_model.loss == "squared_error"

    def test_margin_model_is_unaffected_by_total_loss(self):
        df = _synthetic_df()
        models = fit_final_models(df, FEATURE_COLS, total_loss="squared_error")
        assert models.margin_model.loss == "absolute_error"

    def test_default_total_loss_matches_prior_behavior(self):
        df = _synthetic_df()
        models = fit_final_models(df, FEATURE_COLS)
        assert models.total_model.loss == "absolute_error"
