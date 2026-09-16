"""
Unit tests for pipeline.py's _safe_add_trailing_feature() -- the graceful-
degradation fix for a real production incident, 2026-09-16: the app owner
reported "/api/ratings?sport=NFL" and "?sport=CFB" both 404ing. Root cause:
both sports' trailing-feature datasets (added to R2 well after this
project's original bulk dataset set) were never actually uploaded there,
so a Volume without them locally hit a real, by-design FileNotFoundError
inside load_team_game_epa()/load_team_game_success() -- which took down
the ENTIRE sport's pipeline build (main.py's own try/except around
pipeline building kept the app itself up, but that sport's routes 404'd).
See pipeline.py's own docstring on _safe_add_trailing_feature() for the
full incident writeup.
"""

import pandas as pd
import pytest

from pipeline import _safe_add_trailing_feature


class TestSafeAddTrailingFeature:
    def test_returns_the_builders_real_result_on_success(self):
        games = pd.DataFrame({"game_id": ["g0", "g1"], "x": [1, 2]})

        def real_builder(feats):
            out = feats.copy()
            out["new_col"] = 99.0
            return out

        out = _safe_add_trailing_feature(games, real_builder, ["new_col"], "test feature")
        assert list(out["new_col"]) == [99.0, 99.0]
        assert len(out) == 2  # never drops rows

    def test_falls_back_to_zero_filled_columns_on_file_not_found(self):
        games = pd.DataFrame({"game_id": ["g0", "g1"], "x": [1, 2]})

        def broken_builder(feats):
            raise FileNotFoundError("dataset not uploaded to R2 yet")

        out = _safe_add_trailing_feature(games, broken_builder, ["home_x_trail", "away_x_trail"], "test feature")
        assert list(out["home_x_trail"]) == [0.0, 0.0]
        assert list(out["away_x_trail"]) == [0.0, 0.0]
        assert len(out) == 2  # the rest of the pipeline still gets a real, full-row DataFrame

    def test_never_drops_or_reorders_existing_columns(self):
        games = pd.DataFrame({"game_id": ["g0", "g1"], "keep_me": ["a", "b"]})

        def broken_builder(feats):
            raise FileNotFoundError("simulated")

        out = _safe_add_trailing_feature(games, broken_builder, ["new_col"], "test feature")
        assert list(out["game_id"]) == ["g0", "g1"]
        assert list(out["keep_me"]) == ["a", "b"]

    def test_only_catches_file_not_found_not_other_exceptions(self):
        games = pd.DataFrame({"game_id": ["g0"]})

        def other_broken_builder(feats):
            raise ValueError("a real bug, not a missing-dataset situation")

        with pytest.raises(ValueError):
            _safe_add_trailing_feature(games, other_broken_builder, ["col"], "test feature")
