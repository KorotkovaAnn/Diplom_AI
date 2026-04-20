"""
Тесты для scripts/forecast_utils.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.forecast_utils import (
    DIRECTION_MULTIPLIERS,
    FORECAST_HORIZON,
    NEGATIVE_SCENARIO_FEATURES,
    SCENARIOS_CONFIG,
    SELECTED_FEATURES,
    TARGET_COLUMN,
    TREND_WINDOW,
    YEAR_COLUMN,
    build_all_scenario_frames,
    build_future_exogenous_rows,
    compute_annual_slopes,
    detect_conservative_pattern,
    evaluate_predictions,
    get_step_multiplier,
    walk_forward_validate,
)


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------

class TestEvaluatePredictions:
    def test_perfect_prediction_gives_zero_errors(self):
        y = np.array([100.0, 200.0, 300.0])
        metrics = evaluate_predictions(y, y)
        assert metrics["mae"]  == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mape"] == pytest.approx(0.0)

    def test_returns_all_keys(self):
        y = np.array([100.0, 200.0])
        metrics = evaluate_predictions(y, y * 1.1)
        assert set(metrics.keys()) == {"mae", "rmse", "mape"}

    def test_mae_greater_than_zero_for_imperfect(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 320.0])
        metrics = evaluate_predictions(y_true, y_pred)
        assert metrics["mae"] > 0

    def test_rmse_geq_mae(self):
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([150.0, 180.0, 250.0, 500.0])
        metrics = evaluate_predictions(y_true, y_pred)
        assert metrics["rmse"] >= metrics["mae"]

    def test_mape_is_relative(self):
        y_true = np.array([1000.0, 2000.0])
        y_pred = np.array([1100.0, 2200.0])
        metrics = evaluate_predictions(y_true, y_pred)
        assert metrics["mape"] == pytest.approx(0.1, rel=1e-3)


# ---------------------------------------------------------------------------
# detect_conservative_pattern
# ---------------------------------------------------------------------------

class TestDetectConservativePattern:
    def _df(self, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({
            YEAR_COLUMN: range(2000, 2000 + len(values)),
            TARGET_COLUMN: values,
        })

    def test_up_up_returns_up_up_down(self):
        df = self._df([100, 200, 300])
        assert detect_conservative_pattern(df) == ["up", "up", "down"]

    def test_down_down_returns_down_down_up(self):
        df = self._df([300, 200, 100])
        assert detect_conservative_pattern(df) == ["down", "down", "up"]

    def test_mixed_returns_down_up_up(self):
        df = self._df([200, 300, 200])  # ↑ ↓
        assert detect_conservative_pattern(df) == ["down", "up", "up"]

    def test_mixed_down_up_returns_down_up_up(self):
        df = self._df([300, 200, 300])  # ↓ ↑
        assert detect_conservative_pattern(df) == ["down", "up", "up"]

    def test_too_few_rows_returns_default(self):
        df = self._df([100])
        result = detect_conservative_pattern(df)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_returns_list_of_length_three(self, sample_df):
        result = detect_conservative_pattern(sample_df)
        assert len(result) == 3
        assert all(d in ("up", "down") for d in result)


# ---------------------------------------------------------------------------
# compute_annual_slopes
# ---------------------------------------------------------------------------

class TestComputeAnnualSlopes:
    def test_positive_trend_gives_positive_slope(self, sample_df):
        features = [SELECTED_FEATURES[0]]
        sample_df[features[0]] = [float(i * 1000) for i in range(len(sample_df))]
        slopes = compute_annual_slopes(sample_df, features)
        assert slopes[features[0]] > 0

    def test_negative_trend_gives_negative_slope(self, sample_df):
        features = [SELECTED_FEATURES[0]]
        n = len(sample_df)
        sample_df[features[0]] = [float((n - i) * 1000) for i in range(n)]
        slopes = compute_annual_slopes(sample_df, features)
        assert slopes[features[0]] < 0

    def test_flat_trend_gives_near_zero_slope(self, sample_df):
        features = [SELECTED_FEATURES[0]]
        sample_df[features[0]] = 5000.0
        slopes = compute_annual_slopes(sample_df, features)
        assert slopes[features[0]] == pytest.approx(0.0, abs=1e-6)

    def test_uses_only_trend_window(self, sample_df):
        """Наклон считается только по последним TREND_WINDOW годам."""
        features = [SELECTED_FEATURES[0]]
        # Последние TREND_WINDOW лет — строго линейные (+1 в год),
        # остальные — сильно зашумлены, чтобы не влиять на результат
        values = [float(i) for i in range(len(sample_df))]  # 0,1,2,...,19 → наклон = 1.0
        for i in range(len(sample_df) - TREND_WINDOW):
            values[i] = 9_999_999.0
        sample_df[features[0]] = values
        slopes = compute_annual_slopes(sample_df, features)
        assert slopes[features[0]] == pytest.approx(1.0, rel=0.01)

    def test_returns_zero_for_insufficient_data(self, sample_df):
        features = ["__missing__"]
        sample_df["__missing__"] = np.nan
        slopes = compute_annual_slopes(sample_df, features)
        assert slopes["__missing__"] == 0.0


# ---------------------------------------------------------------------------
# get_step_multiplier
# ---------------------------------------------------------------------------

class TestGetStepMultiplier:
    BASE_CFG = SCENARIOS_CONFIG["Базовый"]
    CONS_CFG = SCENARIOS_CONFIG["Консервативный"]
    PATTERN  = ["up", "up", "down"]

    def test_fixed_positive_feature(self):
        feat = SELECTED_FEATURES[0]  # не негативный
        m = get_step_multiplier(feat, self.BASE_CFG, step=2, conservative_pattern=self.PATTERN)
        assert m == self.BASE_CFG["positive_multiplier"]

    def test_fixed_negative_feature(self):
        feat = next(iter(NEGATIVE_SCENARIO_FEATURES))
        m = get_step_multiplier(feat, self.BASE_CFG, step=2, conservative_pattern=self.PATTERN)
        assert m == self.BASE_CFG["negative_multiplier"]

    def test_auto_pattern_up_direction_positive_feature(self):
        feat = SELECTED_FEATURES[0]
        # pattern[0]="up", step=2 → pattern_idx=0
        m = get_step_multiplier(feat, self.CONS_CFG, step=2, conservative_pattern=["up", "up", "down"])
        assert m == DIRECTION_MULTIPLIERS["up"]["positive"]

    def test_auto_pattern_down_direction_negative_feature(self):
        feat = next(iter(NEGATIVE_SCENARIO_FEATURES))
        # pattern[0]="down", step=2 → pattern_idx=0
        m = get_step_multiplier(feat, self.CONS_CFG, step=2, conservative_pattern=["down", "up", "up"])
        assert m == DIRECTION_MULTIPLIERS["down"]["negative"]

    def test_step1_uses_pattern_index_zero(self):
        feat = SELECTED_FEATURES[0]
        m_step1 = get_step_multiplier(feat, self.CONS_CFG, step=1, conservative_pattern=["up", "down", "up"])
        m_step2 = get_step_multiplier(feat, self.CONS_CFG, step=2, conservative_pattern=["up", "down", "up"])
        assert m_step1 == m_step2  # оба используют pattern_idx=0


# ---------------------------------------------------------------------------
# build_future_exogenous_rows
# ---------------------------------------------------------------------------

class TestBuildFutureExogenousRows:
    PATTERN = ["up", "up", "down"]

    def test_returns_correct_horizon(self, sample_df):
        cfg = SCENARIOS_CONFIG["Базовый"]
        future = build_future_exogenous_rows(sample_df, "Базовый", cfg, self.PATTERN, horizon=4)
        assert len(future) == 4

    def test_years_are_sequential(self, sample_df):
        cfg = SCENARIOS_CONFIG["Базовый"]
        last_year = int(sample_df[YEAR_COLUMN].max())
        future = build_future_exogenous_rows(sample_df, "Базовый", cfg, self.PATTERN, horizon=4)
        expected_years = list(range(last_year + 1, last_year + 5))
        assert future[YEAR_COLUMN].tolist() == expected_years

    def test_target_is_nan(self, sample_df):
        cfg = SCENARIOS_CONFIG["Базовый"]
        future = build_future_exogenous_rows(sample_df, "Базовый", cfg, self.PATTERN)
        assert future[TARGET_COLUMN].isna().all()

    def test_features_are_numeric(self, sample_df):
        cfg = SCENARIOS_CONFIG["Базовый"]
        features = SELECTED_FEATURES[:3]
        future = build_future_exogenous_rows(
            sample_df, "Базовый", cfg, self.PATTERN, features=features
        )
        for feat in features:
            assert future[feat].dtype in (float, np.float64)
            assert future[feat].notna().all()

    def test_scenario_column_set_correctly(self, sample_df):
        cfg = SCENARIOS_CONFIG["Консервативный"]
        future = build_future_exogenous_rows(sample_df, "Консервативный", cfg, self.PATTERN)
        assert (future["scenario"] == "Консервативный").all()


# ---------------------------------------------------------------------------
# build_all_scenario_frames
# ---------------------------------------------------------------------------

class TestBuildAllScenarioFrames:
    def test_returns_both_scenarios(self, sample_df):
        frames = build_all_scenario_frames(sample_df, SELECTED_FEATURES, horizon=4)
        assert "Базовый"       in frames
        assert "Консервативный" in frames

    def test_each_frame_has_correct_length(self, sample_df):
        frames = build_all_scenario_frames(sample_df, SELECTED_FEATURES, horizon=3)
        for name, df in frames.items():
            assert len(df) == 3, f"Сценарий {name!r}: ожидалось 3 строки, получено {len(df)}"

    def test_scenarios_differ(self, sample_df):
        """Базовый и консервативный должны давать разные значения признаков."""
        frames = build_all_scenario_frames(sample_df, SELECTED_FEATURES[:1], horizon=4)
        base = frames["Базовый"][SELECTED_FEATURES[0]].values
        cons = frames["Консервативный"][SELECTED_FEATURES[0]].values
        # Хотя бы одна точка должна отличаться
        assert not np.allclose(base, cons)


# ---------------------------------------------------------------------------
# walk_forward_validate
# ---------------------------------------------------------------------------

class TestWalkForwardValidate:
    def _perfect_fn(self, X_train, y_train, X_test):
        """Всегда возвращает последнее значение train-таргета (для детерминизма)."""
        return float(y_train[-1])

    def test_returns_summary_and_details(self, sample_df):
        features = SELECTED_FEATURES[:3]
        summary, details = walk_forward_validate(
            sample_df, features, self._perfect_fn, min_train_size=12
        )
        assert not summary.empty
        assert len(details) > 0

    def test_summary_has_metric_columns(self, sample_df):
        features = SELECTED_FEATURES[:3]
        summary, _ = walk_forward_validate(
            sample_df, features, self._perfect_fn, min_train_size=12
        )
        assert set(summary.columns) >= {"folds", "walk_forward_mae", "walk_forward_rmse", "walk_forward_mape"}

    def test_detail_rows_have_correct_keys(self, sample_df):
        features = SELECTED_FEATURES[:3]
        _, details = walk_forward_validate(
            sample_df, features, self._perfect_fn, min_train_size=12
        )
        expected_keys = {"predicted_year", "train_end_year", "actual", "predicted", "abs_error", "ape"}
        for row in details:
            assert expected_keys.issubset(row.keys())

    def test_folds_count(self, sample_df):
        """Кол-во фолдов = len(data) - min_train_size."""
        features = SELECTED_FEATURES[:3]
        min_train = 12
        valid_len = sample_df[
            [YEAR_COLUMN, TARGET_COLUMN, *features]
        ].dropna().__len__()

        _, details = walk_forward_validate(
            sample_df, features, self._perfect_fn, min_train_size=min_train
        )
        assert len(details) == valid_len - min_train

    def test_metrics_are_non_negative(self, sample_df):
        features = SELECTED_FEATURES[:3]
        summary, _ = walk_forward_validate(
            sample_df, features, self._perfect_fn, min_train_size=12
        )
        assert summary["walk_forward_mae"].iloc[0]  >= 0
        assert summary["walk_forward_rmse"].iloc[0] >= 0
        assert summary["walk_forward_mape"].iloc[0] >= 0
