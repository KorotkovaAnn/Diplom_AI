"""
Тесты для вспомогательных функций scripts/train_ml.py.

Полное обучение не запускается — тестируем только утилиты:
LogTargetWrapper, make_pipeline, build_fit_predict.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from train_ml import LogTargetWrapper, make_pipeline, build_fit_predict
from forecast_utils import SELECTED_FEATURES, TARGET_COLUMN, YEAR_COLUMN


# ---------------------------------------------------------------------------
# Фикстура: маленький датасет (20 наблюдений, 3 признака)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_xy():
    rng = np.random.default_rng(0)
    n = 20
    X = rng.uniform(1_000, 50_000, (n, 3)).astype(float)
    y = rng.uniform(100_000, 400_000, n).astype(float)
    return X, y


# ---------------------------------------------------------------------------
# make_pipeline
# ---------------------------------------------------------------------------

class TestMakePipeline:
    def test_returns_pipeline(self):
        pipe = make_pipeline(Ridge())
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_three_steps(self):
        pipe = make_pipeline(Ridge())
        assert len(pipe.steps) == 3

    def test_step_names(self):
        pipe = make_pipeline(Ridge())
        names = [name for name, _ in pipe.steps]
        assert names == ["imputer", "scaler", "model"]

    def test_pipeline_fits_and_predicts(self, tiny_xy):
        X, y = tiny_xy
        pipe = make_pipeline(Ridge())
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(y),)

    def test_pipeline_handles_nan(self):
        X = np.array([[1.0, np.nan, 3.0],
                      [4.0, 5.0, np.nan],
                      [7.0, 8.0, 9.0]])
        y = np.array([10.0, 20.0, 30.0])
        pipe = make_pipeline(Ridge())
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# LogTargetWrapper
# ---------------------------------------------------------------------------

class TestLogTargetWrapper:
    def test_fit_predict_returns_correct_shape(self, tiny_xy):
        X, y = tiny_xy
        pipe = make_pipeline(Ridge())
        wrapper = LogTargetWrapper(pipe)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert preds.shape == (len(y),)

    def test_predictions_are_positive(self, tiny_xy):
        """log1p → expm1 гарантирует положительные предсказания."""
        X, y = tiny_xy
        wrapper = LogTargetWrapper(make_pipeline(Ridge()))
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert (preds > 0).all()

    def test_predictions_are_in_reasonable_range(self, tiny_xy):
        """Предсказания не должны быть экстремально далеко от реальных."""
        X, y = tiny_xy
        wrapper = LogTargetWrapper(make_pipeline(Ridge()))
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        # Допускаем 10-кратное отклонение от диапазона обучающих данных
        assert preds.max() < y.max() * 10
        assert preds.min() > y.min() / 10

    def test_log_wrapper_differs_from_plain(self, tiny_xy):
        """Обёртка с логарифмом должна давать другие предсказания."""
        X, y = tiny_xy
        pipe_plain = make_pipeline(Ridge())
        pipe_plain.fit(X, y)

        wrapper = LogTargetWrapper(make_pipeline(Ridge()))
        wrapper.fit(X, y)

        preds_plain = pipe_plain.predict(X)
        preds_log   = wrapper.predict(X)
        assert not np.allclose(preds_plain, preds_log)


# ---------------------------------------------------------------------------
# build_fit_predict
# ---------------------------------------------------------------------------

class TestBuildFitPredict:
    def test_returns_callable(self):
        pipe = make_pipeline(Ridge())
        fn = build_fit_predict(pipe, use_log=False)
        assert callable(fn)

    def test_returns_float(self, tiny_xy):
        X, y = tiny_xy
        pipe = make_pipeline(Ridge())
        fn = build_fit_predict(pipe, use_log=False)
        result = fn(X[:-1], y[:-1], X[-1:])
        assert isinstance(result, float)

    def test_log_variant_returns_float(self, tiny_xy):
        X, y = tiny_xy
        pipe = make_pipeline(Ridge())
        fn = build_fit_predict(pipe, use_log=True)
        result = fn(X[:-1], y[:-1], X[-1:])
        assert isinstance(result, float)

    def test_log_variant_returns_positive(self, tiny_xy):
        X, y = tiny_xy
        pipe = make_pipeline(Ridge())
        fn = build_fit_predict(pipe, use_log=True)
        result = fn(X[:-1], y[:-1], X[-1:])
        assert result > 0

    def test_consistent_predictions(self, tiny_xy):
        """Одни и те же данные → одинаковый результат."""
        X, y = tiny_xy
        pipe = make_pipeline(Ridge())
        fn = build_fit_predict(pipe, use_log=False)
        r1 = fn(X[:-1], y[:-1], X[-1:])
        r2 = fn(X[:-1], y[:-1], X[-1:])
        assert r1 == pytest.approx(r2)
