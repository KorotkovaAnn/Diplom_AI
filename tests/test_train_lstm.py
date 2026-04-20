"""
Тесты для вспомогательных функций scripts/train_lstm.py.

Полное обучение не запускается — тестируем утилиты:
add_dynamic_features, get_dynamic_feature_columns,
transform_target / inverse_transform_target,
prepare_sequences, make_recency_weights, build_lstm_model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.train_lstm import (
    DYNAMIC_BASE_FEATURES,
    add_dynamic_features,
    get_dynamic_feature_columns,
    inverse_transform_target,
    make_recency_weights,
    prepare_sequences,
    transform_target,
    build_lstm_model,
    build_lstm_window,
)
from scripts.forecast_utils import SELECTED_FEATURES, TARGET_COLUMN, YEAR_COLUMN


# ---------------------------------------------------------------------------
# add_dynamic_features
# ---------------------------------------------------------------------------

class TestAddDynamicFeatures:
    def test_original_columns_preserved(self, sample_df):
        df_dyn = add_dynamic_features(sample_df)
        for col in sample_df.columns:
            assert col in df_dyn.columns

    def test_new_columns_added(self, sample_df):
        # Добавляем хотя бы одну базовую колонку из DYNAMIC_BASE_FEATURES
        col = DYNAMIC_BASE_FEATURES[0]
        sample_df[col] = np.arange(len(sample_df), dtype=float)
        df_dyn = add_dynamic_features(sample_df)
        assert f"{col}_diff"     in df_dyn.columns
        assert f"{col}_pct"      in df_dyn.columns
        assert f"{col}_roll3"    in df_dyn.columns
        assert f"{col}_momentum" in df_dyn.columns

    def test_diff_is_correct(self, sample_df):
        col = DYNAMIC_BASE_FEATURES[0]
        sample_df[col] = np.arange(len(sample_df), dtype=float) * 100
        df_dyn = add_dynamic_features(sample_df)
        expected = pd.Series(np.arange(len(sample_df), dtype=float) * 100).diff()
        np.testing.assert_allclose(
            df_dyn[f"{col}_diff"].values,
            expected.values,
            equal_nan=True,
        )

    def test_does_not_modify_original(self, sample_df):
        col = DYNAMIC_BASE_FEATURES[0]
        sample_df[col] = 1.0
        original_cols = set(sample_df.columns)
        _ = add_dynamic_features(sample_df)
        assert set(sample_df.columns) == original_cols

    def test_missing_base_column_skipped(self, sample_df):
        """Если базовой колонки нет — просто не добавляем производные."""
        col = DYNAMIC_BASE_FEATURES[0]
        if col in sample_df.columns:
            sample_df = sample_df.drop(columns=[col])
        df_dyn = add_dynamic_features(sample_df)
        assert f"{col}_diff" not in df_dyn.columns


# ---------------------------------------------------------------------------
# get_dynamic_feature_columns
# ---------------------------------------------------------------------------

class TestGetDynamicFeatureColumns:
    def test_includes_selected_features(self, sample_df):
        col = DYNAMIC_BASE_FEATURES[0]
        sample_df[col] = 1.0
        df_dyn = add_dynamic_features(sample_df)
        feats = get_dynamic_feature_columns(df_dyn)
        for f in SELECTED_FEATURES:
            assert f in feats

    def test_includes_dynamic_extras(self, sample_df):
        col = DYNAMIC_BASE_FEATURES[0]
        sample_df[col] = 1.0
        df_dyn = add_dynamic_features(sample_df)
        feats = get_dynamic_feature_columns(df_dyn)
        assert f"{col}_diff" in feats

    def test_no_duplicates(self, sample_df):
        col = DYNAMIC_BASE_FEATURES[0]
        sample_df[col] = 1.0
        df_dyn = add_dynamic_features(sample_df)
        feats = get_dynamic_feature_columns(df_dyn)
        assert len(feats) == len(set(feats))


# ---------------------------------------------------------------------------
# transform_target / inverse_transform_target
# ---------------------------------------------------------------------------

class TestTransformTarget:
    def test_raw_mode_is_identity(self):
        y = np.array([[100.0], [200.0], [300.0]])
        out = transform_target(y, "raw")
        np.testing.assert_array_equal(out, y)

    def test_log_mode_applies_log1p(self):
        y = np.array([[100.0], [200.0]])
        out = transform_target(y, "log")
        np.testing.assert_allclose(out, np.log1p(y))

    def test_inverse_raw_roundtrip(self):
        y = np.array([[100_000.0], [200_000.0], [300_000.0]])
        scaler = StandardScaler()
        y_t = transform_target(y, "raw")
        scaler.fit(y_t)
        y_scaled = scaler.transform(y_t).flatten()

        y_recovered = inverse_transform_target(y_scaled, scaler, "raw")
        np.testing.assert_allclose(y_recovered, y.flatten(), rtol=1e-5)

    def test_inverse_log_roundtrip(self):
        y = np.array([[100_000.0], [200_000.0], [300_000.0]])
        scaler = StandardScaler()
        y_t = transform_target(y, "log")
        scaler.fit(y_t)
        y_scaled = scaler.transform(y_t).flatten()

        y_recovered = inverse_transform_target(y_scaled, scaler, "log")
        np.testing.assert_allclose(y_recovered, y.flatten(), rtol=1e-4)


# ---------------------------------------------------------------------------
# make_recency_weights
# ---------------------------------------------------------------------------

class TestMakeRecencyWeights:
    def test_returns_same_length(self):
        years = np.array([2000, 2001, 2002, 2003])
        w = make_recency_weights(years)
        assert len(w) == len(years)

    def test_all_positive(self):
        years = np.arange(2000, 2020)
        w = make_recency_weights(years)
        assert (w > 0).all()

    def test_latest_year_has_highest_weight(self):
        years = np.arange(2000, 2010)
        w = make_recency_weights(years)
        assert w[-1] == w.max()

    def test_weights_sum_to_n(self):
        years = np.arange(2000, 2015)
        w = make_recency_weights(years)
        assert w.sum() == pytest.approx(len(years), rel=1e-5)

    def test_monotonically_increasing(self):
        years = np.arange(2000, 2010)
        w = make_recency_weights(years)
        assert (np.diff(w) > 0).all()


# ---------------------------------------------------------------------------
# prepare_sequences
# ---------------------------------------------------------------------------

class TestPrepareSequences:
    N_FEATURES = 3
    FEATURES   = SELECTED_FEATURES[:N_FEATURES]

    def _make_df(self, n: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        data = {YEAR_COLUMN: range(2000, 2000 + n),
                TARGET_COLUMN: rng.uniform(100_000, 400_000, n)}
        for f in self.FEATURES:
            data[f] = rng.uniform(1_000, 50_000, n)
        return pd.DataFrame(data)

    def test_output_shapes(self):
        df = self._make_df(20)
        window = 4
        X, y, yrs, arts = prepare_sequences(df, df, self.FEATURES, window, "raw")
        assert X.shape == (20 - window, window, self.N_FEATURES)
        assert y.shape == (20 - window,)
        assert yrs.shape == (20 - window,)

    def test_artifacts_keys(self):
        df = self._make_df(20)
        _, _, _, arts = prepare_sequences(df, df, self.FEATURES, 4, "raw")
        assert set(arts.keys()) == {
            "feature_imputer", "target_imputer",
            "feature_scaler", "target_scaler",
            "input_columns", "target_mode",
        }

    def test_input_columns_without_target_history(self):
        df = self._make_df(20)
        _, _, _, arts = prepare_sequences(df, df, self.FEATURES, 4, "raw", include_target_history=False)
        assert TARGET_COLUMN not in arts["input_columns"]

    def test_input_columns_with_target_history(self):
        df = self._make_df(20)
        _, _, _, arts = prepare_sequences(df, df, self.FEATURES, 4, "raw", include_target_history=True)
        assert TARGET_COLUMN in arts["input_columns"]
        assert arts["input_columns"][0] == TARGET_COLUMN

    def test_scaled_y_has_zero_mean(self):
        """После StandardScaler среднее ≈ 0."""
        df = self._make_df(20)
        _, y, _, _ = prepare_sequences(df, df, self.FEATURES, 4, "raw")
        assert abs(y.mean()) < 2.0  # нестрого из-за малого размера выборки

    def test_log_mode_target_is_positive_before_scaling(self):
        df = self._make_df(20)
        # log1p(положительных значений) > 0, поэтому обратное преобразование > 0
        _, _, _, arts = prepare_sequences(df, df, self.FEATURES, 4, "log")
        assert arts["target_mode"] == "log"

    def test_different_train_and_source(self):
        """Можно передать разные train_fit_df и sequence_source_df."""
        df_train  = self._make_df(15)
        df_source = self._make_df(20)
        X, y, yrs, _ = prepare_sequences(df_train, df_source, self.FEATURES, 4, "raw")
        assert X.shape[0] == 20 - 4


# ---------------------------------------------------------------------------
# build_lstm_model
# ---------------------------------------------------------------------------

class TestBuildLstmModel:
    @pytest.mark.parametrize("architecture", ["small", "medium", "stacked"])
    def test_model_compiles(self, architecture):
        import tensorflow as tf
        model = build_lstm_model(window_size=4, n_features=3, architecture=architecture)
        assert isinstance(model, tf.keras.Model)

    def test_output_shape(self):
        model = build_lstm_model(window_size=4, n_features=3, architecture="small")
        dummy = np.zeros((2, 4, 3), dtype=np.float32)
        out = model.predict(dummy, verbose=0)
        assert out.shape == (2, 1)

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Неизвестная архитектура"):
            build_lstm_model(4, 3, "unknown_arch")


# ---------------------------------------------------------------------------
# build_lstm_window
# ---------------------------------------------------------------------------

class TestBuildLstmWindow:
    def _make_artifacts(self, df, features, window_size):
        _, _, _, arts = prepare_sequences(df, df, features, window_size, "raw")
        return arts

    def test_output_shape(self, sample_df):
        features   = SELECTED_FEATURES[:3]
        window_size = 4
        arts = self._make_artifacts(sample_df, features, window_size)
        window = build_lstm_window(sample_df, arts, window_size)
        assert window.shape == (1, window_size, len(arts["input_columns"]))

    def test_uses_last_n_rows(self, sample_df):
        """Окно не должно зависеть от строк в начале датафрейма."""
        features    = SELECTED_FEATURES[:3]
        window_size = 4
        arts = self._make_artifacts(sample_df, features, window_size)

        w_full = build_lstm_window(sample_df, arts, window_size)
        w_tail = build_lstm_window(sample_df.iloc[5:], arts, window_size)

        # Оба должны использовать только последние window_size строк
        assert w_full.shape == w_tail.shape
