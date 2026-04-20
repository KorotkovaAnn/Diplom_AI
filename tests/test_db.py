"""
Тесты для scripts/db.py.

Реальные файлы БД не трогаются — все функции работают с in-memory
соединениями, которые подставляются через monkeypatch.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.db as db
from scripts.forecast_utils import SELECTED_FEATURES, TARGET_COLUMN


# ---------------------------------------------------------------------------
# Хелперы — подменяем коннекторы на in-memory из conftest
# ---------------------------------------------------------------------------

def patch_indicators(conn):
    return patch("scripts.db.get_indicators_conn", return_value=conn)


def patch_models(conn):
    return patch("scripts.db.get_models_conn", return_value=conn)


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_returns_wide_dataframe(self, indicators_conn):
        with patch_indicators(indicators_conn):
            df = db.load_dataset()

        assert "Год" in df.columns
        assert TARGET_COLUMN in df.columns
        assert SELECTED_FEATURES[0] in df.columns

    def test_year_column_is_not_index(self, indicators_conn):
        with patch_indicators(indicators_conn):
            df = db.load_dataset()

        assert df.index.name != "Год"
        assert "Год" in df.columns

    def test_row_count_matches_years(self, indicators_conn):
        with patch_indicators(indicators_conn):
            df = db.load_dataset()

        assert len(df) == 24  # 2000–2023

    def test_filter_by_indicator_ids(self, indicators_conn):
        with patch_indicators(indicators_conn):
            df = db.load_dataset(indicator_ids=[1])

        assert TARGET_COLUMN in df.columns
        assert SELECTED_FEATURES[0] not in df.columns

    def test_values_are_numeric(self, indicators_conn):
        with patch_indicators(indicators_conn):
            df = db.load_dataset()

        assert df[TARGET_COLUMN].dtype == float


# ---------------------------------------------------------------------------
# get_indicator_id
# ---------------------------------------------------------------------------

class TestGetIndicatorId:
    def test_returns_correct_id(self, indicators_conn):
        with patch_indicators(indicators_conn):
            iid = db.get_indicator_id(TARGET_COLUMN)

        assert iid == 1

    def test_raises_for_unknown_name(self, indicators_conn):
        with patch_indicators(indicators_conn):
            with pytest.raises(ValueError, match="не найден"):
                db.get_indicator_id("Несуществующий показатель")


# ---------------------------------------------------------------------------
# get_all_indicators
# ---------------------------------------------------------------------------

class TestGetAllIndicators:
    def test_columns_present(self, indicators_conn):
        with patch_indicators(indicators_conn):
            df = db.get_all_indicators()

        assert set(df.columns) >= {"id_indicator", "sphere", "name", "unit"}

    def test_returns_all_indicators(self, indicators_conn):
        with patch_indicators(indicators_conn):
            df = db.get_all_indicators()

        assert len(df) == 2


# ---------------------------------------------------------------------------
# save_run / update_run_status
# ---------------------------------------------------------------------------

class TestForecastRun:
    def test_save_run_returns_id(self, models_conn):
        with patch_models(models_conn):
            id_run = db.save_run(4, 2000, 2023, "pending")

        assert isinstance(id_run, int)
        assert id_run > 0

    def test_save_run_persists_data(self, models_conn):
        with patch_models(models_conn):
            id_run = db.save_run(4, 2000, 2023, "pending")

        row = models_conn.execute(
            "SELECT forecast_horizon, train_start_year, train_end_year, status FROM ForecastRun WHERE id_run=?",
            (id_run,),
        ).fetchone()
        assert row == (4, 2000, 2023, "pending")

    def test_update_run_status(self, models_conn):
        with patch_models(models_conn):
            id_run = db.save_run(4, 2000, 2023, "pending")
            db.update_run_status(id_run, "completed")

        status = models_conn.execute(
            "SELECT status FROM ForecastRun WHERE id_run=?", (id_run,)
        ).fetchone()[0]
        assert status == "completed"

    def test_update_run_status_with_error(self, models_conn):
        with patch_models(models_conn):
            id_run = db.save_run(4, 2000, 2023, "running")
            db.update_run_status(id_run, "error", "что-то пошло не так")

        row = models_conn.execute(
            "SELECT status, error_message FROM ForecastRun WHERE id_run=?", (id_run,)
        ).fetchone()
        assert row == ("error", "что-то пошло не так")

    def test_save_run_indicator_roles(self, models_conn):
        with patch_models(models_conn):
            id_run = db.save_run(4, 2000, 2023, "pending")
            db.save_run_indicator_roles(id_run, {1: "target", 2: "feature"})

        rows = models_conn.execute(
            "SELECT id_indicator, role FROM RunIndicatorRole WHERE id_run=? ORDER BY id_indicator",
            (id_run,),
        ).fetchall()
        assert rows == [(1, "target"), (2, "feature")]


# ---------------------------------------------------------------------------
# save_model / update_model_path
# ---------------------------------------------------------------------------

class TestModel:
    def _make_run(self, models_conn) -> int:
        with patch_models(models_conn):
            return db.save_run(4, 2000, 2023, "running")

    def test_save_model_returns_id(self, models_conn):
        id_run = self._make_run(models_conn)
        with patch_models(models_conn):
            id_model = db.save_model(id_run, "Ridge", "machine_learning", "Ridge", "trained")

        assert isinstance(id_model, int)
        assert id_model > 0

    def test_save_model_persists_fields(self, models_conn):
        id_run = self._make_run(models_conn)
        with patch_models(models_conn):
            id_model = db.save_model(
                id_run, "Ridge", "machine_learning", "Ridge", "trained", "models/ml/ridge.pkl"
            )

        row = models_conn.execute(
            "SELECT model_name, model_type, algorithm, status, model_path FROM Model WHERE id_model=?",
            (id_model,),
        ).fetchone()
        assert row == ("Ridge", "machine_learning", "Ridge", "trained", "models/ml/ridge.pkl")

    def test_update_model_path(self, models_conn):
        id_run = self._make_run(models_conn)
        with patch_models(models_conn):
            id_model = db.save_model(id_run, "Ridge", "ml", "Ridge", "trained")
            db.update_model_path(id_model, "models/ml/ridge_v2.pkl")

        path = models_conn.execute(
            "SELECT model_path FROM Model WHERE id_model=?", (id_model,)
        ).fetchone()[0]
        assert path == "models/ml/ridge_v2.pkl"


# ---------------------------------------------------------------------------
# save_metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def _make_model(self, models_conn) -> int:
        with patch_models(models_conn):
            id_run = db.save_run(4, 2000, 2023, "running")
            return db.save_model(id_run, "Ridge", "ml", "Ridge", "trained")

    def test_save_metrics_persists(self, models_conn):
        id_model = self._make_model(models_conn)
        with patch_models(models_conn):
            db.save_metrics(id_model, mae=1000.0, rmse=1200.0, mape=0.05)

        row = models_conn.execute(
            "SELECT mae, rmse, mape FROM ModelMetric WHERE id_model=?", (id_model,)
        ).fetchone()
        assert row == (1000.0, 1200.0, 0.05)

    def test_save_metrics_upsert(self, models_conn):
        """Повторный вызов обновляет, не дублирует."""
        id_model = self._make_model(models_conn)
        with patch_models(models_conn):
            db.save_metrics(id_model, 1000.0, 1200.0, 0.05)
            db.save_metrics(id_model, 500.0, 600.0, 0.02)

        count = models_conn.execute(
            "SELECT COUNT(*) FROM ModelMetric WHERE id_model=?", (id_model,)
        ).fetchone()[0]
        assert count == 1

        mape = models_conn.execute(
            "SELECT mape FROM ModelMetric WHERE id_model=?", (id_model,)
        ).fetchone()[0]
        assert mape == 0.02


# ---------------------------------------------------------------------------
# save_forecast_results
# ---------------------------------------------------------------------------

class TestForecastResults:
    def _make_model(self, models_conn) -> int:
        with patch_models(models_conn):
            id_run = db.save_run(4, 2000, 2023, "running")
            return db.save_model(id_run, "Ridge", "ml", "Ridge", "trained")

    def test_returns_result_ids_mapping(self, models_conn):
        id_model = self._make_model(models_conn)
        rows = [
            {"year": 2024, "scenario_name": "Базовый",       "forecast_value": 350_000.0},
            {"year": 2024, "scenario_name": "Консервативный", "forecast_value": 310_000.0},
            {"year": 2025, "scenario_name": "Базовый",       "forecast_value": 370_000.0},
        ]
        with patch_models(models_conn):
            result_ids = db.save_forecast_results(id_model, rows)

        assert (2024, "Базовый")       in result_ids
        assert (2024, "Консервативный") in result_ids
        assert (2025, "Базовый")       in result_ids

    def test_persists_correct_values(self, models_conn):
        id_model = self._make_model(models_conn)
        with patch_models(models_conn):
            result_ids = db.save_forecast_results(
                id_model,
                [{"year": 2024, "scenario_name": "Базовый", "forecast_value": 350_000.0}],
            )

        id_result = result_ids[(2024, "Базовый")]
        row = models_conn.execute(
            "SELECT year, scenario_name, forecast_value FROM ForecastResult WHERE id_result=?",
            (id_result,),
        ).fetchone()
        assert row == (2024, "Базовый", 350_000.0)


# ---------------------------------------------------------------------------
# save_shap_contributions
# ---------------------------------------------------------------------------

class TestShapContributions:
    def _make_result(self, models_conn) -> int:
        with patch_models(models_conn):
            id_run   = db.save_run(4, 2000, 2023, "running")
            id_model = db.save_model(id_run, "Ridge", "ml", "Ridge", "trained")
            result_ids = db.save_forecast_results(
                id_model,
                [{"year": 2024, "scenario_name": "Базовый", "forecast_value": 350_000.0}],
            )
        return result_ids[(2024, "Базовый")]

    def test_saves_contributions(self, models_conn):
        id_result = self._make_result(models_conn)
        contribs = [
            {"id_result": id_result, "id_indicator": 2, "contribution_value": 5000.0,
             "direction": "positive", "rank_position": 1},
            {"id_result": id_result, "id_indicator": 3, "contribution_value": -2000.0,
             "direction": "negative", "rank_position": 2},
        ]
        with patch_models(models_conn):
            db.save_shap_contributions(contribs)

        count = models_conn.execute(
            "SELECT COUNT(*) FROM ShapContribution WHERE id_result=?", (id_result,)
        ).fetchone()[0]
        assert count == 2

    def test_contribution_rank_order(self, models_conn):
        id_result = self._make_result(models_conn)
        contribs = [
            {"id_result": id_result, "id_indicator": 2, "contribution_value": 5000.0,
             "direction": "positive", "rank_position": 1},
            {"id_result": id_result, "id_indicator": 3, "contribution_value": -2000.0,
             "direction": "negative", "rank_position": 2},
        ]
        with patch_models(models_conn):
            db.save_shap_contributions(contribs)

        rows = models_conn.execute(
            "SELECT rank_position, direction FROM ShapContribution WHERE id_result=? ORDER BY rank_position",
            (id_result,),
        ).fetchall()
        assert rows[0] == (1, "positive")
        assert rows[1] == (2, "negative")
