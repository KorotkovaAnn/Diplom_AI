"""
FastAPI-сервер: читает indicators.db и models.db, отдаёт JSON фронту.
Запуск: .venv/bin/uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scripts.db import get_indicators_conn, get_models_conn

app = FastAPI(title="Investment Forecast API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic-модели ответов
# ---------------------------------------------------------------------------

class IndicatorOut(BaseModel):
    id: int
    sphere: str
    name: str
    unit: str


class HistoryPoint(BaseModel):
    year: int
    value: float


class ModelOut(BaseModel):
    id: int
    run_id: int
    name: str
    type: str
    algorithm: str
    status: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


class ForecastPoint(BaseModel):
    indicator_id: int
    year: int
    scenario: str
    value: float


class ShapItem(BaseModel):
    indicator_id: int
    indicator_name: str
    contribution: float
    direction: str
    rank: int


class StatsOut(BaseModel):
    indicator_count: int
    target_indicator_count: int
    forecast_horizon: int
    best_mape: Optional[float]
    model_count: int


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _fetch_best_model(conn) -> tuple[int, int] | tuple[None, None]:
    """Возвращает (id_model, id_run) лучшей модели по MAPE."""
    row = conn.execute("""
        SELECT m.id_model, m.id_run
        FROM Model m
        JOIN ModelMetric mm ON m.id_model = mm.id_model
        WHERE m.status = 'trained'
        ORDER BY mm.mape ASC
        LIMIT 1
    """).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _fetch_model_target_indicator_id(conn, model_id: int) -> int:
    """Возвращает target id_indicator для модели через связанный ForecastRun."""
    row = conn.execute(
        """
        SELECT rr.id_indicator
        FROM Model m
        JOIN RunIndicatorRole rr ON rr.id_run = m.id_run AND rr.role = 'target'
        WHERE m.id_model = ?
        """,
        (model_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(404, f"Target-показатель для модели {model_id} не найден")
    return row[0]


def _indicator_names(indicator_ids: list[int]) -> dict[int, str]:
    """Читает имена показателей из indicators.db по списку id."""
    if not indicator_ids:
        return {}
    placeholders = ",".join("?" * len(indicator_ids))
    with get_indicators_conn() as conn:
        rows = conn.execute(
            f"SELECT id_indicator, name FROM Indicator WHERE id_indicator IN ({placeholders})",
            indicator_ids,
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def _row_to_model(row) -> ModelOut:
    return ModelOut(
        id=row[0], run_id=row[1], name=row[2], type=row[3],
        algorithm=row[4], status=row[5], mae=row[6], rmse=row[7], mape=row[8],
    )


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/api/indicators", response_model=list[IndicatorOut])
def list_indicators(
    role: Optional[str] = Query(None, description="target | feature; если не указан — все показатели"),
):
    """Показатели с их сферой и единицей измерения."""
    indicator_ids: list[int] | None = None
    if role is not None:
        if role not in {"target", "feature"}:
            raise HTTPException(400, "role должен быть target или feature")

        with get_models_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT id_indicator FROM RunIndicatorRole WHERE role = ?",
                (role,),
            ).fetchall()
            indicator_ids = [r[0] for r in rows]

        if not indicator_ids:
            return []

    with get_indicators_conn() as conn:
        params: list[int] = []
        where_clause = ""
        if indicator_ids is not None:
            placeholders = ",".join("?" * len(indicator_ids))
            where_clause = f"WHERE i.id_indicator IN ({placeholders})"
            params = indicator_ids

        rows = conn.execute(
            f"""
                SELECT i.id_indicator, s.name, i.name, i.unit
                FROM Indicator i
                JOIN Sphere s ON i.id_sphere = s.id_sphere
                {where_clause}
                ORDER BY i.id_sphere, i.id_indicator
            """,
            params,
        ).fetchall()
    return [IndicatorOut(id=r[0], sphere=r[1], name=r[2], unit=r[3]) for r in rows]


@app.get("/api/indicators/{indicator_id}/history", response_model=list[HistoryPoint])
def indicator_history(indicator_id: int):
    """Исторический ряд значений показателя."""
    with get_indicators_conn() as conn:
        rows = conn.execute(
            "SELECT year, value FROM Dataset WHERE id_indicator = ? ORDER BY year",
            (indicator_id,),
        ).fetchall()
    if not rows:
        raise HTTPException(404, f"Нет данных для показателя id={indicator_id}")
    return [HistoryPoint(year=r[0], value=r[1]) for r in rows]


@app.get("/api/models", response_model=list[ModelOut])
def list_models():
    """Все обученные модели с метриками, отсортированы по MAPE."""
    with get_models_conn() as conn:
        rows = conn.execute("""
            SELECT m.id_model, m.id_run, m.model_name, m.model_type, m.algorithm, m.status,
                   mm.mae, mm.rmse, mm.mape
            FROM Model m
            LEFT JOIN ModelMetric mm ON m.id_model = mm.id_model
            WHERE m.status = 'trained'
            ORDER BY mm.mape ASC
        """).fetchall()
    return [_row_to_model(r) for r in rows]


@app.get("/api/models/best", response_model=ModelOut)
def best_model():
    """Лучшая модель по MAPE."""
    with get_models_conn() as conn:
        row = conn.execute("""
            SELECT m.id_model, m.id_run, m.model_name, m.model_type, m.algorithm, m.status,
                   mm.mae, mm.rmse, mm.mape
            FROM Model m
            JOIN ModelMetric mm ON m.id_model = mm.id_model
            WHERE m.status = 'trained'
            ORDER BY mm.mape ASC
            LIMIT 1
        """).fetchone()
    if not row:
        raise HTTPException(404, "Нет обученных моделей")
    return _row_to_model(row)


@app.get("/api/forecasts", response_model=list[ForecastPoint])
def forecasts(
    model_id: Optional[int] = Query(None, description="id модели; если не указан — берётся лучшая"),
    indicator_id: Optional[int] = Query(None, description="target id_indicator; если не указан — все target модели"),
    scenario: Optional[str] = Query(None, description="Базовый | Оптимистичный | Пессимистичный"),
):
    """Прогнозные значения. Без фильтров — все сценарии лучшей модели."""
    with get_models_conn() as conn:
        if model_id is None:
            mid, _ = _fetch_best_model(conn)
            if mid is None:
                raise HTTPException(404, "Нет обученных моделей")
            model_id = mid

        params: list = [model_id]
        sql = "SELECT id_indicator, year, scenario_name, forecast_value FROM ForecastResult WHERE id_model = ?"
        if indicator_id is not None:
            sql += " AND id_indicator = ?"
            params.append(indicator_id)
        if scenario:
            sql += " AND scenario_name = ?"
            params.append(scenario)
        sql += " ORDER BY scenario_name, year"

        rows = conn.execute(sql, params).fetchall()

    return [ForecastPoint(indicator_id=r[0], year=r[1], scenario=r[2], value=r[3]) for r in rows]


@app.get("/api/shap", response_model=list[ShapItem])
def shap_contributions(
    model_id: int = Query(...),
    indicator_id: Optional[int] = Query(None, description="target id_indicator; если не указан — target модели"),
    year: int = Query(...),
    scenario: str = Query(...),
):
    """SHAP-вклады факторов для конкретной модели, года и сценария."""
    with get_models_conn() as conn:
        target_indicator_id = (
            indicator_id
            if indicator_id is not None
            else _fetch_model_target_indicator_id(conn, model_id)
        )
        result_row = conn.execute(
            """
            SELECT id_result
            FROM ForecastResult
            WHERE id_model=? AND id_indicator=? AND year=? AND scenario_name=?
            """,
            (model_id, target_indicator_id, year, scenario),
        ).fetchone()
        if not result_row:
            raise HTTPException(
                404,
                f"Нет прогноза: model={model_id}, year={year}, scenario={scenario}",
            )
        rows = conn.execute(
            """SELECT id_indicator, contribution_value, direction, rank_position
               FROM ShapContribution WHERE id_result = ? ORDER BY rank_position""",
            (result_row[0],),
        ).fetchall()

    names = _indicator_names([r[0] for r in rows])
    return [
        ShapItem(
            indicator_id=r[0],
            indicator_name=names.get(r[0], f"Показатель {r[0]}"),
            contribution=r[1],
            direction=r[2],
            rank=r[3],
        )
        for r in rows
    ]


@app.get("/api/dashboard")
def dashboard(
    model_id: Optional[int] = Query(None, description="id модели; если не указан — лучшая по MAPE"),
    indicator_id: Optional[int] = Query(None, description="target id_indicator; если не указан — target модели"),
    scenario: str = Query("Базовый", description="Базовый | Оптимистичный | Пессимистичный"),
):
    """
    Всё необходимое для DashboardsPage за один запрос:
    - target_indicator  — целевой показатель
    - history           — исторические значения (из indicators.db)
    - forecasts         — прогнозы по всем сценариям
    - shap              — SHAP-вклады для каждого (год, сценарий)
    - best_model        — метаданные модели
    """
    with get_models_conn() as conn:
        # Модель
        if model_id is None:
            mid, id_run = _fetch_best_model(conn)
            if mid is None:
                raise HTTPException(404, "Нет обученных моделей")
            model_id = mid
        else:
            row = conn.execute("SELECT id_run FROM Model WHERE id_model=?", (model_id,)).fetchone()
            if not row:
                raise HTTPException(404, f"Модель {model_id} не найдена")
            id_run = row[0]

        model_row = conn.execute("""
            SELECT m.id_model, m.id_run, m.model_name, m.model_type, m.algorithm, m.status,
                   mm.mae, mm.rmse, mm.mape
            FROM Model m LEFT JOIN ModelMetric mm ON m.id_model = mm.id_model
            WHERE m.id_model = ?
        """, (model_id,)).fetchone()
        model_info = _row_to_model(model_row)

        # Целевой показатель из RunIndicatorRole
        target_row = conn.execute(
            "SELECT id_indicator FROM RunIndicatorRole WHERE id_run=? AND role='target'",
            (id_run,),
        ).fetchone()
        target_id = indicator_id if indicator_id is not None else (target_row[0] if target_row else None)

        # Все прогнозы для модели
        forecast_rows = conn.execute(
            """SELECT year, scenario_name, forecast_value, id_result
               FROM ForecastResult
               WHERE id_model=? AND id_indicator=?
               ORDER BY scenario_name, year""",
            (model_id, target_id),
        ).fetchall()

        # SHAP для каждого (год, сценарий)
        shap_raw: dict[str, list] = {}
        for year, sc_name, _, id_result in forecast_rows:
            key = f"{year}_{sc_name}"
            shap_items = conn.execute(
                """SELECT id_indicator, contribution_value, direction, rank_position
                   FROM ShapContribution WHERE id_result=? ORDER BY rank_position""",
                (id_result,),
            ).fetchall()
            if shap_items:
                shap_raw[key] = shap_items

    # Данные из indicators.db
    with get_indicators_conn() as conn:
        # Целевой показатель
        if target_id:
            t = conn.execute("""
                SELECT i.id_indicator, s.name, i.name, i.unit
                FROM Indicator i JOIN Sphere s ON i.id_sphere = s.id_sphere
                WHERE i.id_indicator = ?
            """, (target_id,)).fetchone()
            target_indicator = IndicatorOut(id=t[0], sphere=t[1], name=t[2], unit=t[3])
            hist_rows = conn.execute(
                "SELECT year, value FROM Dataset WHERE id_indicator=? ORDER BY year",
                (target_id,),
            ).fetchall()
        else:
            target_indicator = None
            hist_rows = []

        # Имена факторов для SHAP
        all_ind_ids = {r[0] for items in shap_raw.values() for r in items}
        names = _indicator_names(list(all_ind_ids))

    # Формируем прогнозы по сценариям
    forecasts_by_scenario: dict[str, list] = {}
    for year, sc_name, val, _ in forecast_rows:
        forecasts_by_scenario.setdefault(sc_name, []).append({"year": year, "value": val})

    # Формируем SHAP
    shap_out: dict[str, list] = {}
    for key, items in shap_raw.items():
        shap_out[key] = [
            {
                "indicator_id": r[0],
                "indicator_name": names.get(r[0], f"Показатель {r[0]}"),
                "contribution": r[1],
                "direction": r[2],
                "rank": r[3],
            }
            for r in items
        ]

    return {
        "target_indicator": target_indicator.model_dump() if target_indicator else None,
        "history": [{"year": r[0], "value": r[1]} for r in hist_rows],
        "forecasts": forecasts_by_scenario,
        "shap": shap_out,
        "model": model_info.model_dump(),
    }


@app.get("/api/stats", response_model=StatsOut)
def stats():
    """Сводная статистика для главной страницы."""
    with get_indicators_conn() as conn:
        fallback_indicator_count = conn.execute("SELECT COUNT(*) FROM Indicator").fetchone()[0]

    with get_models_conn() as conn:
        indicator_count_row = conn.execute("""
            SELECT COUNT(DISTINCT id_indicator)
            FROM RunIndicatorRole
            WHERE role IN ('target', 'feature')
        """).fetchone()
        indicator_count = (
            indicator_count_row[0]
            if indicator_count_row and indicator_count_row[0]
            else fallback_indicator_count
        )
        target_indicator_count = conn.execute("""
            SELECT COUNT(DISTINCT id_indicator)
            FROM RunIndicatorRole
            WHERE role = 'target'
        """).fetchone()[0]
        model_count = conn.execute(
            "SELECT COUNT(*) FROM Model WHERE status='trained'"
        ).fetchone()[0]
        best_mape_row = conn.execute(
            "SELECT MIN(mape) FROM ModelMetric"
        ).fetchone()
        best_mape = best_mape_row[0] if best_mape_row else None
        horizon_row = conn.execute(
            "SELECT MAX(forecast_horizon) FROM ForecastRun WHERE status='completed'"
        ).fetchone()
        horizon = horizon_row[0] if horizon_row and horizon_row[0] else 4

    return StatsOut(
        indicator_count=indicator_count,
        target_indicator_count=target_indicator_count,
        forecast_horizon=horizon,
        best_mape=best_mape,
        model_count=model_count,
    )
