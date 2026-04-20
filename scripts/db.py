"""
Слой работы с базами данных.
indicators.db — исходные данные (только чтение).
models.db     — результаты обучения (запись).
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
INDICATORS_DB = ROOT / "data-base" / "indicators.db"
MODELS_DB     = ROOT / "data-base" / "models.db"


# ---------------------------------------------------------------------------
# Соединения
# ---------------------------------------------------------------------------

def get_indicators_conn() -> sqlite3.Connection:
    return sqlite3.connect(INDICATORS_DB)


def get_models_conn() -> sqlite3.Connection:
    return sqlite3.connect(MODELS_DB)


# ---------------------------------------------------------------------------
# Чтение из indicators.db
# ---------------------------------------------------------------------------

def load_dataset(indicator_ids: list[int] | None = None) -> pd.DataFrame:
    """
    Возвращает широкую таблицу: строки — годы, столбцы — названия показателей.
    Если indicator_ids=None — загружает все показатели.
    """
    with get_indicators_conn() as conn:
        if indicator_ids:
            placeholders = ",".join("?" * len(indicator_ids))
            query = f"""
                SELECT d.year, i.name, d.value
                FROM Dataset d
                JOIN Indicator i ON d.id_indicator = i.id_indicator
                WHERE d.id_indicator IN ({placeholders})
                ORDER BY d.year
            """
            df_long = pd.read_sql_query(query, conn, params=indicator_ids)
        else:
            query = """
                SELECT d.year, i.name, d.value
                FROM Dataset d
                JOIN Indicator i ON d.id_indicator = i.id_indicator
                ORDER BY d.year
            """
            df_long = pd.read_sql_query(query, conn)

    df_wide = df_long.pivot(index="year", columns="name", values="value").reset_index()
    df_wide = df_wide.rename(columns={"year": "Год"})
    df_wide.columns.name = None
    return df_wide


def get_indicator_id(name: str) -> int:
    """Возвращает id_indicator по точному названию."""
    with get_indicators_conn() as conn:
        row = conn.execute(
            "SELECT id_indicator FROM Indicator WHERE name = ?", (name,)
        ).fetchone()
    if row is None:
        raise ValueError(f"Показатель не найден: {name!r}")
    return row[0]


def get_all_indicators() -> pd.DataFrame:
    """Возвращает таблицу всех показателей с id, именем и сферой."""
    with get_indicators_conn() as conn:
        return pd.read_sql_query(
            """
            SELECT i.id_indicator, s.name AS sphere, i.name, i.unit
            FROM Indicator i
            JOIN Sphere s ON i.id_sphere = s.id_sphere
            ORDER BY i.id_indicator
            """,
            conn,
        )


# ---------------------------------------------------------------------------
# Запись в models.db
# ---------------------------------------------------------------------------

def save_run(
    forecast_horizon: int,
    train_start_year: int,
    train_end_year: int,
    status: str = "pending",
    error_message: str | None = None,
) -> int:
    """Создаёт запись ForecastRun, возвращает id_run."""
    with get_models_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO ForecastRun
                (forecast_horizon, train_start_year, train_end_year, status, error_message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (forecast_horizon, train_start_year, train_end_year, status, error_message),
        )
        conn.commit()
        return cur.lastrowid


def update_run_status(id_run: int, status: str, error_message: str | None = None) -> None:
    with get_models_conn() as conn:
        conn.execute(
            "UPDATE ForecastRun SET status = ?, error_message = ? WHERE id_run = ?",
            (status, error_message, id_run),
        )
        conn.commit()


def save_run_indicator_roles(id_run: int, roles: dict[int, str]) -> None:
    """roles = {id_indicator: 'target' | 'feature'}"""
    with get_models_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO RunIndicatorRole (id_run, id_indicator, role) VALUES (?, ?, ?)",
            [(id_run, iid, role) for iid, role in roles.items()],
        )
        conn.commit()


def save_model(
    id_run: int,
    model_name: str,
    model_type: str,
    algorithm: str,
    status: str,
    model_path: str | None = None,
) -> int:
    """Создаёт запись Model, возвращает id_model."""
    with get_models_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO Model
                (id_run, model_name, model_type, algorithm, status, model_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (id_run, model_name, model_type, algorithm, status, model_path),
        )
        conn.commit()
        return cur.lastrowid


def update_model_path(id_model: int, model_path: str) -> None:
    with get_models_conn() as conn:
        conn.execute(
            "UPDATE Model SET model_path = ? WHERE id_model = ?",
            (model_path, id_model),
        )
        conn.commit()


def save_metrics(id_model: int, mae: float, rmse: float, mape: float) -> None:
    with get_models_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO ModelMetric (id_model, mae, rmse, mape)
            VALUES (?, ?, ?, ?)
            """,
            (id_model, mae, rmse, mape),
        )
        conn.commit()


def save_forecast_results(
    id_model: int,
    rows: list[dict],
) -> dict[tuple[int, str], int]:
    """
    rows — список словарей с ключами: year, scenario_name, forecast_value.
    Возвращает маппинг (year, scenario_name) -> id_result.
    """
    result_ids: dict[tuple[int, str], int] = {}
    with get_models_conn() as conn:
        for row in rows:
            cur = conn.execute(
                """
                INSERT INTO ForecastResult (id_model, year, scenario_name, forecast_value)
                VALUES (?, ?, ?, ?)
                """,
                (id_model, row["year"], row["scenario_name"], row["forecast_value"]),
            )
            result_ids[(row["year"], row["scenario_name"])] = cur.lastrowid
        conn.commit()
    return result_ids


def save_shap_contributions(
    contributions: list[dict],
) -> None:
    """
    contributions — список словарей:
        id_result, id_indicator, contribution_value, direction, rank_position
    """
    with get_models_conn() as conn:
        conn.executemany(
            """
            INSERT INTO ShapContribution
                (id_result, id_indicator, contribution_value, direction, rank_position)
            VALUES (:id_result, :id_indicator, :contribution_value, :direction, :rank_position)
            """,
            contributions,
        )
        conn.commit()
