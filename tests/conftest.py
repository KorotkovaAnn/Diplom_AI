"""
Общие фикстуры для всех тестов.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from forecast_utils import SELECTED_FEATURES, TARGET_COLUMN, YEAR_COLUMN


# ---------------------------------------------------------------------------
# Схемы БД
# ---------------------------------------------------------------------------

INDICATORS_SCHEMA = """
CREATE TABLE Sphere (
    id_sphere INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE Indicator (
    id_indicator INTEGER PRIMARY KEY AUTOINCREMENT,
    id_sphere INTEGER NOT NULL,
    name TEXT NOT NULL,
    unit TEXT NOT NULL,
    FOREIGN KEY (id_sphere) REFERENCES Sphere(id_sphere)
);
CREATE TABLE Dataset (
    id_indicator INTEGER NOT NULL,
    year INTEGER NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (id_indicator, year),
    FOREIGN KEY (id_indicator) REFERENCES Indicator(id_indicator)
);
"""

MODELS_SCHEMA = """
CREATE TABLE ForecastRun (
    id_run INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_horizon INTEGER NOT NULL,
    train_start_year INTEGER NOT NULL,
    train_end_year INTEGER NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT
);
CREATE TABLE RunIndicatorRole (
    id_run INTEGER NOT NULL,
    id_indicator INTEGER NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY (id_run, id_indicator),
    FOREIGN KEY (id_run) REFERENCES ForecastRun(id_run)
);
CREATE TABLE Model (
    id_model INTEGER PRIMARY KEY AUTOINCREMENT,
    id_run INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    status TEXT NOT NULL,
    model_path TEXT,
    FOREIGN KEY (id_run) REFERENCES ForecastRun(id_run)
);
CREATE TABLE ModelMetric (
    id_metric INTEGER PRIMARY KEY AUTOINCREMENT,
    id_model INTEGER NOT NULL UNIQUE,
    mae REAL,
    rmse REAL,
    mape REAL,
    FOREIGN KEY (id_model) REFERENCES Model(id_model)
);
CREATE TABLE ForecastResult (
    id_result INTEGER PRIMARY KEY AUTOINCREMENT,
    id_model INTEGER NOT NULL,
    year INTEGER NOT NULL,
    scenario_name TEXT NOT NULL,
    forecast_value REAL NOT NULL,
    FOREIGN KEY (id_model) REFERENCES Model(id_model)
);
CREATE TABLE ShapContribution (
    id_contribution INTEGER PRIMARY KEY AUTOINCREMENT,
    id_result INTEGER NOT NULL,
    id_indicator INTEGER NOT NULL,
    contribution_value REAL NOT NULL,
    direction TEXT NOT NULL,
    rank_position INTEGER NOT NULL,
    FOREIGN KEY (id_result) REFERENCES ForecastResult(id_result)
);
"""


@pytest.fixture
def indicators_conn():
    """In-memory indicators.db с тестовыми данными."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDICATORS_SCHEMA)

    conn.execute("INSERT INTO Sphere (name) VALUES ('Экономика')")
    conn.execute("INSERT INTO Sphere (name) VALUES ('Цены')")

    # Целевой показатель — id=1
    conn.execute(
        "INSERT INTO Indicator (id_sphere, name, unit) VALUES (1, ?, 'млн. руб.')",
        (TARGET_COLUMN,),
    )
    # Один признак — id=2
    conn.execute(
        "INSERT INTO Indicator (id_sphere, name, unit) VALUES (2, ?, '%')",
        (SELECTED_FEATURES[0],),
    )

    years = list(range(2000, 2024))
    for y in years:
        conn.execute("INSERT INTO Dataset VALUES (1, ?, ?)", (y, float(y * 1000)))
        conn.execute("INSERT INTO Dataset VALUES (2, ?, ?)", (y, float(y * 10)))

    conn.commit()
    return conn


@pytest.fixture
def models_conn():
    """In-memory models.db с чистой схемой."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(MODELS_SCHEMA)
    conn.commit()
    return conn


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Минимальный DataFrame с нужными колонками:
    TARGET_COLUMN + все SELECTED_FEATURES, 20 лет.
    """
    rng = np.random.default_rng(42)
    n = 20
    years = list(range(2000, 2000 + n))

    data = {YEAR_COLUMN: years, TARGET_COLUMN: rng.uniform(100_000, 400_000, n)}
    for feat in SELECTED_FEATURES:
        data[feat] = rng.uniform(1_000, 50_000, n)

    return pd.DataFrame(data)
