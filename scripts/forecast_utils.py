"""
Утилиты для сценарного прогнозирования.
Общая логика, используемая и ML-, и LSTM-скриптами.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

YEAR_COLUMN   = "Год"
TARGET_COLUMN = "Объем инвестиций в основной капитал"
FORECAST_HORIZON = 4
TREND_WINDOW     = 3

SELECTED_FEATURES = [
    "Потребительские расходы в среднем на душу населения",
    "Ввод в действие основных фондов",
    "Среднедушевые денежные доходы населения в год",
    "Оборот розничной торговли",
    "Среднемесячная номинальная начисленная заработная плата работников организаций",
    "Численность населения",
    "Стоимость основных фондов по Тюм обл, на конец года, по полной учетной стоимости",
    "Задолженность по кредитам в рублях, предоставленным кредитными организациями физическим лицам",
    "Валовой региональный продукт на душу населения",
    "Население в трудоспособном возрасте",
    "Индексы потребительских цен",
    "Удельный вес убыточных организаций",
]

# Негативные факторы — при сценарии "вниз" усиливаются, при "вверх" снижаются
NEGATIVE_SCENARIO_FEATURES = {
    "Индексы потребительских цен",
    "Удельный вес убыточных организаций",
}

DIRECTION_MULTIPLIERS = {
    "up":   {"positive": 0.60, "negative": 0.90},
    "down": {"positive": -0.40, "negative": 1.50},
}

SCENARIOS_CONFIG = {
    "Базовый": {
        "type": "fixed",
        "positive_multiplier": 1.05,
        "negative_multiplier": 0.95,
    },
    "Консервативный": {
        "type": "auto_pattern",
    },
}


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Авто-детекция паттерна консервативного сценария
# ---------------------------------------------------------------------------

def detect_conservative_pattern(df: pd.DataFrame) -> list[str]:
    """
    Определяет паттерн ['up'/'down', ...] для шагов 2–4 консервативного сценария
    по двум последним годовым изменениям целевого показателя.
    """
    target_series = df[TARGET_COLUMN].dropna()
    if len(target_series) < 2:
        return ["up", "up", "down"]

    last_two = target_series.iloc[-2:]
    delta1 = float(last_two.iloc[-1]) - float(last_two.iloc[-2])

    if len(target_series) >= 3:
        prev_two = target_series.iloc[-3:-1]
        delta0 = float(prev_two.iloc[-1]) - float(prev_two.iloc[-2])
    else:
        delta0 = delta1

    up_up   = delta0 >= 0 and delta1 >= 0
    down_down = delta0 < 0 and delta1 < 0

    if up_up:
        return ["up", "up", "down"]
    elif down_down:
        return ["down", "down", "up"]
    else:
        return ["down", "up", "up"]


# ---------------------------------------------------------------------------
# Вспомогательные функции прогноза
# ---------------------------------------------------------------------------

def compute_annual_slopes(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    slopes: dict[str, float] = {}
    for feature in features:
        history = df[[YEAR_COLUMN, feature]].dropna().tail(TREND_WINDOW)
        if len(history) >= 2:
            x = history[YEAR_COLUMN].to_numpy(dtype=float)
            y = history[feature].to_numpy(dtype=float)
            slope, _ = np.polyfit(x, y, deg=1)
            slopes[feature] = float(slope)
        else:
            slopes[feature] = 0.0
    return slopes


def get_step_multiplier(
    feature: str,
    scenario_cfg: dict,
    step: int,
    conservative_pattern: list[str],
) -> float:
    if scenario_cfg["type"] == "fixed":
        if feature in NEGATIVE_SCENARIO_FEATURES:
            return scenario_cfg["negative_multiplier"]
        return scenario_cfg["positive_multiplier"]

    # auto_pattern: step=1 использует паттерн[0], step=2 → [1], step=3 → [2]
    pattern_idx = max(0, step - 2)
    direction = conservative_pattern[pattern_idx]
    if feature in NEGATIVE_SCENARIO_FEATURES:
        return DIRECTION_MULTIPLIERS[direction]["negative"]
    return DIRECTION_MULTIPLIERS[direction]["positive"]


def build_future_exogenous_rows(
    df: pd.DataFrame,
    scenario_name: str,
    scenario_cfg: dict,
    conservative_pattern: list[str],
    features: list[str] | None = None,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Строит DataFrame с экзогенными переменными на горизонт прогноза.
    TARGET_COLUMN заполнен NaN — будет подставлен моделью.
    """
    if features is None:
        features = SELECTED_FEATURES

    latest_year = int(df[YEAR_COLUMN].max())
    slopes = compute_annual_slopes(df, features)
    running = {f: float(df[f].dropna().iloc[-1]) for f in features}

    rows = []
    for step in range(1, horizon + 1):
        future_year = latest_year + step
        row: dict = {YEAR_COLUMN: future_year, TARGET_COLUMN: np.nan, "scenario": scenario_name}

        for feature in features:
            if step == 1:
                new_val = running[feature] + slopes[feature]
            else:
                m = get_step_multiplier(feature, scenario_cfg, step, conservative_pattern)
                new_val = running[feature] + slopes[feature] * m
            row[feature] = new_val
            running[feature] = new_val

        rows.append(row)

    return pd.DataFrame(rows)


def build_all_scenario_frames(
    df: pd.DataFrame,
    features: list[str] | None = None,
    horizon: int = FORECAST_HORIZON,
) -> dict[str, pd.DataFrame]:
    """
    Возвращает {scenario_name: future_df} для всех сценариев из SCENARIOS_CONFIG.
    """
    conservative_pattern = detect_conservative_pattern(df)
    result = {}
    for name, cfg in SCENARIOS_CONFIG.items():
        result[name] = build_future_exogenous_rows(
            df, name, cfg, conservative_pattern, features, horizon
        )
    return result


# ---------------------------------------------------------------------------
# Walk-forward валидация (общая)
# ---------------------------------------------------------------------------

def walk_forward_validate(
    data: pd.DataFrame,
    feature_columns: list[str],
    fit_predict_fn,
    min_train_size: int = 12,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    fit_predict_fn(X_train, y_train, X_test) -> float

    Возвращает (summary_df, detail_rows).
    """
    valid_data = data[[YEAR_COLUMN, TARGET_COLUMN, *feature_columns]].dropna().reset_index(drop=True)
    detail_rows = []

    for split_idx in range(min_train_size, len(valid_data)):
        train_part = valid_data.iloc[:split_idx]
        test_part  = valid_data.iloc[split_idx : split_idx + 1]

        X_tr = train_part[feature_columns].to_numpy()
        y_tr = train_part[TARGET_COLUMN].to_numpy()
        X_te = test_part[feature_columns].to_numpy()
        y_te = float(test_part[TARGET_COLUMN].iloc[0])

        pred = float(fit_predict_fn(X_tr, y_tr, X_te))

        detail_rows.append({
            "predicted_year":  int(test_part[YEAR_COLUMN].iloc[0]),
            "train_end_year":  int(train_part[YEAR_COLUMN].iloc[-1]),
            "actual":          y_te,
            "predicted":       pred,
            "abs_error":       abs(y_te - pred),
            "ape":             abs(y_te - pred) / y_te if y_te != 0 else np.nan,
        })

    y_true = np.array([r["actual"]    for r in detail_rows])
    y_pred = np.array([r["predicted"] for r in detail_rows])
    metrics = evaluate_predictions(y_true, y_pred)

    summary = pd.DataFrame([{
        "folds":              len(detail_rows),
        "walk_forward_mae":   metrics["mae"],
        "walk_forward_rmse":  metrics["rmse"],
        "walk_forward_mape":  metrics["mape"],
    }])
    return summary, detail_rows
