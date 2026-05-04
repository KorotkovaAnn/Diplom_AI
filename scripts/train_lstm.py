"""
Обучение LSTM-ансамбля.

Скрипт повторяет пайплайн из notebooks/lstm_investment_forecast.ipynb:
    1. Читает данные из indicators.db
    2. Создаёт динамические признаки и удаляет технические начальные NaN
    3. Перебирает 24 конфигурации (feature set x window x architecture x target mode)
    4. Выбирает лучшую из top-5 по walk-forward
    5. Обучает финальный ансамбль на всех доступных годах
    6. Сохраняет .keras, ForecastRun, Model, ModelMetric, ForecastResult, ShapContribution
"""

import itertools
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

import db
from forecast_utils import (
    FORECAST_HORIZON,
    SELECTED_FEATURES,
    TARGET_COLUMN,
    YEAR_COLUMN,
)

MODELS_DIR = ROOT / "models" / "lstm"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Конфигурация из ноутбука
# ---------------------------------------------------------------------------

FULL_FEATURES = list(SELECTED_FEATURES)

COMPACT_FEATURES = [
    "Потребительские расходы в среднем на душу населения",
    "Ввод в действие основных фондов",
    "Среднедушевые денежные доходы населения в год",
    "Среднемесячная номинальная начисленная заработная плата работников организаций",
    "Валовой региональный продукт на душу населения",
    "Индексы потребительских цен",
]

WINDOW_SIZES = [2, 3]
EPOCHS = 300
BATCH_SIZE = 4
LEARNING_RATE = 0.001
L2_REGULARIZATION = 1e-5
RECENCY_WEIGHT_DECAY = 0.90
MIN_TRAIN_SIZE = 12
FINAL_ENSEMBLE_SEEDS = [42, 7, 13, 21, 101]

ARCHITECTURES = ["small", "medium"]
TARGET_MODES = ["level", "log"]
INCLUDE_TARGET_HISTORY_OPTIONS = [True]

TREND_WINDOW = 3
BASE_SCENARIO_TARGET_CAGR = 0.085
BIAS_CORRECTION_WINDOW = 3
BIAS_CORRECTION_STRENGTH = 1.00

SCENARIOS_CONFIG = {
    "Базовый": {
        "type": "fixed",
        "positive_multiplier": 1.10,
        "negative_multiplier": 0.90,
    },
    "Консервативный": {
        "type": "auto_pattern",
    },
}

DIRECTION_MULTIPLIERS = {
    "up": {"positive": 0.60, "negative": 0.90},
    "down": {"positive": -0.40, "negative": 1.50},
}

CONSERVATIVE_PATTERNS = {
    "up_up": ["up", "up", "down"],
    "down_down": ["down", "down", "up"],
    "down_up": ["down", "up", "up"],
    "up_down": ["down", "up", "up"],
}

NEGATIVE_SCENARIO_FEATURES = [
    "Население в трудоспособном возрасте",
    "Индексы потребительских цен",
    "Удельный вес убыточных организаций",
]

DYNAMIC_BASE_COLUMNS = [TARGET_COLUMN, *COMPACT_FEATURES]
DYNAMIC_FEATURES = [
    dynamic_name
    for base_name in DYNAMIC_BASE_COLUMNS
    for dynamic_name in (f"{base_name}__diff_1", f"{base_name}__pct_change_1")
]
DYNAMIC_FEATURES.extend([
    f"{TARGET_COLUMN}__rolling_mean_3",
    f"{TARGET_COLUMN}__momentum_2",
])

# Совместимость с существующими тестами/импортами.
DYNAMIC_BASE_FEATURES = DYNAMIC_BASE_COLUMNS


# ---------------------------------------------------------------------------
# Динамические признаки
# ---------------------------------------------------------------------------

def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Старый helper оставлен для тестов. Для боевого LSTM-пайплайна используется
    add_notebook_dynamic_features с именами признаков из ноутбука.
    """
    df = df.copy()
    for col in DYNAMIC_BASE_FEATURES:
        if col not in df.columns:
            continue
        df[f"{col}_diff"] = df[col].diff()
        df[f"{col}_pct"] = df[col].pct_change()
        df[f"{col}_roll3"] = df[col].rolling(3).mean()
        df[f"{col}_momentum"] = df[col] - df[col].shift(2)
    return df


def get_dynamic_feature_columns(df: pd.DataFrame) -> list[str]:
    base = list(SELECTED_FEATURES)
    extras = [
        c for c in df.columns
        if any(c.startswith(b + "_") for b in DYNAMIC_BASE_FEATURES)
    ]
    return list(dict.fromkeys(base + extras))


def add_notebook_dynamic_features(df: pd.DataFrame, drop_initial_rows: bool = True) -> pd.DataFrame:
    out = df.copy()
    for col in DYNAMIC_BASE_COLUMNS:
        out[f"{col}__diff_1"] = out[col].diff(1)
        out[f"{col}__pct_change_1"] = out[col].pct_change(1).replace([np.inf, -np.inf], np.nan)
    out[f"{TARGET_COLUMN}__rolling_mean_3"] = out[TARGET_COLUMN].rolling(3).mean()
    out[f"{TARGET_COLUMN}__momentum_2"] = out[TARGET_COLUMN].diff(1) - out[TARGET_COLUMN].diff(2)

    if drop_initial_rows:
        out = out.dropna(subset=DYNAMIC_FEATURES).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Target transform / inverse
# ---------------------------------------------------------------------------

def _normalize_target_mode(mode: str) -> str:
    return "level" if mode == "raw" else mode


def transform_target(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y).reshape(-1, 1)
    mode = _normalize_target_mode(mode)
    if mode == "level":
        return y
    if mode == "log":
        return np.log1p(y)
    raise ValueError(f"Неизвестный target mode: {mode}")


def inverse_transform_target(
    y_transformed_scaled: np.ndarray,
    target_scaler: StandardScaler,
    mode: str,
) -> np.ndarray:
    mode = _normalize_target_mode(mode)
    y_transformed = target_scaler.inverse_transform(
        np.asarray(y_transformed_scaled).reshape(-1, 1)
    )
    if mode == "level":
        return y_transformed.flatten()
    if mode == "log":
        return np.expm1(y_transformed).flatten()
    raise ValueError(f"Неизвестный target mode: {mode}")


# ---------------------------------------------------------------------------
# Подготовка последовательностей
# ---------------------------------------------------------------------------

def prepare_sequences(
    train_fit_df: pd.DataFrame,
    sequence_source_df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    target_mode: str,
    include_target_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    input_columns = list(feature_columns)
    if include_target_history and TARGET_COLUMN not in input_columns:
        input_columns = [TARGET_COLUMN, *input_columns]

    feature_imputer = SimpleImputer(strategy="median")
    target_imputer = SimpleImputer(strategy="median")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_imputer.fit(train_fit_df[input_columns])
    target_imputer.fit(train_fit_df[[TARGET_COLUMN]])

    X_train_for_scaler = feature_imputer.transform(train_fit_df[input_columns])
    X_source = feature_imputer.transform(sequence_source_df[input_columns])
    y_train_for_scaler = target_imputer.transform(train_fit_df[[TARGET_COLUMN]])
    y_source = target_imputer.transform(sequence_source_df[[TARGET_COLUMN]])

    y_train_transformed = transform_target(y_train_for_scaler, target_mode)
    y_source_transformed = transform_target(y_source, target_mode)

    feature_scaler.fit(X_train_for_scaler)
    target_scaler.fit(y_train_transformed)

    X_scaled = feature_scaler.transform(X_source)
    y_scaled = target_scaler.transform(y_source_transformed).flatten()
    years = sequence_source_df[YEAR_COLUMN].to_numpy().astype(int)

    X_seq, y_seq, seq_years = [], [], []
    for i in range(window_size, len(sequence_source_df)):
        X_seq.append(X_scaled[i - window_size:i])
        y_seq.append(y_scaled[i])
        seq_years.append(years[i])

    artifacts = {
        "feature_imputer": feature_imputer,
        "target_imputer": target_imputer,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "input_columns": input_columns,
        "target_mode": _normalize_target_mode(target_mode),
    }
    return np.array(X_seq), np.array(y_seq), np.array(seq_years), artifacts


# ---------------------------------------------------------------------------
# LSTM model / fit
# ---------------------------------------------------------------------------

def make_recency_weights(years: np.ndarray, decay: float = RECENCY_WEIGHT_DECAY) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    weights = decay ** (years.max() - years)
    return weights / weights.mean()


def evaluate_arrays(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(mean_absolute_percentage_error(actual, predicted)),
        "mape_percent": float(mean_absolute_percentage_error(actual, predicted) * 100),
    }


def build_lstm_model(window_size: int, n_features: int, architecture: str) -> tf.keras.Model:
    if architecture == "small":
        layers = [
            Input(shape=(window_size, n_features)),
            LSTM(16, kernel_regularizer=l2(L2_REGULARIZATION)),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    elif architecture == "medium":
        layers = [
            Input(shape=(window_size, n_features)),
            LSTM(32, kernel_regularizer=l2(L2_REGULARIZATION)),
            Dropout(0.1),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    elif architecture == "stacked":
        layers = [
            Input(shape=(window_size, n_features)),
            LSTM(32, return_sequences=True, kernel_regularizer=l2(L2_REGULARIZATION)),
            Dropout(0.1),
            LSTM(16, kernel_regularizer=l2(L2_REGULARIZATION)),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    else:
        raise ValueError(f"Неизвестная архитектура: {architecture}")

    model = Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_lstm(
    X_train,
    y_train,
    X_val,
    y_val,
    window_size: int,
    n_features: int,
    architecture: str,
    seed: int = 42,
    sample_weight=None,
) -> tuple[tf.keras.Model, object]:
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = build_lstm_model(window_size, n_features, architecture)
    has_val = X_val is not None and y_val is not None
    monitor = "val_loss" if has_val else "loss"

    callbacks = [
        EarlyStopping(monitor=monitor, patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=12, min_lr=0.00005),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if has_val else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
        sample_weight=sample_weight,
    )
    return model, history


# ---------------------------------------------------------------------------
# Подбор и walk-forward как в ноутбуке
# ---------------------------------------------------------------------------

def make_feature_sets() -> dict[str, list[str]]:
    return {
        "compact": COMPACT_FEATURES,
        "compact_dynamic": COMPACT_FEATURES + DYNAMIC_FEATURES,
        "full": FULL_FEATURES,
    }


def grid_search(model_df: pd.DataFrame) -> pd.DataFrame:
    print("Перебор конфигураций LSTM...")
    feature_sets = make_feature_sets()
    train_df = model_df.iloc[:-2].copy()
    validation_df = model_df.iloc[-2:-1].copy()
    test_df = model_df.iloc[-1:].copy()

    experiment_rows = []
    for feature_set_name, feature_columns in feature_sets.items():
        for window_size, architecture, target_mode, include_target_history in itertools.product(
            WINDOW_SIZES,
            ARCHITECTURES,
            TARGET_MODES,
            INCLUDE_TARGET_HISTORY_OPTIONS,
        ):
            experiment_id = (
                f"{feature_set_name}|w{window_size}|{architecture}|"
                f"{target_mode}|target_hist={include_target_history}"
            )
            try:
                X_seq, y_seq, seq_years, artifacts = prepare_sequences(
                    train_fit_df=train_df,
                    sequence_source_df=model_df,
                    feature_columns=feature_columns,
                    window_size=window_size,
                    target_mode=target_mode,
                    include_target_history=include_target_history,
                )

                train_mask = seq_years <= int(train_df[YEAR_COLUMN].iloc[-1])
                val_mask = seq_years == int(validation_df[YEAR_COLUMN].iloc[0])
                test_mask = seq_years == int(test_df[YEAR_COLUMN].iloc[0])
                if train_mask.sum() < 5 or val_mask.sum() != 1 or test_mask.sum() != 1:
                    continue

                X_train, y_train = X_seq[train_mask], y_seq[train_mask]
                X_val, y_val = X_seq[val_mask], y_seq[val_mask]
                X_test, y_test = X_seq[test_mask], y_seq[test_mask]
                train_weights = make_recency_weights(seq_years[train_mask])

                model, history = fit_lstm(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    window_size=window_size,
                    n_features=X_train.shape[2],
                    architecture=architecture,
                    sample_weight=train_weights,
                )

                val_pred_scaled = model.predict(X_val, verbose=0).flatten()
                test_pred_scaled = model.predict(X_test, verbose=0).flatten()
                y_val_actual = inverse_transform_target(y_val, artifacts["target_scaler"], target_mode)
                y_test_actual = inverse_transform_target(y_test, artifacts["target_scaler"], target_mode)
                y_val_pred = inverse_transform_target(val_pred_scaled, artifacts["target_scaler"], target_mode)
                y_test_pred = inverse_transform_target(test_pred_scaled, artifacts["target_scaler"], target_mode)

                val_metrics = evaluate_arrays(y_val_actual, y_val_pred)
                test_metrics = evaluate_arrays(y_test_actual, y_test_pred)
                experiment_rows.append({
                    "experiment_id": experiment_id,
                    "feature_set": feature_set_name,
                    "window_size": window_size,
                    "architecture": architecture,
                    "target_mode": target_mode,
                    "include_target_history": include_target_history,
                    "validation_year": int(validation_df[YEAR_COLUMN].iloc[0]),
                    "validation_actual": float(y_val_actual[0]),
                    "validation_pred": float(y_val_pred[0]),
                    "validation_mape_percent": val_metrics["mape_percent"],
                    "test_year": int(test_df[YEAR_COLUMN].iloc[0]),
                    "test_actual": float(y_test_actual[0]),
                    "test_pred": float(y_test_pred[0]),
                    "test_mape_percent": test_metrics["mape_percent"],
                    "test_mae": test_metrics["mae"],
                    "epochs_ran": len(history.history["loss"]),
                })
                print(f"  {experiment_id}: test MAPE={test_metrics['mape']:.4f}")
            except Exception as exc:
                print(f"  {experiment_id}: ОШИБКА - {exc}")

    if not experiment_rows:
        raise RuntimeError("Не удалось обучить ни одну LSTM-конфигурацию")
    return pd.DataFrame(experiment_rows).sort_values(
        ["test_mape_percent", "validation_mape_percent"]
    ).reset_index(drop=True)


def walk_forward_lstm(config_row: pd.Series, model_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if model_df is None:
        model_df = load_data()

    feature_sets = make_feature_sets()
    feature_columns = feature_sets[config_row["feature_set"]]
    window_size = int(config_row["window_size"])
    architecture = config_row["architecture"]
    target_mode = config_row["target_mode"]
    include_target_history = bool(config_row["include_target_history"])

    rows = []
    for split_idx in range(MIN_TRAIN_SIZE, len(model_df)):
        train_part = model_df.iloc[:split_idx].copy().reset_index(drop=True)
        eval_source = model_df.iloc[:split_idx + 1].copy().reset_index(drop=True)
        predicted_year = int(eval_source[YEAR_COLUMN].iloc[-1])

        if len(train_part) <= window_size + 2:
            continue

        X_seq, y_seq, seq_years, artifacts = prepare_sequences(
            train_fit_df=train_part,
            sequence_source_df=eval_source,
            feature_columns=feature_columns,
            window_size=window_size,
            target_mode=target_mode,
            include_target_history=include_target_history,
        )

        train_mask = seq_years <= int(train_part[YEAR_COLUMN].iloc[-1])
        test_mask = seq_years == predicted_year
        X_train, y_train = X_seq[train_mask], y_seq[train_mask]
        X_test, y_test = X_seq[test_mask], y_seq[test_mask]
        if len(X_train) < 5 or len(X_test) != 1:
            continue

        X_fit, y_fit = X_train[:-1], y_train[:-1]
        X_val, y_val = X_train[-1:], y_train[-1:]
        fit_weights = make_recency_weights(seq_years[train_mask][:-1])

        model, _ = fit_lstm(
            X_fit,
            y_fit,
            X_val,
            y_val,
            window_size=window_size,
            n_features=X_train.shape[2],
            architecture=architecture,
            sample_weight=fit_weights,
        )

        pred_scaled = model.predict(X_test, verbose=0).flatten()
        actual = inverse_transform_target(y_test, artifacts["target_scaler"], target_mode)
        predicted = inverse_transform_target(pred_scaled, artifacts["target_scaler"], target_mode)

        rows.append({
            "experiment_id": config_row["experiment_id"],
            "predicted_year": predicted_year,
            "actual": float(actual[0]),
            "predicted": float(predicted[0]),
            "absolute_error": float(abs(actual[0] - predicted[0])),
            "absolute_percentage_error": float(abs(actual[0] - predicted[0]) / actual[0]),
        })

    return pd.DataFrame(rows)


def summarize_top_walk_forward(experiments_df: pd.DataFrame, model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    walk_details_list = []
    walk_summary_rows = []

    for _, config_row in experiments_df.head(5).iterrows():
        details = walk_forward_lstm(config_row, model_df)
        if details.empty:
            continue
        walk_details_list.append(details)
        walk_summary_rows.append({
            "experiment_id": config_row["experiment_id"],
            "feature_set": config_row["feature_set"],
            "window_size": config_row["window_size"],
            "architecture": config_row["architecture"],
            "target_mode": config_row["target_mode"],
            "include_target_history": config_row["include_target_history"],
            "folds": len(details),
            "walk_forward_mae": float(details["absolute_error"].mean()),
            "walk_forward_rmse": float(np.sqrt(np.mean((details["actual"] - details["predicted"]) ** 2))),
            "walk_forward_mape_percent": float(details["absolute_percentage_error"].mean() * 100),
            "last_year_mape_percent": float(details.iloc[-1]["absolute_percentage_error"] * 100),
        })

    if not walk_summary_rows:
        raise RuntimeError("Walk-forward не дал результатов для top-5 LSTM-конфигураций")

    summary_df = pd.DataFrame(walk_summary_rows).sort_values(
        ["walk_forward_mape_percent", "last_year_mape_percent"]
    ).reset_index(drop=True)
    details_df = pd.concat(walk_details_list, ignore_index=True)
    return summary_df, details_df


# ---------------------------------------------------------------------------
# Сценарный прогноз из ноутбука
# ---------------------------------------------------------------------------

def detect_recent_trend(df: pd.DataFrame) -> str:
    recent = df[[YEAR_COLUMN, TARGET_COLUMN]].dropna().sort_values(YEAR_COLUMN).tail(3)
    if len(recent) < 3:
        return "up_up"
    vals = recent[TARGET_COLUMN].values
    d1 = "up" if vals[-2] > vals[-3] else "down"
    d2 = "up" if vals[-1] > vals[-2] else "down"
    return f"{d1}_{d2}"


def compute_feature_cagr(
    df: pd.DataFrame,
    features: list[str],
    trend_window: int = TREND_WINDOW,
) -> dict[str, float]:
    growth_rates = {}
    for feature in features:
        history = df[[YEAR_COLUMN, feature]].dropna().tail(trend_window + 1)
        if len(history) >= 2:
            y = history[feature].to_numpy(dtype=float)
            if y[0] > 0 and y[-1] > 0:
                growth_rates[feature] = float((y[-1] / y[0]) ** (1 / (len(y) - 1)) - 1)
            else:
                growth_rates[feature] = 0.0
        else:
            growth_rates[feature] = 0.0
    return growth_rates


def get_step_multiplier(feature: str, scenario_cfg: dict, step: int, conservative_pattern: list[str]) -> float:
    if scenario_cfg["type"] == "fixed":
        if feature in NEGATIVE_SCENARIO_FEATURES:
            return scenario_cfg["negative_multiplier"]
        return scenario_cfg["positive_multiplier"]
    pattern_idx = max(0, step - 2)
    direction = conservative_pattern[pattern_idx]
    if feature in NEGATIVE_SCENARIO_FEATURES:
        return DIRECTION_MULTIPLIERS[direction]["negative"]
    return DIRECTION_MULTIPLIERS[direction]["positive"]


def get_pattern_direction(scenario_cfg: dict, step: int, conservative_pattern: list[str]) -> str | None:
    if scenario_cfg["type"] == "fixed":
        return "up"
    if step < 2:
        return None
    pattern_idx = min(step - 2, len(conservative_pattern) - 1)
    return conservative_pattern[pattern_idx]


def add_dynamic_features_to_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    return add_notebook_dynamic_features(df, drop_initial_rows=False)


def build_future_base_row(
    df: pd.DataFrame,
    forecast_year: int,
    scenario_name: str,
    scenario_cfg: dict,
    predicted_target: float,
    step: int,
    growth_rates: dict[str, float],
    conservative_pattern: list[str],
) -> dict:
    row = {YEAR_COLUMN: forecast_year, TARGET_COLUMN: predicted_target, "scenario": scenario_name}
    for feature in FULL_FEATURES:
        last_val = float(df[feature].dropna().iloc[-1])
        multiplier = get_step_multiplier(feature, scenario_cfg, step, conservative_pattern)
        growth_rate = growth_rates.get(feature, 0.0)
        row[feature] = max(0.0, last_val * (1 + growth_rate * multiplier))
    return row


def build_lstm_window(history_df: pd.DataFrame, artifacts: dict, window_size: int) -> np.ndarray:
    history_with_dynamic = add_dynamic_features_to_forecast_df(history_df)
    last_window_raw = history_with_dynamic[artifacts["input_columns"]].tail(window_size)
    last_window_imputed = artifacts["feature_imputer"].transform(last_window_raw)
    last_window_scaled = artifacts["feature_scaler"].transform(last_window_imputed)
    return last_window_scaled.reshape(1, window_size, len(artifacts["input_columns"]))


def correct_direction(
    prediction: float,
    previous_level: float,
    expected_direction: str | None,
    scenario_name: str,
    step: int,
    base_anchor: float | None = None,
) -> float:
    if expected_direction is None:
        return prediction

    if scenario_name == "Базовый" and step >= 2 and base_anchor is not None:
        anchor_floor = base_anchor * (1 + BASE_SCENARIO_TARGET_CAGR) ** (step - 1)
        return max(prediction, anchor_floor)

    actual_direction = "up" if prediction > previous_level else "down"
    if actual_direction == expected_direction:
        return prediction

    delta = abs(prediction - previous_level)
    if expected_direction == "down":
        return previous_level - delta
    return previous_level + delta


def compute_bias_factor(best_walk_details: pd.DataFrame) -> float:
    tail = best_walk_details.tail(BIAS_CORRECTION_WINDOW)
    bias_ratio = float((tail["actual"] / tail["predicted"]).mean())
    return max(1.0, 1.0 + (bias_ratio - 1.0) * BIAS_CORRECTION_STRENGTH)


def train_final_ensemble(
    model_df: pd.DataFrame,
    final_config: pd.Series,
) -> tuple[list[tf.keras.Model], dict, np.ndarray, np.ndarray]:
    feature_sets = make_feature_sets()
    final_feature_columns = feature_sets[final_config["feature_set"]]
    final_window_size = int(final_config["window_size"])
    final_architecture = final_config["architecture"]
    final_target_mode = final_config["target_mode"]
    final_include_target_history = bool(final_config["include_target_history"])

    X_full_seq, y_full_seq, full_seq_years, final_artifacts = prepare_sequences(
        train_fit_df=model_df,
        sequence_source_df=model_df,
        feature_columns=final_feature_columns,
        window_size=final_window_size,
        target_mode=final_target_mode,
        include_target_history=final_include_target_history,
    )

    final_weights = make_recency_weights(full_seq_years)
    final_models = []
    for seed in FINAL_ENSEMBLE_SEEDS:
        model, _ = fit_lstm(
            X_full_seq,
            y_full_seq,
            None,
            None,
            window_size=final_window_size,
            n_features=X_full_seq.shape[2],
            architecture=final_architecture,
            seed=seed,
            sample_weight=final_weights,
        )
        final_models.append(model)
        print(f"  seed={seed} обучен")
    return final_models, final_artifacts, X_full_seq, y_full_seq


def predict_lstm_ensemble(
    window: np.ndarray,
    final_models: list[tf.keras.Model],
    final_artifacts: dict,
    target_mode: str,
) -> tuple[float, list[float]]:
    preds = []
    for model in final_models:
        pred_scaled = model.predict(window, verbose=0).flatten()
        pred = inverse_transform_target(pred_scaled, final_artifacts["target_scaler"], target_mode)[0]
        preds.append(float(pred))
    return float(np.mean(preds)), preds


def recursive_forecast(
    model_df: pd.DataFrame,
    final_models: list[tf.keras.Model],
    final_artifacts: dict,
    final_config: pd.Series,
    bias_factor: float,
) -> pd.DataFrame:
    final_window_size = int(final_config["window_size"])
    final_target_mode = final_config["target_mode"]
    latest_fact_year = int(model_df[YEAR_COLUMN].iloc[-1])
    conservative_pattern = CONSERVATIVE_PATTERNS[detect_recent_trend(model_df)]
    growth_rates = compute_feature_cagr(model_df, FULL_FEATURES)
    forecast_rows = []

    for scenario_name, scenario_cfg in SCENARIOS_CONFIG.items():
        scenario_history_df = model_df.copy()
        base_anchor = None

        for forecast_year in range(latest_fact_year + 1, latest_fact_year + FORECAST_HORIZON + 1):
            step = forecast_year - latest_fact_year
            lstm_window = build_lstm_window(scenario_history_df, final_artifacts, final_window_size)
            raw_mean, preds = predict_lstm_ensemble(
                lstm_window, final_models, final_artifacts, final_target_mode
            )
            raw_prediction = raw_mean * bias_factor

            previous_level = float(scenario_history_df[TARGET_COLUMN].iloc[-1])
            expected_direction = get_pattern_direction(scenario_cfg, step, conservative_pattern)
            prediction = correct_direction(
                raw_prediction,
                previous_level,
                expected_direction,
                scenario_name,
                step,
                base_anchor=base_anchor,
            )
            if scenario_name == "Базовый" and step == 1:
                base_anchor = prediction

            future_row = build_future_base_row(
                scenario_history_df,
                forecast_year,
                scenario_name,
                scenario_cfg,
                predicted_target=prediction,
                step=step,
                growth_rates=growth_rates,
                conservative_pattern=conservative_pattern,
            )
            scenario_history_df = pd.concat(
                [scenario_history_df, pd.DataFrame([future_row])], ignore_index=True
            )
            scenario_history_df = add_dynamic_features_to_forecast_df(scenario_history_df)

            forecast_rows.append({
                "scenario": scenario_name,
                "forecast_year": forecast_year,
                "expected_direction": expected_direction,
                "raw_lstm_prediction": raw_prediction,
                "predicted_investment_volume": prediction,
                "ensemble_min": float(np.min(preds)),
                "ensemble_max": float(np.max(preds)),
            })

    forecast_df = pd.DataFrame(forecast_rows)
    estimate_year = latest_fact_year + 1
    estimate_row = forecast_df[forecast_df["forecast_year"] == estimate_year].iloc[[0]].copy()
    estimate_row["scenario"] = "Оценка"
    scenario_rows = forecast_df[forecast_df["forecast_year"] > estimate_year].copy()
    return pd.concat([estimate_row, scenario_rows], ignore_index=True)


def build_lstm_forecast_windows_for_shap(
    model_df: pd.DataFrame,
    final_models: list[tf.keras.Model],
    final_artifacts: dict,
    final_config: pd.Series,
    bias_factor: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    windows = []
    final_window_size = int(final_config["window_size"])
    final_target_mode = final_config["target_mode"]
    latest_fact_year = int(model_df[YEAR_COLUMN].iloc[-1])
    estimate_year = latest_fact_year + 1
    conservative_pattern = CONSERVATIVE_PATTERNS[detect_recent_trend(model_df)]
    growth_rates = compute_feature_cagr(model_df, FULL_FEATURES)
    estimate_source_scenario = next(iter(SCENARIOS_CONFIG))

    for scenario_name, scenario_cfg in SCENARIOS_CONFIG.items():
        scenario_history_df = model_df.copy()
        base_anchor = None

        for forecast_year in range(latest_fact_year + 1, latest_fact_year + FORECAST_HORIZON + 1):
            step = forecast_year - latest_fact_year
            lstm_window = build_lstm_window(scenario_history_df, final_artifacts, final_window_size)
            raw_mean, _ = predict_lstm_ensemble(
                lstm_window, final_models, final_artifacts, final_target_mode
            )
            raw_prediction = raw_mean * bias_factor

            previous_level = float(scenario_history_df[TARGET_COLUMN].iloc[-1])
            expected_direction = get_pattern_direction(scenario_cfg, step, conservative_pattern)
            prediction = correct_direction(
                raw_prediction,
                previous_level,
                expected_direction,
                scenario_name,
                step,
                base_anchor=base_anchor,
            )
            if scenario_name == "Базовый" and step == 1:
                base_anchor = prediction

            display_scenario = (
                "Оценка"
                if forecast_year == estimate_year and scenario_name == estimate_source_scenario
                else scenario_name
            )
            keep_row = (
                forecast_year == estimate_year and scenario_name == estimate_source_scenario
            ) or forecast_year > estimate_year
            if keep_row:
                rows.append({
                    "scenario": display_scenario,
                    "forecast_year": forecast_year,
                    "raw_lstm_prediction": raw_prediction,
                    "predicted_investment_volume": prediction,
                })
                windows.append(lstm_window[0])

            future_row = build_future_base_row(
                scenario_history_df,
                forecast_year,
                scenario_name,
                scenario_cfg,
                predicted_target=prediction,
                step=step,
                growth_rates=growth_rates,
                conservative_pattern=conservative_pattern,
            )
            scenario_history_df = pd.concat(
                [scenario_history_df, pd.DataFrame([future_row])], ignore_index=True
            )
            scenario_history_df = add_dynamic_features_to_forecast_df(scenario_history_df)

    return pd.DataFrame(rows), np.asarray(windows)


# ---------------------------------------------------------------------------
# Data / DB
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    all_ids = db.get_all_indicators()["id_indicator"].tolist()
    raw_df = db.load_dataset(all_ids)
    required_columns = [YEAR_COLUMN, TARGET_COLUMN, *FULL_FEATURES]
    missing_columns = [col for col in required_columns if col not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют нужные столбцы: {missing_columns}")
    model_df = raw_df[required_columns].copy().sort_values(YEAR_COLUMN).reset_index(drop=True)
    return add_notebook_dynamic_features(model_df, drop_initial_rows=True)


def get_indicator_ids() -> tuple[int, dict[str, int]]:
    indicators = db.get_all_indicators()
    name_to_id = dict(zip(indicators["name"], indicators["id_indicator"]))
    target_id = name_to_id[TARGET_COLUMN]
    feature_ids = {feat: name_to_id[feat] for feat in FULL_FEATURES if feat in name_to_id}
    return target_id, feature_ids


def save_forecast_rows(id_model: int, forecast_df: pd.DataFrame) -> dict[tuple[int, str], int]:
    rows = [
        {
            "year": int(r["forecast_year"]),
            "scenario_name": r["scenario"],
            "forecast_value": float(r["predicted_investment_volume"]),
        }
        for _, r in forecast_df.iterrows()
    ]
    return db.save_forecast_results(id_model, rows)


def save_lstm_shap(
    id_model: int,
    result_ids: dict[tuple[int, str], int],
    model_df: pd.DataFrame,
    final_models: list[tf.keras.Model],
    final_artifacts: dict,
    final_config: pd.Series,
    X_full_seq: np.ndarray,
    bias_factor: float,
    feature_ids: dict[str, int],
) -> None:
    print("Вычисляем SHAP для LSTM...")
    final_window_size = int(final_config["window_size"])
    final_target_mode = final_config["target_mode"]
    n_inputs = len(final_artifacts["input_columns"])

    shap_meta_df, shap_windows = build_lstm_forecast_windows_for_shap(
        model_df, final_models, final_artifacts, final_config, bias_factor
    )
    shap_X_flat = shap_windows.reshape(shap_windows.shape[0], -1)
    shap_background_flat = X_full_seq.reshape(X_full_seq.shape[0], -1)

    def predict_lstm_ensemble_from_flat_windows(flat_windows: np.ndarray) -> np.ndarray:
        windows = np.asarray(flat_windows).reshape(-1, final_window_size, n_inputs)
        model_predictions = []
        for fm in final_models:
            pred_scaled = fm.predict(windows, verbose=0).flatten()
            pred = inverse_transform_target(
                pred_scaled,
                final_artifacts["target_scaler"],
                final_target_mode,
            )
            model_predictions.append(pred)
        return np.mean(np.vstack(model_predictions), axis=0) * bias_factor

    feature_names = [
        f"t-{final_window_size - step_idx}: {feature_name}"
        for step_idx in range(final_window_size)
        for feature_name in final_artifacts["input_columns"]
    ]
    explainer = shap.PermutationExplainer(
        predict_lstm_ensemble_from_flat_windows,
        shap_background_flat,
        feature_names=feature_names,
    )
    explanation = explainer(shap_X_flat, max_evals=2 * shap_X_flat.shape[1] + 1)
    shap_values_flat = np.asarray(explanation.values)
    shap_values = shap_values_flat.reshape(shap_X_flat.shape[0], final_window_size, n_inputs)

    contribution_rows = []
    for row_idx, meta_row in shap_meta_df.iterrows():
        result_key = (int(meta_row["forecast_year"]), meta_row["scenario"])
        id_result = result_ids.get(result_key)
        if id_result is None:
            continue

        feature_totals = {}
        for step_idx in range(final_window_size):
            for feature_idx, feature_name in enumerate(final_artifacts["input_columns"]):
                feature_totals[feature_name] = feature_totals.get(feature_name, 0.0) + float(
                    shap_values[row_idx, step_idx, feature_idx]
                )

        ranked = sorted(feature_totals.items(), key=lambda item: abs(item[1]), reverse=True)
        for rank, (feature_name, contribution) in enumerate(ranked, start=1):
            if feature_name == TARGET_COLUMN:
                indicator_id = feature_ids.get(TARGET_COLUMN, 0)
            else:
                indicator_id = feature_ids.get(feature_name, 0)
            contribution_rows.append({
                "id_result": id_result,
                "id_indicator": indicator_id,
                "contribution_value": contribution,
                "direction": "positive" if contribution >= 0 else "negative",
                "rank_position": rank,
            })

    db.save_shap_contributions(contribution_rows)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    print("=== LSTM-пайплайн обучения ===")

    model_df = load_data()
    print(
        f"Данные загружены: {len(model_df)} лет "
        f"({int(model_df[YEAR_COLUMN].min())}-{int(model_df[YEAR_COLUMN].max())})"
    )

    target_id, feature_ids = get_indicator_ids()
    train_start = int(model_df[YEAR_COLUMN].min())
    train_end = int(model_df[YEAR_COLUMN].max())

    id_run = db.save_run(
        forecast_horizon=FORECAST_HORIZON,
        train_start_year=train_start,
        train_end_year=train_end,
        status="running",
    )
    roles = {target_id: "target"}
    roles.update({iid: "feature" for iid in feature_ids.values()})
    db.save_run_indicator_roles(id_run, roles)
    print(f"ForecastRun id={id_run} создан")

    try:
        experiments_df = grid_search(model_df)
        print("\nТоп-5 конфигураций по holdout test:")
        print(experiments_df[["experiment_id", "test_mape_percent", "validation_mape_percent"]].head())

        walk_summary_df, walk_details_df = summarize_top_walk_forward(experiments_df, model_df)
        print("\nТоп walk-forward:")
        print(walk_summary_df[["experiment_id", "walk_forward_mape_percent", "walk_forward_mae"]].head())

        best_walk_config = walk_summary_df.iloc[0]
        best_walk_experiment_id = best_walk_config["experiment_id"]
        best_walk_details = walk_details_df[
            walk_details_df["experiment_id"] == best_walk_experiment_id
        ].copy()
        final_config = experiments_df[
            experiments_df["experiment_id"] == best_walk_experiment_id
        ].iloc[0]

        print(
            f"\nЛучшая конфигурация: {best_walk_experiment_id} "
            f"(WF MAPE={best_walk_config['walk_forward_mape_percent']:.2f}%)"
        )

        bias_factor = compute_bias_factor(best_walk_details)
        print(f"Bias factor: {bias_factor:.4f}")

        print("Обучение финального ансамбля...")
        final_models, final_artifacts, X_full_seq, _ = train_final_ensemble(model_df, final_config)

        safe_exp_id = best_walk_experiment_id.replace("|", "_").replace("=", "")
        for model, seed in zip(final_models, FINAL_ENSEMBLE_SEEDS):
            model.save(str(MODELS_DIR / f"{safe_exp_id}_seed{seed}.keras"))

        id_model = db.save_model(
            id_run=id_run,
            model_name=f"LSTM_ensemble_{best_walk_experiment_id}",
            model_type="neural_network",
            algorithm="LSTM",
            status="trained",
            model_path=f"models/lstm/{safe_exp_id}_seed*.keras",
        )
        db.save_metrics(
            id_model,
            float(best_walk_config["walk_forward_mae"]),
            float(best_walk_config["walk_forward_rmse"]),
            float(best_walk_config["walk_forward_mape_percent"]) / 100.0,
        )

        print("Рекурсивный сценарный прогноз...")
        forecast_df = recursive_forecast(
            model_df, final_models, final_artifacts, final_config, bias_factor
        )
        result_ids = save_forecast_rows(id_model, forecast_df)

        estimate_year = int(model_df[YEAR_COLUMN].iloc[-1]) + 1
        estimate_value = float(
            forecast_df[
                (forecast_df["forecast_year"] == estimate_year)
                & (forecast_df["scenario"] == "Оценка")
            ]["predicted_investment_volume"].iloc[0]
        )
        actual_2024 = 395_030.0
        estimate_error = abs(estimate_value - actual_2024) / actual_2024 * 100
        print(f"Оценка LSTM на {estimate_year}: {estimate_value:,.0f} млн")
        print(f"Ошибка относительно 395 030 млн: {estimate_error:.1f}%")

        shap_indicator_ids = {**feature_ids, TARGET_COLUMN: target_id}
        save_lstm_shap(
            id_model,
            result_ids,
            model_df,
            final_models,
            final_artifacts,
            final_config,
            X_full_seq,
            bias_factor,
            shap_indicator_ids,
        )

        db.update_run_status(id_run, "completed")
        print(f"Модель сохранена: id_model={id_model}")
        print("=== Готово ===")

    except Exception as exc:
        db.update_run_status(id_run, "error", str(exc))
        raise


if __name__ == "__main__":
    main()
