"""
Обучение LSTM-ансамбля.

Запуск:
    python scripts/train_lstm.py

Что делает:
    1. Читает данные из indicators.db
    2. Перебирает 24 конфигурации (наборы фичей × окна × архитектуры × режим таргета)
    3. Финальный ансамбль из 5 LSTM с лучшей конфигурацией
    4. Сохраняет веса (.keras) в models/lstm/
    5. Пишет ForecastRun, Model, ModelMetric, ForecastResult, ShapContribution в models.db
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

import db
from forecast_utils import (
    SELECTED_FEATURES,
    TARGET_COLUMN,
    YEAR_COLUMN,
    FORECAST_HORIZON,
    NEGATIVE_SCENARIO_FEATURES,
    SCENARIOS_CONFIG,
    build_all_scenario_frames,
    detect_conservative_pattern,
    compute_annual_slopes,
    evaluate_predictions,
)

MODELS_DIR = ROOT / "models" / "lstm"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Гиперпараметры
# ---------------------------------------------------------------------------

EPOCHS        = 300
BATCH_SIZE    = 8
LEARNING_RATE = 0.001
L2_REG        = 1e-4
MIN_TRAIN_SIZE = 12
FINAL_ENSEMBLE_SEEDS = [42, 7, 13, 99, 2024]

FEATURE_SETS = {
    "compact": SELECTED_FEATURES,
    "dynamic": None,  # заполняется после добавления динамических признаков
}

WINDOW_SIZES   = [4, 8]
ARCHITECTURES  = ["small", "medium"]
TARGET_MODES   = ["raw", "log"]


# ---------------------------------------------------------------------------
# Динамические признаки
# ---------------------------------------------------------------------------

DYNAMIC_BASE_FEATURES = [
    "Валовый региональный продукт",
    "Валовой региональный продукт на душу населения",
    "Среднедушевые денежные доходы населения в год",
    "Стоимость основных фондов по Тюм обл, на конец года, по полной учетной стоимости",
]


def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DYNAMIC_BASE_FEATURES:
        if col not in df.columns:
            continue
        df[f"{col}_diff"]         = df[col].diff()
        df[f"{col}_pct"]          = df[col].pct_change()
        df[f"{col}_roll3"]        = df[col].rolling(3).mean()
        df[f"{col}_momentum"]     = df[col] - df[col].shift(2)
    return df


def get_dynamic_feature_columns(df: pd.DataFrame) -> list[str]:
    base = list(SELECTED_FEATURES)
    extras = [c for c in df.columns if any(
        c.startswith(b + "_") for b in DYNAMIC_BASE_FEATURES
    )]
    return base + extras


# ---------------------------------------------------------------------------
# Target transform / inverse
# ---------------------------------------------------------------------------

def transform_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log":
        return np.log1p(y)
    return y


def inverse_transform_target(y: np.ndarray, scaler: StandardScaler, mode: str) -> np.ndarray:
    y_unscaled = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    if mode == "log":
        return np.expm1(y_unscaled)
    return y_unscaled


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
    target_imputer  = SimpleImputer(strategy="median")
    feature_scaler  = StandardScaler()
    target_scaler   = StandardScaler()

    feature_imputer.fit(train_fit_df[input_columns])
    target_imputer.fit(train_fit_df[[TARGET_COLUMN]])

    X_train_raw   = feature_imputer.transform(train_fit_df[input_columns])
    X_source_raw  = feature_imputer.transform(sequence_source_df[input_columns])
    y_train_raw   = target_imputer.transform(train_fit_df[[TARGET_COLUMN]])
    y_source_raw  = target_imputer.transform(sequence_source_df[[TARGET_COLUMN]])

    y_train_t  = transform_target(y_train_raw, target_mode)
    y_source_t = transform_target(y_source_raw, target_mode)

    feature_scaler.fit(X_train_raw)
    target_scaler.fit(y_train_t)

    X_scaled = feature_scaler.transform(X_source_raw)
    y_scaled = target_scaler.transform(y_source_t).flatten()
    years    = sequence_source_df[YEAR_COLUMN].to_numpy().astype(int)

    X_seq, y_seq, seq_years = [], [], []
    for i in range(window_size, len(sequence_source_df)):
        X_seq.append(X_scaled[i - window_size : i])
        y_seq.append(y_scaled[i])
        seq_years.append(years[i])

    artifacts = {
        "feature_imputer": feature_imputer,
        "target_imputer":  target_imputer,
        "feature_scaler":  feature_scaler,
        "target_scaler":   target_scaler,
        "input_columns":   input_columns,
        "target_mode":     target_mode,
    }
    return np.array(X_seq), np.array(y_seq), np.array(seq_years), artifacts


# ---------------------------------------------------------------------------
# Recency weights
# ---------------------------------------------------------------------------

def make_recency_weights(years: np.ndarray, decay: float = 0.85) -> np.ndarray:
    max_year = years.max()
    weights = decay ** (max_year - years).astype(float)
    return weights / weights.sum() * len(weights)


# ---------------------------------------------------------------------------
# LSTM модели
# ---------------------------------------------------------------------------

def build_lstm_model(window_size: int, n_features: int, architecture: str) -> tf.keras.Model:
    if architecture == "small":
        layers = [
            Input(shape=(window_size, n_features)),
            LSTM(16, kernel_regularizer=l2(L2_REG)),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    elif architecture == "medium":
        layers = [
            Input(shape=(window_size, n_features)),
            LSTM(32, kernel_regularizer=l2(L2_REG)),
            Dropout(0.1),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    elif architecture == "stacked":
        layers = [
            Input(shape=(window_size, n_features)),
            LSTM(32, return_sequences=True, kernel_regularizer=l2(L2_REG)),
            Dropout(0.1),
            LSTM(16, kernel_regularizer=l2(L2_REG)),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    else:
        raise ValueError(f"Неизвестная архитектура: {architecture!r}")

    model = Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_lstm(
    X_train, y_train,
    X_val, y_val,
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
        ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=12, min_lr=5e-5),
    ]

    history = model.fit(
        X_train, y_train,
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
# Перебор конфигураций
# ---------------------------------------------------------------------------

def grid_search(df: pd.DataFrame, df_dynamic: pd.DataFrame) -> pd.DataFrame:
    print("Перебор конфигураций LSTM...")
    results = []

    compact_features = SELECTED_FEATURES
    dynamic_features = get_dynamic_feature_columns(df_dynamic)

    feature_set_map = {"compact": compact_features, "dynamic": dynamic_features}

    train_df = df.iloc[:-2]
    val_df   = df.iloc[-2:]

    train_dyn = df_dynamic.iloc[:-2]
    val_dyn   = df_dynamic.iloc[-2:]

    for feat_set_name, feat_cols in feature_set_map.items():
        source_train = train_dyn if feat_set_name == "dynamic" else train_df
        source_val   = pd.concat([train_dyn if feat_set_name == "dynamic" else train_df,
                                   val_dyn if feat_set_name == "dynamic" else val_df])

        for window_size in WINDOW_SIZES:
            for architecture in ARCHITECTURES:
                for target_mode in TARGET_MODES:
                    exp_id = f"{feat_set_name}_w{window_size}_{architecture}_{target_mode}"
                    try:
                        X_tr, y_tr, yr_tr, arts = prepare_sequences(
                            source_train, source_train, feat_cols, window_size, target_mode
                        )
                        X_all, y_all, yr_all, _ = prepare_sequences(
                            source_train, source_val, feat_cols, window_size, target_mode
                        )
                        # val — последние 2 точки
                        n_val = 2
                        X_val_seq = X_all[-n_val:]
                        y_val_seq = y_all[-n_val:]

                        weights = make_recency_weights(yr_tr)
                        model, _ = fit_lstm(
                            X_tr, y_tr, X_val_seq, y_val_seq,
                            window_size, X_tr.shape[2], architecture,
                            sample_weight=weights,
                        )

                        preds_scaled = model.predict(X_val_seq, verbose=0).flatten()
                        preds = inverse_transform_target(preds_scaled, arts["target_scaler"], target_mode)
                        actuals = val_df[TARGET_COLUMN].to_numpy()[-n_val:]

                        metrics = evaluate_predictions(actuals, preds)
                        results.append({
                            "experiment_id": exp_id,
                            "feat_set":      feat_set_name,
                            "window_size":   window_size,
                            "architecture":  architecture,
                            "target_mode":   target_mode,
                            **metrics,
                        })
                        print(f"  {exp_id}: MAPE={metrics['mape']:.4f}")

                    except Exception as e:
                        print(f"  {exp_id}: ОШИБКА — {e}")

    return pd.DataFrame(results).sort_values("mape").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward для лучшей конфигурации
# ---------------------------------------------------------------------------

def walk_forward_lstm(
    df: pd.DataFrame,
    df_dynamic: pd.DataFrame,
    feat_cols: list[str],
    window_size: int,
    architecture: str,
    target_mode: str,
) -> dict:
    source_df = df_dynamic if feat_cols != SELECTED_FEATURES else df
    valid_data = source_df.dropna(subset=feat_cols + [TARGET_COLUMN]).reset_index(drop=True)

    actuals, predictions = [], []

    for split_idx in range(MIN_TRAIN_SIZE, len(valid_data)):
        train_part = valid_data.iloc[:split_idx]
        test_part  = valid_data.iloc[split_idx : split_idx + 1]

        full_part = valid_data.iloc[: split_idx + 1]

        X_seq, y_seq, yr_seq, arts = prepare_sequences(
            train_part, full_part, feat_cols, window_size, target_mode
        )
        if len(X_seq) == 0:
            continue

        weights = make_recency_weights(yr_seq[:-1])
        model, _ = fit_lstm(
            X_seq[:-1], y_seq[:-1], None, None,
            window_size, X_seq.shape[2], architecture,
            sample_weight=weights,
        )

        pred_scaled = model.predict(X_seq[-1:], verbose=0).flatten()
        pred = inverse_transform_target(pred_scaled, arts["target_scaler"], target_mode)[0]

        actual = float(test_part[TARGET_COLUMN].iloc[0])
        actuals.append(actual)
        predictions.append(float(pred))

    return evaluate_predictions(np.array(actuals), np.array(predictions))


# ---------------------------------------------------------------------------
# Построение окна для рекурсивного прогноза
# ---------------------------------------------------------------------------

def build_lstm_window(history_df: pd.DataFrame, artifacts: dict, window_size: int) -> np.ndarray:
    last_window = history_df[artifacts["input_columns"]].tail(window_size)
    imputed = artifacts["feature_imputer"].transform(last_window)
    scaled  = artifacts["feature_scaler"].transform(imputed)
    return scaled.reshape(1, window_size, len(artifacts["input_columns"]))


# ---------------------------------------------------------------------------
# Рекурсивный сценарный прогноз
# ---------------------------------------------------------------------------

def recursive_forecast(
    df: pd.DataFrame,
    final_models: list[tf.keras.Model],
    final_artifacts: dict,
    final_window_size: int,
    final_target_mode: str,
    bias_factor: float,
) -> pd.DataFrame:
    conservative_pattern = detect_conservative_pattern(df)
    scenario_frames = build_all_scenario_frames(df, SELECTED_FEATURES, FORECAST_HORIZON)

    forecast_rows = []

    for scenario_name, future_df in scenario_frames.items():
        history_df = df.copy()
        base_anchor = None

        for step_idx, row in future_df.iterrows():
            forecast_year = int(row[YEAR_COLUMN])
            step = forecast_year - int(df[YEAR_COLUMN].max())

            window = build_lstm_window(history_df, final_artifacts, final_window_size)

            preds = []
            for fm in final_models:
                pred_scaled = fm.predict(window, verbose=0).flatten()
                p = inverse_transform_target(pred_scaled, final_artifacts["target_scaler"], final_target_mode)[0]
                preds.append(float(p))

            raw_pred = float(np.mean(preds)) * bias_factor

            # Добавляем строку с прогнозом в историю
            new_row = row.to_dict()
            new_row[TARGET_COLUMN] = raw_pred
            history_df = pd.concat(
                [history_df, pd.DataFrame([new_row])], ignore_index=True
            )

            forecast_rows.append({
                "scenario":       scenario_name,
                "year":           forecast_year,
                "forecast_value": raw_pred,
                "ensemble_min":   float(np.min(preds)),
                "ensemble_max":   float(np.max(preds)),
            })

            if scenario_name == "Базовый" and step == 1:
                base_anchor = raw_pred

    return pd.DataFrame(forecast_rows)


# ---------------------------------------------------------------------------
# Bias factor по 2023
# ---------------------------------------------------------------------------

def compute_bias_factor(
    df: pd.DataFrame,
    final_models: list[tf.keras.Model],
    final_artifacts: dict,
    final_window_size: int,
    final_target_mode: str,
) -> float:
    train_minus_one = df.iloc[:-1]
    window = build_lstm_window(train_minus_one, final_artifacts, final_window_size)
    preds = []
    for fm in final_models:
        p_scaled = fm.predict(window, verbose=0).flatten()
        p = inverse_transform_target(p_scaled, final_artifacts["target_scaler"], final_target_mode)[0]
        preds.append(float(p))
    raw = float(np.mean(preds))
    actual = float(df[TARGET_COLUMN].iloc[-1])
    return actual / raw if raw != 0 else 1.0


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    print("=== LSTM-пайплайн обучения ===")

    all_ids = db.get_all_indicators()["id_indicator"].tolist()
    df = db.load_dataset(all_ids)
    df_dynamic = add_dynamic_features(df)

    print(f"Данные загружены: {len(df)} лет ({int(df[YEAR_COLUMN].min())}–{int(df[YEAR_COLUMN].max())})")

    indicators = db.get_all_indicators()
    name_to_id = dict(zip(indicators["name"], indicators["id_indicator"]))
    target_id  = name_to_id[TARGET_COLUMN]
    feature_ids = {feat: name_to_id[feat] for feat in SELECTED_FEATURES if feat in name_to_id}

    train_start = int(df[YEAR_COLUMN].min())
    train_end   = int(df[YEAR_COLUMN].max())

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
        # --- Перебор конфигураций ---
        grid_results = grid_search(df, df_dynamic)
        print("\nТоп-5 конфигураций:")
        print(grid_results[["experiment_id", "mape", "mae"]].head())

        best = grid_results.iloc[0]
        feat_set_name  = best["feat_set"]
        final_window   = int(best["window_size"])
        final_arch     = best["architecture"]
        final_mode     = best["target_mode"]
        best_exp_id    = best["experiment_id"]

        feat_cols = (
            get_dynamic_feature_columns(df_dynamic)
            if feat_set_name == "dynamic"
            else SELECTED_FEATURES
        )
        source_df = df_dynamic if feat_set_name == "dynamic" else df

        print(f"\nЛучшая конфигурация: {best_exp_id} (MAPE={best['mape']:.4f})")

        # --- Walk-forward для лучшей конфигурации ---
        print("Walk-forward валидация лучшей конфигурации...")
        wf_metrics = walk_forward_lstm(df, df_dynamic, feat_cols, final_window, final_arch, final_mode)
        print(f"Walk-forward MAPE={wf_metrics['mape']:.4f}, MAE={wf_metrics['mae']:.0f}")

        # --- Финальный ансамбль ---
        print("Обучение финального ансамбля...")
        X_full, y_full, yr_full, final_artifacts = prepare_sequences(
            source_df, source_df, feat_cols, final_window, final_mode
        )
        weights_full = make_recency_weights(yr_full)

        final_models = []
        for seed in FINAL_ENSEMBLE_SEEDS:
            m, _ = fit_lstm(
                X_full, y_full, None, None,
                final_window, X_full.shape[2], final_arch,
                seed=seed, sample_weight=weights_full,
            )
            final_models.append(m)
            print(f"  seed={seed} обучен")

        # --- Сохранение весов ---
        for i, (m, seed) in enumerate(zip(final_models, FINAL_ENSEMBLE_SEEDS)):
            path = MODELS_DIR / f"{best_exp_id}_seed{seed}.keras"
            m.save(str(path))

        # --- Bias correction ---
        bias_factor = compute_bias_factor(source_df, final_models, final_artifacts, final_window, final_mode)
        print(f"Bias factor: {bias_factor:.4f}")

        # --- Запись в БД ---
        model_path_str = f"models/lstm/{best_exp_id}_seed*.keras"
        id_model = db.save_model(
            id_run=id_run,
            model_name=f"LSTM_ensemble_{best_exp_id}",
            model_type="neural_network",
            algorithm="LSTM",
            status="trained",
            model_path=model_path_str,
        )
        db.save_metrics(id_model, wf_metrics["mae"], wf_metrics["rmse"], wf_metrics["mape"])

        # --- Рекурсивный прогноз ---
        print("Рекурсивный сценарный прогноз...")
        forecast_df = recursive_forecast(
            source_df, final_models, final_artifacts,
            final_window, final_mode, bias_factor,
        )

        forecast_rows_db = [
            {"year": int(r["year"]), "scenario_name": r["scenario"], "forecast_value": r["forecast_value"]}
            for _, r in forecast_df.iterrows()
        ]
        result_ids = db.save_forecast_results(id_model, forecast_rows_db)

        # --- SHAP (PermutationExplainer) ---
        print("Вычисляем SHAP для LSTM...")
        shap_background_raw = X_full[-5:].reshape(5, -1)
        shap_input_raw = np.array([
            build_lstm_window(source_df, final_artifacts, final_window).reshape(-1)
            for _ in range(len(forecast_df))
        ])

        n_flat = final_window * len(final_artifacts["input_columns"])

        def ensemble_predict_flat(X_flat: np.ndarray) -> np.ndarray:
            X_3d = X_flat.reshape(-1, final_window, len(final_artifacts["input_columns"]))
            preds = np.mean(
                [m.predict(X_3d, verbose=0).flatten() for m in final_models], axis=0
            )
            return preds

        feat_names_flat = [
            f"{feat}_t-{final_window - t}"
            for t in range(final_window)
            for feat in final_artifacts["input_columns"]
        ]

        explainer = shap.PermutationExplainer(
            ensemble_predict_flat,
            shap_background_raw,
            feature_names=feat_names_flat,
        )
        explanation = explainer(shap_input_raw, max_evals=2 * n_flat + 1)
        shap_values_flat = np.asarray(explanation.values)

        shap_contributions = []
        for row_idx, (_, frow) in enumerate(forecast_df.iterrows()):
            year = int(frow["year"])
            scenario = frow["scenario"]
            id_result = result_ids.get((year, scenario))
            if id_result is None:
                continue

            row_shap = shap_values_flat[row_idx]
            ranked = np.argsort(np.abs(row_shap))[::-1]

            seen_features: dict[str, float] = {}
            for flat_idx in ranked:
                feat_full = feat_names_flat[flat_idx]
                feat_base = feat_full.rsplit("_t-", 1)[0]
                if feat_base not in seen_features:
                    seen_features[feat_base] = float(row_shap[flat_idx])

            for rank, (feat_base, contrib) in enumerate(
                sorted(seen_features.items(), key=lambda x: abs(x[1]), reverse=True)
            ):
                shap_contributions.append({
                    "id_result":          id_result,
                    "id_indicator":       feature_ids.get(feat_base, 0),
                    "contribution_value": contrib,
                    "direction":          "positive" if contrib >= 0 else "negative",
                    "rank_position":      rank + 1,
                })

        db.save_shap_contributions(shap_contributions)

        db.update_run_status(id_run, "completed")
        print(f"Модель сохранена: id_model={id_model}")
        print("=== Готово ===")

    except Exception as exc:
        db.update_run_status(id_run, "error", str(exc))
        raise


if __name__ == "__main__":
    main()
