"""
Обучение ML-моделей (sklearn).

Запуск:
    python scripts/train_ml.py

Что делает:
    1. Читает данные из indicators.db
    2. Перебирает 3 конфигурации × все модели через walk-forward валидацию
       (baseline / log_target / lagged_log_target)
    3. Сохраняет лучшую модель (.pkl) в models/ml/
    4. Пишет ForecastRun, Model, ModelMetric, ForecastResult, ShapContribution в models.db
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import db
from forecast_utils import (
    SELECTED_FEATURES,
    TARGET_COLUMN,
    YEAR_COLUMN,
    FORECAST_HORIZON,
    build_all_scenario_frames,
    evaluate_predictions,
    remap_forecast_rows_for_db,
    walk_forward_validate,
)

MODELS_DIR = ROOT / "models" / "ml"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN_SIZE = 12

# ---------------------------------------------------------------------------
# Лаговые признаки
# ---------------------------------------------------------------------------

LAG_FEATURES = [
    TARGET_COLUMN,
    "Ввод в действие основных фондов",
    "Валовой региональный продукт на душу населения",
    "Среднемесячная номинальная начисленная заработная плата работников организаций",
    "Потребительские расходы в среднем на душу населения",
]


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    lagged_df = df.copy()
    for col in columns:
        for lag in lags:
            lagged_df[f"{col}_lag_{lag}"] = lagged_df[col].shift(lag)
    return lagged_df


# ---------------------------------------------------------------------------
# Конфигурации моделей — scaled (линейные, SVR, KNN, MLP) и tree (деревья)
# ---------------------------------------------------------------------------

def make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])


def build_scaled_model(estimator) -> Pipeline:
    return make_pipeline(estimator)


def build_tree_model(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   estimator),
    ])


MODEL_CONFIGS: dict[str, Pipeline] = {
    "LinearRegression": build_scaled_model(LinearRegression()),
    "Ridge":            build_scaled_model(Ridge(alpha=1.0)),
    "Lasso":            build_scaled_model(Lasso(alpha=0.01, max_iter=10000, random_state=42)),
    "ElasticNet":       build_scaled_model(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)),
    "SVR_rbf":          build_scaled_model(SVR(kernel="rbf", C=10.0, epsilon=0.1)),
    "KNN":              build_scaled_model(KNeighborsRegressor(n_neighbors=3, weights="distance")),
    "DecisionTree":     build_tree_model(DecisionTreeRegressor(max_depth=4, random_state=42)),
    "RandomForest":     build_tree_model(RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)),
    "ExtraTrees":       build_tree_model(ExtraTreesRegressor(n_estimators=300, max_depth=5, random_state=42)),
    "GradientBoosting": build_tree_model(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)),
    "AdaBoost":         build_tree_model(AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
    "MLPRegressor":     build_scaled_model(MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.0001, learning_rate_init=0.001, max_iter=5000, random_state=42,
    )),
}


# ---------------------------------------------------------------------------
# Обёртка с преобразованием таргета (TransformedTargetRegressor)
# ---------------------------------------------------------------------------

def wrap_for_target(model: Pipeline, use_log_target: bool = False) -> TransformedTargetRegressor:
    if use_log_target:
        transformer = FunctionTransformer(np.log1p, np.expm1, validate=True)
    else:
        transformer = StandardScaler()
    return TransformedTargetRegressor(regressor=clone(model), transformer=transformer)


class LogTargetWrapper:
    """Совместимый helper для тестов; production-код использует wrap_for_target."""

    def __init__(self, pipeline: Pipeline):
        self._model = wrap_for_target(pipeline, use_log_target=True)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)


def build_fit_predict(model: Pipeline, use_log: bool = False):
    """Возвращает замыкание fit_predict(X_train, y_train, X_test) -> float для walk_forward_validate."""
    def fit_predict(X_train, y_train, X_test) -> float:
        wrapped = wrap_for_target(model, use_log_target=use_log)
        wrapped.fit(X_train, y_train)
        return float(wrapped.predict(X_test)[0])
    return fit_predict


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    all_ids = db.get_all_indicators()["id_indicator"].tolist()
    return db.load_dataset(all_ids)


def get_feature_indicator_ids() -> dict[str, int]:
    indicators = db.get_all_indicators()
    name_to_id = dict(zip(indicators["name"], indicators["id_indicator"]))
    return {feat: name_to_id[feat] for feat in SELECTED_FEATURES if feat in name_to_id}


# ---------------------------------------------------------------------------
# Walk-forward по всем конфигурациям
# ---------------------------------------------------------------------------

def run_walk_forward(
    model_df: pd.DataFrame,
    lagged_model_df: pd.DataFrame,
    lagged_features: list[str],
) -> pd.DataFrame:
    print("Walk-forward валидация...")
    summary_rows = []

    configs = [
        ("baseline",           model_df,        SELECTED_FEATURES, False),
        ("log_target",         model_df,        SELECTED_FEATURES, True),
        ("lagged_log_target",  lagged_model_df, lagged_features,   True),
    ]

    for config_name, data_df, features, use_log in configs:
        for algo_name, pipeline in MODEL_CONFIGS.items():
            exp_name = f"{config_name}|{algo_name}"
            fn = build_fit_predict(pipeline, use_log=use_log)
            summary, _ = walk_forward_validate(data_df, features, fn, MIN_TRAIN_SIZE)
            row = {
                "experiment_id": exp_name,
                "configuration":  config_name,
                "algorithm":      algo_name,
                "use_log":        use_log,
                "walk_forward_mae":  float(summary["walk_forward_mae"].iloc[0]),
                "walk_forward_rmse": float(summary["walk_forward_rmse"].iloc[0]),
                "walk_forward_mape": float(summary["walk_forward_mape"].iloc[0]),
            }
            summary_rows.append(row)
            print(f"  {exp_name}: MAPE={row['walk_forward_mape']:.4f}")

    return pd.DataFrame(summary_rows).sort_values("walk_forward_mape").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Финальное обучение, прогноз, SHAP
# ---------------------------------------------------------------------------

def train_and_forecast(
    model_df: pd.DataFrame,
    lagged_model_df: pd.DataFrame,
    lagged_features: list[str],
    best_configuration: str,
    best_algo: str,
    id_run: int,
    feature_ids: dict[str, int],
    wf_summary_row: pd.Series,
) -> int:
    use_log = bool(wf_summary_row["use_log"])

    if best_configuration == "baseline":
        production_df       = model_df.copy()
        production_features = SELECTED_FEATURES
    elif best_configuration == "log_target":
        production_df       = model_df.copy()
        production_features = SELECTED_FEATURES
    else:
        production_df       = lagged_model_df.dropna().reset_index(drop=True)
        production_features = lagged_features

    pipeline = MODEL_CONFIGS[best_algo]
    production_model = wrap_for_target(pipeline, use_log_target=use_log)
    production_model.fit(
        production_df[production_features].to_numpy(),
        production_df[TARGET_COLUMN].to_numpy(),
    )

    # --- Сохранение модели на диск ---
    model_path = MODELS_DIR / f"{best_configuration}_{best_algo}.pkl"
    joblib.dump(production_model, model_path)

    # --- Запись в БД ---
    id_model = db.save_model(
        id_run=id_run,
        model_name=f"{best_configuration}|{best_algo}",
        model_type="machine_learning",
        algorithm=best_algo,
        status="trained",
        model_path=str(model_path.relative_to(ROOT)),
    )
    db.save_metrics(
        id_model,
        float(wf_summary_row["walk_forward_mae"]),
        float(wf_summary_row["walk_forward_rmse"]),
        float(wf_summary_row["walk_forward_mape"]),
    )

    # --- Сценарный прогноз ---
    base_year = int(model_df[YEAR_COLUMN].max())
    latest_fact_year = base_year
    scenario_frames = build_all_scenario_frames(model_df, SELECTED_FEATURES, FORECAST_HORIZON)
    raw_forecast_rows = []
    shap_meta = []  # (year, scenario_name, X_row)

    for scenario_name, scenario_cfg_df in scenario_frames.items():
        forecast_base_df = pd.concat([model_df, scenario_cfg_df], ignore_index=True)

        for forecast_year in range(latest_fact_year + 1, latest_fact_year + FORECAST_HORIZON + 1):
            step = forecast_year - latest_fact_year
            forecast_idx = forecast_base_df.index[forecast_base_df[YEAR_COLUMN] == forecast_year][0]

            if best_configuration == "lagged_log_target":
                temp_lagged_df = add_lag_features(
                    forecast_base_df.iloc[: forecast_idx + 1].copy(), LAG_FEATURES, lags=(1, 2)
                )
                feature_row = temp_lagged_df.iloc[[-1]][production_features].to_numpy()
            else:
                feature_row = forecast_base_df.iloc[[forecast_idx]][production_features].to_numpy()

            prediction = float(production_model.predict(feature_row)[0])
            forecast_base_df.loc[forecast_idx, TARGET_COLUMN] = prediction

            raw_forecast_rows.append({
                "year":           forecast_year,
                "scenario_name":  scenario_name,
                "forecast_value": prediction,
            })
            shap_meta.append((forecast_year, scenario_name, feature_row[0]))

    forecast_rows = remap_forecast_rows_for_db(raw_forecast_rows, base_year)
    result_ids = db.save_forecast_results(id_model, forecast_rows)

    shap_meta_remapped = []
    for year, scenario_name, x_row in shap_meta:
        if year == base_year + 1:
            if scenario_name == "Базовый":
                shap_meta_remapped.append((year, "Оценка", x_row))
        else:
            shap_meta_remapped.append((year, scenario_name, x_row))

    # --- SHAP ---
    print(f"  Вычисляем SHAP для {best_algo}...")
    background = production_df[production_features].to_numpy()
    predict_fn = production_model.predict

    explainer = shap.Explainer(predict_fn, background)
    all_X = np.array([m[2] for m in shap_meta_remapped])
    explanation = explainer(all_X)
    shap_values = np.asarray(explanation.values)

    shap_contributions = []
    for row_idx, (year, scenario_name, _) in enumerate(shap_meta_remapped):
        id_result = result_ids.get((year, scenario_name))
        if id_result is None:
            continue
        row_shap = shap_values[row_idx]
        ranked = np.argsort(np.abs(row_shap))[::-1]

        for rank, feat_idx in enumerate(ranked):
            feat_name = production_features[feat_idx]
            contrib = float(row_shap[feat_idx])
            shap_contributions.append({
                "id_result":          id_result,
                "id_indicator":       feature_ids.get(feat_name, 0),
                "contribution_value": contrib,
                "direction":          "positive" if contrib >= 0 else "negative",
                "rank_position":      rank + 1,
            })

    db.save_shap_contributions(shap_contributions)
    return id_model


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    print("=== ML-пайплайн обучения ===")

    df = load_data()
    print(f"Данные загружены: {len(df)} лет ({int(df[YEAR_COLUMN].min())}–{int(df[YEAR_COLUMN].max())})")

    lagged_df = add_lag_features(df, LAG_FEATURES, lags=(1, 2))
    lagged_features: list[str] = SELECTED_FEATURES + [
        col for col in lagged_df.columns
        if any(col == f"{base}_lag_{lag}" for base in LAG_FEATURES for lag in (1, 2))
    ]

    feature_ids = get_feature_indicator_ids()
    target_id   = db.get_indicator_id(TARGET_COLUMN)
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
        wf_summary = run_walk_forward(df, lagged_df, lagged_features)
        print("\nТоп-5 конфигураций:")
        print(wf_summary[["experiment_id", "walk_forward_mape", "walk_forward_mae"]].head())

        best = wf_summary.iloc[0]
        best_configuration = best["configuration"]
        best_algo          = best["algorithm"]
        print(f"\nЛучшая конфигурация: {best['experiment_id']} (MAPE={best['walk_forward_mape']:.4f})")

        id_model = train_and_forecast(
            df, lagged_df, lagged_features,
            best_configuration, best_algo,
            id_run, feature_ids, best,
        )
        print(f"Модель сохранена: id_model={id_model}")

        db.update_run_status(id_run, "completed")
        print("=== Готово ===")

    except Exception as exc:
        db.update_run_status(id_run, "error", str(exc))
        raise


if __name__ == "__main__":
    main()
