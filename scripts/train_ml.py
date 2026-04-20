"""
Обучение ML-моделей (sklearn).

Запуск:
    python scripts/train_ml.py

Что делает:
    1. Читает данные из indicators.db
    2. Обучает все конфигурации sklearn-моделей с walk-forward валидацией
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import db
from forecast_utils import (
    SELECTED_FEATURES,
    TARGET_COLUMN,
    YEAR_COLUMN,
    FORECAST_HORIZON,
    build_all_scenario_frames,
    evaluate_predictions,
    walk_forward_validate,
)

MODELS_DIR = ROOT / "models" / "ml"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN_SIZE = 12

# ---------------------------------------------------------------------------
# Конфигурации моделей
# ---------------------------------------------------------------------------

def make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])


MODEL_CONFIGS: dict[str, Pipeline] = {
    "LinearRegression":    make_pipeline(LinearRegression()),
    "Ridge":               make_pipeline(Ridge(alpha=1.0)),
    "Lasso":               make_pipeline(Lasso(alpha=0.1, max_iter=5000)),
    "ElasticNet":          make_pipeline(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    "RandomForest":        make_pipeline(RandomForestRegressor(n_estimators=100, random_state=42)),
    "GradientBoosting":    make_pipeline(GradientBoostingRegressor(n_estimators=100, random_state=42)),
    "SVR":                 make_pipeline(SVR(kernel="rbf", C=1.0)),
    "KNN":                 make_pipeline(KNeighborsRegressor(n_neighbors=5)),
    "MLP":                 make_pipeline(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)),
}


# ---------------------------------------------------------------------------
# Log-target wrapper
# ---------------------------------------------------------------------------

class LogTargetWrapper:
    def __init__(self, pipeline: Pipeline):
        self._pipe = pipeline

    def fit(self, X, y):
        self._pipe.fit(X, np.log1p(y))
        return self

    def predict(self, X) -> np.ndarray:
        return np.expm1(self._pipe.predict(X))


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    all_ids = db.get_all_indicators()["id_indicator"].tolist()
    return db.load_dataset(all_ids)


def get_feature_indicator_ids() -> dict[str, int]:
    indicators = db.get_all_indicators()
    name_to_id = dict(zip(indicators["name"], indicators["id_indicator"]))
    return {feat: name_to_id[feat] for feat in SELECTED_FEATURES if feat in name_to_id}


def build_fit_predict(pipeline, use_log: bool = False):
    """Возвращает замыкание для walk_forward_validate."""
    def fit_predict(X_train, y_train, X_test) -> float:
        if use_log:
            wrapper = LogTargetWrapper(pipeline)
            wrapper.fit(X_train, y_train)
            return float(wrapper.predict(X_test)[0])
        else:
            pipeline.fit(X_train, y_train)
            return float(pipeline.predict(X_test)[0])
    return fit_predict


# ---------------------------------------------------------------------------
# Walk-forward для всех моделей
# ---------------------------------------------------------------------------

def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    print("Walk-forward валидация...")
    summary_rows = []

    for algo_name, pipeline in MODEL_CONFIGS.items():
        for use_log in [False, True]:
            config_name = f"{algo_name}{'_log' if use_log else ''}"
            fn = build_fit_predict(pipeline, use_log=use_log)
            summary, _ = walk_forward_validate(df, SELECTED_FEATURES, fn, MIN_TRAIN_SIZE)
            summary_rows.append({
                "config":     config_name,
                "algorithm":  algo_name,
                "use_log":    use_log,
                **summary.iloc[0].to_dict(),
            })
            print(f"  {config_name}: MAPE={summary['walk_forward_mape'].iloc[0]:.4f}")

    return pd.DataFrame(summary_rows).sort_values("walk_forward_mape").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Финальное обучение + прогноз + SHAP
# ---------------------------------------------------------------------------

def train_and_forecast(
    df: pd.DataFrame,
    algo_name: str,
    use_log: bool,
    id_run: int,
    feature_ids: dict[str, int],
) -> int:
    """Обучает модель на всех данных, сохраняет в БД и на диск. Возвращает id_model."""
    pipeline = MODEL_CONFIGS[algo_name]
    X_train = df[SELECTED_FEATURES].to_numpy()
    y_train = df[TARGET_COLUMN].to_numpy()

    if use_log:
        model = LogTargetWrapper(pipeline)
        model.fit(X_train, y_train)
        predict_fn = model.predict
    else:
        pipeline.fit(X_train, y_train)
        predict_fn = pipeline.predict

    # --- Walk-forward метрики финальной модели ---
    fn = build_fit_predict(pipeline, use_log)
    summary, _ = walk_forward_validate(df, SELECTED_FEATURES, fn, MIN_TRAIN_SIZE)
    mae  = float(summary["walk_forward_mae"].iloc[0])
    rmse = float(summary["walk_forward_rmse"].iloc[0])
    mape = float(summary["walk_forward_mape"].iloc[0])

    # --- Сохранение модели на диск ---
    model_path = MODELS_DIR / f"{algo_name}{'_log' if use_log else ''}.pkl"
    joblib.dump(pipeline if not use_log else model, model_path)

    # --- Запись в БД ---
    id_model = db.save_model(
        id_run=id_run,
        model_name=f"{algo_name}{'_log' if use_log else ''}",
        model_type="machine_learning",
        algorithm=algo_name,
        status="trained",
        model_path=str(model_path.relative_to(ROOT)),
    )
    db.save_metrics(id_model, mae, rmse, mape)

    # --- Сценарный прогноз ---
    scenario_frames = build_all_scenario_frames(df, SELECTED_FEATURES, FORECAST_HORIZON)
    forecast_rows = []
    shap_meta = []  # (id_result_key, X_row, year, scenario)

    for scenario_name, future_df in scenario_frames.items():
        X_future = future_df[SELECTED_FEATURES].to_numpy()
        preds = predict_fn(X_future)
        for i, year in enumerate(future_df[YEAR_COLUMN].tolist()):
            forecast_rows.append({
                "year":           int(year),
                "scenario_name":  scenario_name,
                "forecast_value": float(preds[i]),
            })
            shap_meta.append((int(year), scenario_name, X_future[i]))

    result_ids = db.save_forecast_results(id_model, forecast_rows)

    # --- SHAP ---
    print(f"  Вычисляем SHAP для {algo_name}...")
    background = df[SELECTED_FEATURES].to_numpy()
    explainer = shap.Explainer(predict_fn, background)

    all_X = np.array([m[2] for m in shap_meta])
    explanation = explainer(all_X)
    shap_values = np.asarray(explanation.values)

    shap_contributions = []
    for row_idx, (year, scenario_name, _) in enumerate(shap_meta):
        id_result = result_ids[(year, scenario_name)]
        row_shap = shap_values[row_idx]
        ranked = np.argsort(np.abs(row_shap))[::-1]

        for rank, feat_idx in enumerate(ranked):
            feat_name = SELECTED_FEATURES[feat_idx]
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

    # Определяем роли показателей
    feature_ids = get_feature_indicator_ids()
    target_id   = db.get_indicator_id(TARGET_COLUMN)

    train_start = int(df[YEAR_COLUMN].min())
    train_end   = int(df[YEAR_COLUMN].max())

    # --- ForecastRun ---
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
        # --- Walk-forward по всем конфигурациям ---
        wf_summary = run_walk_forward(df)
        print("\nТоп-5 конфигураций:")
        print(wf_summary[["config", "walk_forward_mape", "walk_forward_mae"]].head())

        # --- Лучшая конфигурация ---
        best = wf_summary.iloc[0]
        best_algo  = best["algorithm"]
        best_log   = bool(best["use_log"])
        print(f"\nЛучшая конфигурация: {best['config']} (MAPE={best['walk_forward_mape']:.4f})")

        # --- Финальное обучение ---
        id_model = train_and_forecast(df, best_algo, best_log, id_run, feature_ids)
        print(f"Модель сохранена: id_model={id_model}")

        db.update_run_status(id_run, "completed")
        print("=== Готово ===")

    except Exception as exc:
        db.update_run_status(id_run, "error", str(exc))
        raise


if __name__ == "__main__":
    main()
