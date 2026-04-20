# Прогнозирование инвестиций в основной капитал Тюменской области

## Цель проекта

Построить прогноз **объёма инвестиций в основной капитал** Тюменской области (без ХМАО и ЯНАО) на 4 года вперёд двумя подходами — табличными ML-моделями и LSTM — с ошибкой оценки менее 10%.

## Данные

Файл `data/Datasety-1_cleaned.xlsx` — 26 наблюдений (1998–2023), 27 показателей:

- **Целевой признак:** Объём инвестиций в основной капитал (млн руб.)
- **Экзогенные факторы:** ВРП, доходы населения, зарплата, потребительские расходы, численность населения, кредитование, стоимость основных фондов, ИПЦ, доля убыточных организаций и др.

## Структура ноутбуков

### 1. `notebooks/data_quality_and_correlation_analysis.ipynb` — ML-подход

| Этап | Описание |
|------|----------|
| Анализ данных | Проверка типов, пропусков, корреляционный анализ (Pearson, Spearman, Kendall) |
| Отбор признаков | 12 факторов, отобранных по корреляции с целевым показателем |
| ML-пайплайн | Линейные модели, деревья, ансамбли, SVR, KNN, MLP с `StandardScaler` + `SimpleImputer` |
| Конфигурации | 3 варианта: baseline, log(target), lagged + log(target) |
| Валидация | Walk-forward validation по всем возможным временным шагам |
| Прогноз | Сценарный прогноз на 4 года (оценка + базовый + консервативный) |

### 2. `notebooks/lstm_investment_forecast.ipynb` — LSTM-подход

| Этап | Описание |
|------|----------|
| Динамические признаки | diff, pct_change, rolling_mean, momentum на основе ключевых факторов |
| Перебор конфигураций | 24 комбинации: 3 набора фичей × 2 размера окна × 2 архитектуры × 2 режима таргета |
| Валидация | Walk-forward по top-5 конфигурациям |
| Финальная модель | Ансамбль из 5 LSTM (разные seed) с лучшей конфигурацией |
| Прогноз | Рекурсивный сценарный прогноз на 4 года |

## Концепция прогнозирования

### Горизонт и структура прогноза

Прогноз строится на 4 года вперёд от последнего года с фактическими данными:

- **Год 1 — оценка:** единственное точечное значение (наиболее вероятное)
- **Годы 2–4 — два сценария:**
  - **Базовый вариант** — наиболее вероятное развитие с учётом ожидаемых внешних условий и мер, направленных на рост экономики. Основной драйвер — внутренний спрос (потребительский и инвестиционный).
  - **Консервативный вариант** — усиление санкционного давления, рост дисбалансов из-за торговых войн, более медленный рост мирового спроса на нефть, более низкая численность занятых, замедленное смягчение ДКП, более низкие темпы кредитования и инвестиционной активности.

### Итеративная экстраполяция факторов

Для каждого прогнозного года экзогенные факторы строятся **итеративно** — каждый год от предыдущего:

```
feature[step] = feature[step-1] + annual_slope × multiplier
```

- `annual_slope` — наклон линейного тренда по последним 3 годам фактических данных
- `multiplier` — сценарный множитель, определяющий силу и направление изменения

### Авто-детекция паттерна консервативного сценария

Траектория консервативного сценария определяется **автоматически** по двум последним годовым изменениям целевого показателя:

| Факт (2 последних года) | Паттерн прогноза (годы 2–4) | Логика |
|---|---|---|
| ↑ ↑ | ↑ ↑ ↓ | Инерция роста → коррекция |
| ↓ ↓ | ↓ ↓ ↑ | Продолжение спада → восстановление |
| ↓ ↑ или ↑ ↓ | ↓ ↑ ↑ | Краткий откат → восстановление |

Каждый шаг «вверх» или «вниз» задаётся двумя множителями:

| Направление | Позитивные факторы | Негативные факторы |
|---|---|---|
| **up** (рост) | +0.60 | +0.90 |
| **down** (спад) | −0.40 | +1.50 |

Позитивные факторы (ВРП, доходы, зарплата, кредитование и др.) при `down` **снижаются**, негативные факторы (ИПЦ, доля убыточных организаций) при `down` **усиливаются**.

### Сценарные множители базового варианта

Базовый сценарий использует фиксированные множители на всех шагах:
- Позитивные факторы: **1.05** (рост чуть выше тренда)
- Негативные факторы: **0.95** (негатив чуть ниже тренда)

## Валидация

Фактическое значение за 2024 год составляет **395 030 млн руб.** (395,03 млрд). Оценка модели на 2024 год сравнивается с этим значением для контроля точности. Целевой порог ошибки — **менее 10%**.

## Ориентиры по величине прогноза

По официальным документам (полная Тюменская область) базовый сценарий на 2027 год — 530 800 млн руб. Для нашего показателя (без ХМАО и ЯНАО) ожидаемый уровень на 2027 год — **порядка 490 000 млн руб.** по базовому сценарию.

## Базы данных

Готовые пайплайны сохраняют данные и результаты в два SQLite-файла в папке `data-base/`.

### `data-base/indicators.db` — исходные данные и показатели

```
Sphere
├── id_sphere   INTEGER PK AUTOINCREMENT
└── name        TEXT UNIQUE          -- название сферы (экономика, население и т.д.)

Indicator
├── id_indicator  INTEGER PK AUTOINCREMENT
├── id_sphere     INTEGER FK → Sphere
├── name          TEXT               -- название показателя
└── unit          TEXT               -- единица измерения

Dataset
├── id_indicator  INTEGER FK → Indicator  ┐ PK составной
├── year          INTEGER                 ┘
└── value         REAL                    -- значение показателя за год
```

### `data-base/models.db` — модели, метрики и прогнозы

```
ForecastRun
├── id_run             INTEGER PK AUTOINCREMENT
├── forecast_horizon   INTEGER     -- горизонт прогноза (лет)
├── train_start_year   INTEGER
├── train_end_year     INTEGER
├── status             TEXT        -- pending / done / error
└── error_message      TEXT

RunIndicatorRole
├── id_run         INTEGER FK → ForecastRun  ┐ PK составной
├── id_indicator   INTEGER                   ┘  (ссылка на indicators.db)
└── role           TEXT                      -- target / feature

Model
├── id_model     INTEGER PK AUTOINCREMENT
├── id_run       INTEGER FK → ForecastRun
├── model_name   TEXT        -- произвольное имя модели
├── model_type   TEXT        -- ml / lstm
├── algorithm    TEXT        -- конкретный алгоритм (Ridge, LSTM и т.д.)
├── status       TEXT        -- trained / failed
└── model_path   TEXT        -- путь к сериализованному файлу модели

ModelMetric
├── id_metric   INTEGER PK AUTOINCREMENT
├── id_model    INTEGER FK → Model UNIQUE
├── mae         REAL
├── rmse        REAL
└── mape        REAL

ForecastResult
├── id_result        INTEGER PK AUTOINCREMENT
├── id_model         INTEGER FK → Model
├── year             INTEGER
├── scenario_name    TEXT    -- оценка / базовый / консервативный
└── forecast_value   REAL

ShapContribution
├── id_contribution      INTEGER PK AUTOINCREMENT
├── id_result            INTEGER FK → ForecastResult
├── id_indicator         INTEGER     -- ссылка на indicators.db
├── contribution_value   REAL
├── direction            TEXT        -- positive / negative
└── rank_position        INTEGER     -- порядковый номер по важности
```

## Python-скрипты для обучения

Помимо ноутбуков реализован полноценный производственный пайплайн на Python-скриптах. Скрипты читают данные из `indicators.db`, обучают модели и сохраняют все результаты в `models.db`.

### Структура скриптов

```
scripts/
├── db.py             — слой работы с базами данных
├── forecast_utils.py — сценарная логика и утилиты валидации
├── train_ml.py       — обучение ML-моделей (sklearn)
└── train_lstm.py     — обучение LSTM-ансамбля (Keras)

models/
├── ml/               — .pkl файлы sklearn-моделей (joblib)
└── lstm/             — .keras файлы весов LSTM-ансамбля
```

### Описание скриптов

**`scripts/db.py`** — единственная точка доступа к обеим базам данных:
- `load_dataset(indicator_ids)` — читает данные из `indicators.db`, возвращает широкую таблицу (строки — годы, столбцы — показатели)
- `get_indicator_id(name)` / `get_all_indicators()` — справочные запросы
- `save_run`, `save_model`, `save_metrics`, `save_forecast_results`, `save_shap_contributions` — запись всех результатов обучения в `models.db`

**`scripts/forecast_utils.py`** — общая логика для обоих пайплайнов:
- `SELECTED_FEATURES` — список 12 отобранных признаков
- `detect_conservative_pattern(df)` — авто-детекция паттерна консервативного сценария по последним двум изменениям целевого показателя
- `build_all_scenario_frames(df)` — итеративная экстраполяция всех сценариев
- `walk_forward_validate(data, feature_columns, fit_predict_fn)` — универсальная walk-forward валидация
- `evaluate_predictions(y_true, y_pred)` — метрики MAE, RMSE, MAPE

**`scripts/train_ml.py`** — ML-пайплайн:
1. Загружает данные из `indicators.db`
2. Перебирает все sklearn-модели × raw/log-target через walk-forward валидацию
3. Выбирает лучшую конфигурацию по MAPE
4. Обучает финальную модель, вычисляет SHAP-вклады для каждого сценарного прогноза
5. Сохраняет `.pkl` в `models/ml/` и все результаты в `models.db`

**`scripts/train_lstm.py`** — LSTM-пайплайн:
1. Строит динамические признаки (diff, pct_change, rolling_mean, momentum)
2. Grid search по 24 конфигурациям (2 набора фичей × 2 окна × 2 архитектуры × 2 режима таргета)
3. Walk-forward валидация лучшей конфигурации
4. Обучает ансамбль из 5 LSTM с bias-коррекцией
5. Рекурсивный сценарный прогноз, SHAP через `PermutationExplainer`
6. Сохраняет `.keras` в `models/lstm/` и все результаты в `models.db`

### Запуск скриптов

```bash
python scripts/train_ml.py
python scripts/train_lstm.py
```

## Тесты

102 теста покрывают всю бизнес-логику проекта. Реальные базы данных не затрагиваются — тесты работают с in-memory SQLite.

### Структура тестов

```
tests/
├── conftest.py           — общие фикстуры: in-memory БД, sample_df
├── test_db.py            — тесты слоя работы с БД
├── test_forecast_utils.py — тесты логики прогнозирования
├── test_train_ml.py      — тесты утилит ML-пайплайна
└── test_train_lstm.py    — тесты утилит LSTM-пайплайна
```

### Что тестируется

**`tests/test_db.py`** (23 теста) — все функции записи и чтения `db.py`:
- `load_dataset` — широкая таблица, фильтрация по id, тип колонок
- `get_indicator_id` — корректный id и исключение для несуществующего названия
- `save_run` / `update_run_status` — создание запуска, обновление статуса и сообщения об ошибке
- `save_model` / `update_model_path` — сохранение всех полей модели
- `save_metrics` — запись метрик и upsert (повторный вызов обновляет, не дублирует)
- `save_forecast_results` — маппинг `(year, scenario) → id_result`
- `save_shap_contributions` — порядок рангов и направление вклада

**`tests/test_forecast_utils.py`** (31 тест) — вся логика прогнозирования:
- `evaluate_predictions` — идеальный прогноз даёт нули, RMSE ≥ MAE, MAPE как относительная ошибка
- `detect_conservative_pattern` — все три паттерна (↑↑, ↓↓, смешанный), граничный случай с одной точкой
- `compute_annual_slopes` — положительный/отрицательный/нулевой тренд, использование только последних `TREND_WINDOW` лет
- `get_step_multiplier` — fixed vs auto_pattern, позитивные vs негативные факторы, step=1 совпадает с step=2
- `build_future_exogenous_rows` — горизонт, последовательность годов, NaN в таргете, числовые признаки
- `build_all_scenario_frames` — оба сценария присутствуют, различаются между собой
- `walk_forward_validate` — количество фолдов, наличие всех метрик, неотрицательность

**`tests/test_train_ml.py`** (15 тестов) — утилиты `train_ml.py`:
- `make_pipeline` — три шага (imputer → scaler → model), обработка NaN
- `LogTargetWrapper` — предсказания положительные, обёртка отличается от plain-варианта
- `build_fit_predict` — возвращает float, log-вариант даёт положительный результат, детерминированность

**`tests/test_train_lstm.py`** (33 теста) — утилиты `train_lstm.py`:
- `add_dynamic_features` — исходные колонки сохраняются, diff считается корректно, оригинал не мутирует
- `get_dynamic_feature_columns` — включает `SELECTED_FEATURES`, динамические extras, нет дублей
- `transform_target` / `inverse_transform_target` — raw-режим тождественен, round-trip для raw и log
- `make_recency_weights` — все положительные, последний год максимален, монотонный рост, сумма = n
- `prepare_sequences` — форма `(n-window, window, n_features)`, ключи артефактов, include_target_history
- `build_lstm_model` — компилируется для всех трёх архитектур, форма выхода `(batch, 1)`, ValueError для неизвестной архитектуры
- `build_lstm_window` — форма `(1, window, n_features)`

### Запуск тестов

```bash
# Все тесты
.venv/bin/pytest tests/ -v

# Один файл
.venv/bin/pytest tests/test_forecast_utils.py -v

# Один тест
.venv/bin/pytest tests/test_db.py::TestMetrics::test_save_metrics_upsert -v
```

> **Важно:** использовать `.venv/bin/pytest`, а не системный `pytest` — глобальное окружение Anaconda несовместимо с зависимостями проекта.

## Как запустить

1. Создать виртуальное окружение Python 3.11+
2. Открыть нужный ноутбук в Jupyter / PyCharm
3. Выполнить все ячейки последовательно (каждый ноутбук устанавливает зависимости в первой ячейке через `%pip install`)

Порядок запуска не важен — ноутбуки независимы друг от друга.
