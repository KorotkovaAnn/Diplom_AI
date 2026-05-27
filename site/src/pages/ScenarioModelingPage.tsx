import { useEffect, useMemo, useState } from 'react'
import { observer } from 'mobx-react-lite'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useRootStore, type ApiModel, type ScenarioType } from '../stores/rootStore.tsx'

interface FactorState {
  id: string
  name: string
  slider: number
  contribution: number
}

const MAX_VISIBLE_FACTORS = 8

export const ScenarioModelingPage = observer(function ScenarioModelingPage() {
  const { dashboards } = useRootStore()
  const [factors, setFactors] = useState<FactorState[]>([])

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' })
    if (!dashboards.dashboardData && !dashboards.isLoading) {
      void dashboards.initializeDashboard()
    } else if (dashboards.modelOptions.length === 0 && !dashboards.isModelsLoading) {
      void dashboards.fetchModels()
    }
  }, [dashboards])

  const shapFactors = useMemo(() => {
    return dashboards.shapContributions
      .slice()
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, MAX_VISIBLE_FACTORS)
  }, [dashboards.shapContributions])

  useEffect(() => {
    setFactors((current) => {
      const slidersById = new Map(current.map((factor) => [factor.id, factor.slider]))
      return shapFactors.map((factor) => ({
        id: factor.id,
        name: factor.name,
        contribution: factor.contribution,
        slider: slidersById.get(factor.id) ?? 0,
      }))
    })
  }, [shapFactors])

  const baseSeries = dashboards.timeSeries
  const forecastPoints = baseSeries.filter((point) => point.forecast != null)

  const currentMultiplier = useMemo(
    () => 1 + factors.reduce((acc, factor) => acc + (factor.contribution * factor.slider) / 100, 0),
    [factors],
  )

  const scenarioSeries = useMemo(
    () =>
      baseSeries.map((point) => {
        if (point.forecast == null) return point
        return {
          ...point,
          scenario: point.forecast * currentMultiplier,
        }
      }),
    [baseSeries, currentMultiplier],
  )

  const baseForecastSum = forecastPoints.reduce((sum, point) => sum + (point.forecast ?? 0), 0)
  const scenarioSum = scenarioSeries.reduce((sum, point) => sum + (point.scenario ?? 0), 0)
  const diffPercent =
    baseForecastSum > 0 ? ((scenarioSum - baseForecastSum) / baseForecastSum) * 100 : 0
  const diffClass =
    diffPercent > 0.5 ? 'positive' : diffPercent < -0.5 ? 'negative' : 'neutral'

  const selectedModel = dashboards.modelInfo
  const firstForecastYear = forecastPoints[0]?.year
  const lastForecastYear = forecastPoints[forecastPoints.length - 1]?.year
  const hasScenarioData = forecastPoints.length > 0 && factors.length > 0

  function resetFactors() {
    setFactors((current) => current.map((factor) => ({ ...factor, slider: 0 })))
  }

  return (
    <div>
      <h1 className="page-title">Сценарное моделирование</h1>
      <p className="page-subtitle">
        What-if анализ: изменяйте влияние факторов и сразу оценивайте отклонение от выбранного прогноза.
      </p>

      {dashboards.error && <div className="forecast-error">{dashboards.error}</div>}

      <section className="glass-panel scenario-control-panel">
        <div className="scenario-control-grid">
          <div className="dashboard-filter-group">
            <div className="dashboard-filter-label">Модель прогноза</div>
            <select
              className="dashboard-select"
              value={dashboards.selectedModelId ?? ''}
              onChange={(event) => dashboards.setSelectedModel(event.target.value)}
              disabled={dashboards.isModelsLoading || dashboards.modelOptions.length === 0}
            >
              {dashboards.isModelsLoading ? (
                <option value="">Загрузка моделей...</option>
              ) : dashboards.modelOptions.length > 0 ? (
                dashboards.modelOptions.map((model) => (
                  <option key={model.id} value={model.id}>
                    {modelOptionLabel(model)}
                  </option>
                ))
              ) : (
                <option value="">Нет обученных моделей</option>
              )}
            </select>
          </div>

          <div className="dashboard-filter-group">
            <div className="dashboard-filter-label">Исходный сценарий</div>
            <div className="dashboard-segmented">
              {(['base', 'conservative'] as ScenarioType[]).map((scenario) => (
                <button
                  key={scenario}
                  type="button"
                  className={segmentClass(dashboards.scenarioType === scenario)}
                  onClick={() => dashboards.setScenarioType(scenario)}
                >
                  {scenario === 'base' ? 'Базовый' : 'Консервативный'}
                </button>
              ))}
            </div>
          </div>

          <div className="scenario-context-card">
            <span>Период прогноза</span>
            <strong>
              {firstForecastYear && lastForecastYear ? `${firstForecastYear}-${lastForecastYear}` : '-'}
            </strong>
          </div>

          <div className="scenario-context-card">
            <span>Модель</span>
            <strong>{selectedModel ? modelTypeLabel(selectedModel.type) : '-'}</strong>
          </div>
        </div>
      </section>

      {!hasScenarioData && (
        <section className="glass-panel scenario-empty">
          {dashboards.isLoading
            ? 'Загрузка сценарных данных...'
            : 'Нет SHAP-факторов для выбранной модели и сценария.'}
        </section>
      )}

      {hasScenarioData && (
        <div className="scenario-layout glass-panel">
          <aside className="scenario-sidebar">
            <div className="scenario-sidebar-head">
              <div>
                <div className="home-section-title">Факторы сценария</div>
                <p className="home-section-subtitle">
                  Ползунок меняет силу вклада фактора относительно выбранного прогноза.
                </p>
              </div>
              <button type="button" className="btn btn-outline btn-small" onClick={resetFactors}>
                Сбросить
              </button>
            </div>

            <div className="scenario-factors">
              {factors.map((factor) => (
                <div
                  key={factor.id}
                  className={`scenario-factor ${factor.contribution >= 0 ? 'positive' : 'negative'}`}
                >
                  <div className="scenario-factor-header">
                    <span className="scenario-factor-name">{factor.name}</span>
                    <span className="scenario-factor-value">
                      {factor.slider > 0 ? '+' : ''}
                      {factor.slider.toFixed(0)}%
                    </span>
                  </div>
                  <div className="scenario-factor-meta">
                    Вклад: {formatSignedPercent(factor.contribution * 100)}
                  </div>
                  <input
                    type="range"
                    min={-50}
                    max={50}
                    value={factor.slider}
                    onChange={(event) =>
                      setFactors((previous) =>
                        previous.map((item) =>
                          item.id === factor.id ? { ...item, slider: Number(event.target.value) } : item,
                        ),
                      )
                    }
                  />
                </div>
              ))}
            </div>
          </aside>

          <section className="scenario-main">
            <div className="scenario-summary-grid">
              <ScenarioKpi
                label="Исходный прогноз"
                value={formatCompactValue(baseForecastSum, dashboards.targetUnit)}
                caption={`Сумма за ${forecastPoints.length} прогнозных года`}
              />
              <ScenarioKpi
                label="Сценарий пользователя"
                value={formatCompactValue(scenarioSum, dashboards.targetUnit)}
                caption="С учетом текущих ползунков"
              />
              <ScenarioKpi
                label="Отклонение"
                value={`${diffPercent >= 0 ? '+' : ''}${diffPercent.toFixed(1)}%`}
                caption="Относительно исходного прогноза"
                tone={diffClass}
              />
            </div>

            <div className="scenario-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scenarioSeries}>
                  <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis
                    tickFormatter={(value: number) =>
                      value >= 1000 ? `${(value / 1000).toFixed(0)} тыс.` : String(value)
                    }
                  />
                  <Tooltip
                    formatter={
                      ((value: number) =>
                        value.toLocaleString('ru-RU', { maximumFractionDigits: 0 })) as never
                    }
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="Фактические данные"
                    stroke="#6b7280"
                    strokeWidth={1.8}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    name="Исходный прогноз"
                    stroke="#94a3b8"
                    strokeDasharray="5 5"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="scenario"
                    name="Сценарий пользователя"
                    stroke="#2563eb"
                    strokeWidth={2.4}
                    dot={{ r: 3 }}
                    activeDot={{ r: 4.5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="scenario-table-wrapper">
              <table className="scenario-table">
                <thead>
                  <tr>
                    <th>Год</th>
                    <th>Исходный прогноз</th>
                    <th>Сценарий пользователя</th>
                    <th>Разница</th>
                  </tr>
                </thead>
                <tbody>
                  {scenarioSeries
                    .filter((point) => point.forecast != null)
                    .map((point) => {
                      const base = point.forecast ?? 0
                      const scenario = point.scenario ?? base
                      const diff = base ? ((scenario - base) / base) * 100 : 0
                      return (
                        <tr key={point.year}>
                          <td>{point.year}</td>
                          <td>{base.toLocaleString('ru-RU', { maximumFractionDigits: 0 })}</td>
                          <td>{scenario.toLocaleString('ru-RU', { maximumFractionDigits: 0 })}</td>
                          <td className={diff >= 0 ? 'positive' : 'negative'}>
                            {formatSignedPercent(diff)}
                          </td>
                        </tr>
                      )
                    })}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}
    </div>
  )
})

function ScenarioKpi({
  label,
  value,
  caption,
  tone = 'neutral',
}: {
  label: string
  value: string
  caption: string
  tone?: 'positive' | 'negative' | 'neutral'
}) {
  return (
    <div className="scenario-kpi">
      <div className="dashboard-kpi-label">{label}</div>
      <div className={`scenario-kpi-value ${tone}`}>{value}</div>
      <div className="dashboard-kpi-caption">{caption}</div>
    </div>
  )
}

function segmentClass(active: boolean): string {
  return active ? 'dashboard-segment dashboard-segment-active' : 'dashboard-segment'
}

function modelTypeLabel(type: string): string {
  if (type === 'machine_learning' || type === 'ml') return 'ML'
  if (type === 'neural_network' || type === 'lstm') return 'LSTM'
  return type
}

function modelOptionLabel(model: ApiModel): string {
  const mape = model.mape != null ? ` · MAPE ${(model.mape * 100).toFixed(1)}%` : ''
  return `${modelTypeLabel(model.type)} · ${model.name}${mape}`
}

function formatSignedPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`
}

function formatCompactValue(value: number, unit: string): string {
  if (!Number.isFinite(value) || value === 0) return '-'
  const unitLower = unit.toLowerCase()
  if (unitLower.includes('млн') && unitLower.includes('руб')) {
    return `${(value / 1000).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} млрд руб.`
  }
  return value.toLocaleString('ru-RU', { maximumFractionDigits: 0 })
}
