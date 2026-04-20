import { useMemo, useState } from 'react'
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
import { useRootStore } from '../stores/rootStore.tsx'

interface FactorState {
  id: string
  name: string
  slider: number
  contribution: number
}

export const ScenarioModelingPage = observer(function ScenarioModelingPage() {
  const { dashboards } = useRootStore()

  const baseSeries = dashboards.timeSeries

  const [factors, setFactors] = useState<FactorState[]>(
    dashboards.shapContributions.map((f) => ({
      id: f.id,
      name: f.name,
      contribution: f.contribution,
      slider: 0,
    })),
  )

  const currentMultiplier = useMemo(
    () => 1 + factors.reduce((acc, f) => acc + (f.contribution * f.slider) / 100, 0),
    [factors],
  )

  const scenarioSeries = useMemo(
    () =>
      baseSeries.map((point) => {
        if (!point.forecast) return point
        return {
          ...point,
          scenario: point.forecast * currentMultiplier,
        }
      }),
    [baseSeries, currentMultiplier],
  )

  const baseForecastSum = baseSeries.reduce(
    (sum, p) => (p.forecast ? sum + p.forecast : sum),
    0,
  )
  const scenarioSum = scenarioSeries.reduce(
    (sum, p) => (p.scenario ? sum + p.scenario : sum),
    0,
  )
  const diffPercent =
    baseForecastSum > 0 ? ((scenarioSum - baseForecastSum) / baseForecastSum) * 100 : 0

  const diffClass =
    diffPercent > 0.5 ? 'positive' : diffPercent < -0.5 ? 'negative' : 'neutral'

  return (
    <div>
      <h1 className="page-title">Сценарное моделирование</h1>
      <p className="page-subtitle">
        What-if анализ: изменяйте значения факторов и мгновенно оценивайте влияние на
        прогноз показателя.
      </p>

      <div className="scenario-layout glass-panel">
        <aside className="scenario-sidebar">
          <div className="home-section-title" style={{ marginBottom: 8 }}>
            Факторы сценария
          </div>
          <p className="home-section-subtitle">
            Сдвиг ползунков на ±50% имитирует усиление или ослабление влияния соответствующего
            фактора.
          </p>
          <div className="scenario-factors">
            {factors.map((factor) => (
              <div key={factor.id} className="scenario-factor">
                <div className="scenario-factor-header">
                  <span className="scenario-factor-name">{factor.name}</span>
                  <span className="scenario-factor-value">
                    {factor.slider > 0 ? '+' : ''}
                    {factor.slider.toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={-50}
                  max={50}
                  value={factor.slider}
                  onChange={(e) =>
                    setFactors((prev) =>
                      prev.map((f) =>
                        f.id === factor.id ? { ...f, slider: Number(e.target.value) } : f,
                      ),
                    )
                  }
                />
              </div>
            ))}
          </div>
          <div className="scenario-buttons">
            <button
              type="button"
              className="btn btn-outline btn-small"
              onClick={() =>
                setFactors((prev) => prev.map((f) => ({ ...f, slider: 0 })))
              }
            >
              Сбросить к базовому
            </button>
          </div>
        </aside>

        <section className="scenario-main">
          <div className="scenario-summary">
            <div>
              <div className="dashboard-kpi-label">Изменение прогноза по сценарию</div>
              <div className={`scenario-diff ${diffClass}`}>
                {diffPercent >= 0 ? '▲' : '▼'} {Math.abs(diffPercent).toFixed(1)}%
              </div>
              <div className="dashboard-kpi-caption">
                относительно суммарного прогноза по базовому сценарию
              </div>
            </div>
          </div>

          <div className="scenario-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={scenarioSeries}>
                <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Исторические данные"
                  stroke="#6b7280"
                  strokeWidth={1.8}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  name="Базовый прогноз"
                  stroke="#94a3b8"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="scenario"
                  name="Текущий сценарий"
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
                  <th>Базовый прогноз</th>
                  <th>Сценарий пользователя</th>
                  <th>Разница</th>
                </tr>
              </thead>
              <tbody>
                {scenarioSeries
                  .filter((p) => p.forecast != null)
                  .map((p) => {
                    const base = p.forecast ?? 0
                    const scen = (p as any).scenario ?? base
                    const diff = base ? ((scen - base) / base) * 100 : 0
                    return (
                      <tr key={p.year}>
                        <td>{p.year}</td>
                        <td>{base.toLocaleString('ru-RU', { maximumFractionDigits: 0 })}</td>
                        <td>{scen.toLocaleString('ru-RU', { maximumFractionDigits: 0 })}</td>
                        <td className={diff >= 0 ? 'positive' : 'negative'}>
                          {diff >= 0 ? '▲' : '▼'} {Math.abs(diff).toFixed(1)}%
                        </td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  )
})

