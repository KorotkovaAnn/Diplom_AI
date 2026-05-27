import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

interface ApiIndicator {
  id: number
  sphere: string
  name: string
  unit: string
}

interface HistoryPoint {
  year: number
  value: number
}

interface IndicatorHistoryRow {
  id: number
  role: 'target' | 'feature'
  sphere: string
  name: string
  unit: string
  valuesByYear: Map<number, number>
}

const HISTORY_YEAR_COUNT = 8

export function IndicatorsPage() {
  const [targetIndicators, setTargetIndicators] = useState<ApiIndicator[]>([])
  const [featureIndicators, setFeatureIndicators] = useState<ApiIndicator[]>([])
  const [selectedTargetId, setSelectedTargetId] = useState('')
  const [historyByIndicatorId, setHistoryByIndicatorId] = useState<Map<number, HistoryPoint[]>>(new Map())
  const [isFiltersLoading, setIsFiltersLoading] = useState(false)
  const [isTableLoading, setIsTableLoading] = useState(false)
  const [isTableVisible, setIsTableVisible] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [chartFeatureId, setChartFeatureId] = useState('')
  const [chartTimeRange, setChartTimeRange] = useState<'all' | 'last10' | 'last5'>('all')
  const [chartHistory, setChartHistory] = useState<HistoryPoint[]>([])
  const [isChartLoading, setIsChartLoading] = useState(false)

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' })
    void fetchIndicatorOptions()
  }, [])

  useEffect(() => {
    if (featureIndicators.length > 0 && !chartFeatureId) {
      setChartFeatureId(String(featureIndicators[0].id))
    }
  }, [featureIndicators, chartFeatureId])

  useEffect(() => {
    if (!chartFeatureId) return
    let cancelled = false
    setIsChartLoading(true)
    fetch(`/api/indicators/${chartFeatureId}/history`)
      .then((res) => {
        if (!res.ok) throw new Error()
        return res.json()
      })
      .then((data: HistoryPoint[]) => {
        if (!cancelled) setChartHistory(data)
      })
      .catch(() => {
        if (!cancelled) setChartHistory([])
      })
      .finally(() => {
        if (!cancelled) setIsChartLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [chartFeatureId])

  async function fetchIndicatorOptions() {
    setIsFiltersLoading(true)
    setError(null)
    try {
      const [targetRes, featureRes] = await Promise.all([
        fetch('/api/indicators?role=target'),
        fetch('/api/indicators?role=feature'),
      ])

      if (!targetRes.ok || !featureRes.ok) {
        throw new Error('Indicators request failed')
      }

      const targets: ApiIndicator[] = await targetRes.json()
      const features: ApiIndicator[] = await featureRes.json()
      setTargetIndicators(targets)
      setFeatureIndicators(features)
      setSelectedTargetId((current) => current || (targets[0] ? String(targets[0].id) : ''))
    } catch {
      setError('Не удалось загрузить список показателей. Убедитесь, что API-сервер запущен.')
    } finally {
      setIsFiltersLoading(false)
    }
  }

  const selectedTarget = useMemo(() => {
    return targetIndicators.find((indicator) => String(indicator.id) === selectedTargetId)
  }, [selectedTargetId, targetIndicators])

  const tableYears = useMemo(() => {
    const selectedHistory = selectedTarget
      ? historyByIndicatorId.get(selectedTarget.id) ?? []
      : []

    return selectedHistory.slice(-HISTORY_YEAR_COUNT).map((point) => point.year)
  }, [historyByIndicatorId, selectedTarget])

  const tableRows = useMemo<IndicatorHistoryRow[]>(() => {
    if (!selectedTarget) return []

    const targetRow = buildHistoryRow(selectedTarget, 'target', historyByIndicatorId)
    const featureRows = featureIndicators
      .filter((indicator) => indicator.id !== selectedTarget.id)
      .map((indicator) => buildHistoryRow(indicator, 'feature', historyByIndicatorId))

    return [targetRow, ...featureRows]
  }, [featureIndicators, historyByIndicatorId, selectedTarget])

  const selectedChartFeature = useMemo(
    () => featureIndicators.find((i) => String(i.id) === chartFeatureId),
    [featureIndicators, chartFeatureId],
  )

  const filteredChartData = useMemo(() => {
    if (chartHistory.length === 0) return []
    if (chartTimeRange === 'all') return chartHistory
    const maxYear = Math.max(...chartHistory.map((p) => p.year))
    const minYear = chartTimeRange === 'last5' ? maxYear - 4 : maxYear - 9
    return chartHistory.filter((p) => p.year >= minYear)
  }, [chartHistory, chartTimeRange])

  const chartStats = useMemo(() => computeTimeSeriesStats(filteredChartData), [filteredChartData])

  async function handleRenderTable() {
    if (!selectedTarget) return

    setIsTableLoading(true)
    setError(null)
    try {
      const indicators = [selectedTarget, ...featureIndicators]
      const histories = await fetchHistories(indicators.map((indicator) => indicator.id))
      setHistoryByIndicatorId(histories)
      setIsTableVisible(true)
    } catch {
      setError('Не удалось загрузить значения показателей. Убедитесь, что API-сервер запущен.')
      setIsTableVisible(false)
    } finally {
      setIsTableLoading(false)
    }
  }

  return (
    <div>
      <h1 className="page-title">Показатели</h1>
      <p className="page-subtitle">
        Фактические значения target-показателя и feature-факторов для проверки входных данных,
        влияющих на прогноз.
      </p>

      {error && <div className="forecast-error">{error}</div>}

      <section className="glass-panel" style={{ padding: '12px 18px 10px', marginBottom: 16 }}>
        <div className="forecast-toolbar">
          <div className="dashboard-filter-group indicators-target-filter">
            <div className="dashboard-filter-label">Прогнозный показатель</div>
            <select
              className="dashboard-select"
              value={selectedTargetId}
              onChange={(event) => {
                setSelectedTargetId(event.target.value)
                setIsTableVisible(false)
              }}
              disabled={isFiltersLoading || targetIndicators.length === 0}
            >
              {isFiltersLoading ? (
                <option value="">Загрузка...</option>
              ) : targetIndicators.length > 0 ? (
                targetIndicators.map((indicator) => (
                  <option key={indicator.id} value={indicator.id}>
                    {indicator.name}
                  </option>
                ))
              ) : (
                <option value="">Нет target-показателей</option>
              )}
            </select>
          </div>

          <button
            type="button"
            className="btn btn-primary btn-small forecast-render-button"
            onClick={handleRenderTable}
            disabled={isFiltersLoading || isTableLoading || !selectedTarget}
          >
            {isTableLoading ? 'Загрузка...' : 'Отрисовать'}
          </button>
        </div>
      </section>

      {isTableVisible && (
        <section className="glass-panel indicators-table-panel">
          <div className="indicators-history-table-wrapper">
            <table className="forecast-table indicators-history-table">
              <thead>
                <tr>
                  <th>Тип</th>
                  <th>Сфера</th>
                  <th>Показатель</th>
                  {tableYears.map((year) => (
                    <th key={year}>{year}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableRows.map((row) => (
                  <tr
                    key={`${row.role}:${row.id}`}
                    className={row.role === 'target' ? 'indicators-target-row' : undefined}
                  >
                    <td>
                      <span className={`indicator-role-pill ${row.role}`}>
                        {row.role === 'target' ? 'Target' : 'Feature'}
                      </span>
                    </td>
                    <td>{row.sphere}</td>
                    <td>
                      <div className="forecast-name">{row.name}</div>
                      <div className="forecast-unit">{row.unit}</div>
                    </td>
                    {tableYears.map((year, index) => (
                      <IndicatorValueCell
                        key={year}
                        value={row.valuesByYear.get(year)}
                        previousValue={previousYearValue(row, tableYears, index)}
                      />
                    ))}
                  </tr>
                ))}
                {tableRows.length === 0 && (
                  <tr>
                    <td colSpan={3 + tableYears.length} className="forecast-empty-cell">
                      Нет значений для выбранного прогнозного показателя
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Блок просмотра графиков */}
      <section style={{ marginTop: 24 }}>
        <h2 className="page-title" style={{ fontSize: 18 }}>
          Просмотр графиков
        </h2>
        <p className="page-subtitle">
          Визуализация исторической динамики и статистический анализ факторного показателя.
        </p>

        <div className="glass-panel" style={{ padding: '12px 18px 10px', marginBottom: 16 }}>
          <div className="forecast-toolbar">
            <div className="dashboard-filter-group indicators-target-filter">
              <div className="dashboard-filter-label">Прогнозный показатель</div>
              <select
                className="dashboard-select"
                value={selectedTargetId}
                onChange={(event) => setSelectedTargetId(event.target.value)}
                disabled={isFiltersLoading || targetIndicators.length === 0}
              >
                {isFiltersLoading ? (
                  <option value="">Загрузка...</option>
                ) : targetIndicators.length > 0 ? (
                  targetIndicators.map((indicator) => (
                    <option key={indicator.id} value={indicator.id}>
                      {indicator.name}
                    </option>
                  ))
                ) : (
                  <option value="">Нет target-показателей</option>
                )}
              </select>
            </div>

            <div className="dashboard-filter-group" style={{ flex: '1 1 320px', maxWidth: 520 }}>
              <div className="dashboard-filter-label">Факторный показатель</div>
              <select
                className="dashboard-select"
                value={chartFeatureId}
                onChange={(event) => setChartFeatureId(event.target.value)}
                disabled={isFiltersLoading || featureIndicators.length === 0}
              >
                {isFiltersLoading ? (
                  <option value="">Загрузка...</option>
                ) : featureIndicators.length > 0 ? (
                  featureIndicators.map((indicator) => (
                    <option key={indicator.id} value={indicator.id}>
                      {indicator.name}
                    </option>
                  ))
                ) : (
                  <option value="">Нет факторных показателей</option>
                )}
              </select>
            </div>

            <div className="dashboard-filter-group">
              <div className="dashboard-filter-label">Период</div>
              <div className="dashboard-segmented">
                {(['all', 'last10', 'last5'] as const).map((range) => (
                  <button
                    key={range}
                    type="button"
                    className={`dashboard-segment${chartTimeRange === range ? ' dashboard-segment-active' : ''}`}
                    onClick={() => setChartTimeRange(range)}
                  >
                    {range === 'all' ? 'Весь ряд' : range === 'last10' ? '10 лет' : '5 лет'}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="indicators-chart-grid">
          <div className="glass-panel indicators-chart-panel">
            <h3 className="home-section-title">
              {selectedChartFeature?.name ?? 'Динамика показателя'}
            </h3>
            <p className="home-section-subtitle">
              {selectedChartFeature
                ? `${selectedChartFeature.unit} · ${selectedChartFeature.sphere}`
                : 'Выберите факторный показатель'}
            </p>
            {isChartLoading ? (
              <div
                style={{
                  padding: '60px 0',
                  textAlign: 'center',
                  color: '#9ca3af',
                  fontSize: 13,
                }}
              >
                Загрузка данных…
              </div>
            ) : filteredChartData.length > 0 ? (
              <div className="indicators-chart-wrapper">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={filteredChartData}>
                    <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis
                      tickFormatter={(v: number) =>
                        v >= 1_000_000
                          ? `${(v / 1_000_000).toFixed(1)} млн`
                          : v >= 1000
                            ? `${(v / 1000).toFixed(0)} тыс.`
                            : String(Math.round(v))
                      }
                    />
                    <Tooltip
                      formatter={
                        ((value: number) =>
                          value.toLocaleString('ru-RU', {
                            maximumFractionDigits: 1,
                          })) as never
                      }
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      name={selectedChartFeature?.name ?? 'Значение'}
                      stroke="#2563eb"
                      strokeWidth={2.4}
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div
                style={{
                  padding: '60px 0',
                  textAlign: 'center',
                  color: '#9ca3af',
                  fontSize: 13,
                }}
              >
                Нет данных для выбранного показателя
              </div>
            )}
          </div>

          <div className="glass-panel indicators-stats-panel">
            <h3 className="home-section-title">Статистика ряда</h3>
            <p className="home-section-subtitle">
              Описательные характеристики за выбранный период
            </p>
            {chartStats.count > 0 ? (
              <div className="indicators-stats-list">
                <StatItem
                  label="Наблюдений"
                  value={String(chartStats.count)}
                  caption={
                    filteredChartData.length > 1
                      ? `${filteredChartData[0].year} — ${filteredChartData[filteredChartData.length - 1].year}`
                      : undefined
                  }
                />
                <StatItem
                  label="Последнее значение"
                  value={chartStats.last ? formatValue(chartStats.last.value) : '—'}
                  caption={chartStats.last ? `${chartStats.last.year} г.` : undefined}
                />
                <StatItem
                  label="Максимум"
                  value={chartStats.max ? formatValue(chartStats.max.value) : '—'}
                  caption={chartStats.max ? `${chartStats.max.year} г.` : undefined}
                  accent="positive"
                />
                <StatItem
                  label="Минимум"
                  value={chartStats.min ? formatValue(chartStats.min.value) : '—'}
                  caption={chartStats.min ? `${chartStats.min.year} г.` : undefined}
                  accent="negative"
                />
                <StatItem
                  label="Среднее значение"
                  value={chartStats.mean != null ? formatValue(chartStats.mean) : '—'}
                />
                <StatItem
                  label="Среднегодовой темп (CAGR)"
                  value={
                    chartStats.cagr != null
                      ? `${chartStats.cagr >= 0 ? '+' : ''}${chartStats.cagr.toFixed(1)}%`
                      : '—'
                  }
                  accent={
                    chartStats.cagr != null
                      ? chartStats.cagr >= 0
                        ? 'positive'
                        : 'negative'
                      : undefined
                  }
                />
                <StatItem
                  label="Ср. годовой прирост"
                  value={
                    chartStats.avgYoyChange != null
                      ? `${chartStats.avgYoyChange >= 0 ? '+' : ''}${chartStats.avgYoyChange.toFixed(1)}%`
                      : '—'
                  }
                  accent={
                    chartStats.avgYoyChange != null
                      ? chartStats.avgYoyChange >= 0
                        ? 'positive'
                        : 'negative'
                      : undefined
                  }
                />
                <StatItem
                  label="Волатильность (σ YoY)"
                  value={
                    chartStats.volatility != null
                      ? `${chartStats.volatility.toFixed(1)}%`
                      : '—'
                  }
                  caption="стандартное отклонение годовых изменений"
                />
                <StatItem
                  label="Коэффициент вариации"
                  value={chartStats.cv != null ? `${chartStats.cv.toFixed(1)}%` : '—'}
                />
                <StatItem
                  label="Общее изменение за период"
                  value={
                    chartStats.totalChange != null
                      ? `${chartStats.totalChange >= 0 ? '+' : ''}${chartStats.totalChange.toFixed(1)}%`
                      : '—'
                  }
                  accent={
                    chartStats.totalChange != null
                      ? chartStats.totalChange >= 0
                        ? 'positive'
                        : 'negative'
                      : undefined
                  }
                />
              </div>
            ) : (
              <div
                style={{
                  padding: '24px 0',
                  textAlign: 'center',
                  color: '#9ca3af',
                  fontSize: 13,
                }}
              >
                Нет данных для расчёта статистики
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}

async function fetchHistories(indicatorIds: number[]): Promise<Map<number, HistoryPoint[]>> {
  const entries = await Promise.all(
    indicatorIds.map(async (id) => {
      const res = await fetch(`/api/indicators/${id}/history`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const history: HistoryPoint[] = await res.json()
      return [id, history] as const
    }),
  )

  return new Map(entries)
}

function buildHistoryRow(
  indicator: ApiIndicator,
  role: IndicatorHistoryRow['role'],
  historyByIndicatorId: Map<number, HistoryPoint[]>,
): IndicatorHistoryRow {
  const valuesByYear = new Map<number, number>()
  ;(historyByIndicatorId.get(indicator.id) ?? []).forEach((point) => {
    valuesByYear.set(point.year, point.value)
  })

  return {
    id: indicator.id,
    role,
    sphere: indicator.sphere,
    name: indicator.name,
    unit: indicator.unit,
    valuesByYear,
  }
}

function IndicatorValueCell({
  value,
  previousValue,
}: {
  value: number | undefined
  previousValue: number | undefined
}) {
  const change =
    value != null && previousValue != null && previousValue !== 0
      ? ((value - previousValue) / previousValue) * 100
      : null
  const changeClass = change == null ? 'neutral' : change >= 0 ? 'positive' : 'negative'

  return (
    <td className={`forecast-cell-change ${changeClass}`}>
      <div>{formatValue(value)}</div>
      {change != null && (
        <span>
          {change >= 0 ? '▲' : '▼'} {Math.abs(change).toFixed(1)}%
        </span>
      )}
    </td>
  )
}

function previousYearValue(
  row: IndicatorHistoryRow,
  years: number[],
  currentIndex: number,
): number | undefined {
  for (let index = currentIndex - 1; index >= 0; index -= 1) {
    const value = row.valuesByYear.get(years[index])
    if (value != null) return value
  }
  return undefined
}

function formatValue(value: number | undefined): string {
  if (value == null) return '-'
  return value.toLocaleString('ru-RU', { maximumFractionDigits: 1 })
}

interface TimeSeriesStats {
  count: number
  mean: number | null
  min: { value: number; year: number } | null
  max: { value: number; year: number } | null
  last: { value: number; year: number } | null
  cagr: number | null
  avgYoyChange: number | null
  volatility: number | null
  cv: number | null
  totalChange: number | null
}

function computeTimeSeriesStats(data: HistoryPoint[]): TimeSeriesStats {
  const empty: TimeSeriesStats = {
    count: 0,
    mean: null,
    min: null,
    max: null,
    last: null,
    cagr: null,
    avgYoyChange: null,
    volatility: null,
    cv: null,
    totalChange: null,
  }
  if (data.length === 0) return empty

  const n = data.length
  const values = data.map((p) => p.value)
  const sum = values.reduce((s, v) => s + v, 0)
  const mean = sum / n

  let minPoint = data[0]
  let maxPoint = data[0]
  for (const p of data) {
    if (p.value < minPoint.value) minPoint = p
    if (p.value > maxPoint.value) maxPoint = p
  }

  const first = data[0]
  const last = data[n - 1]

  let cagr: number | null = null
  const years = last.year - first.year
  if (years > 0 && first.value > 0 && last.value > 0) {
    cagr = (Math.pow(last.value / first.value, 1 / years) - 1) * 100
  }

  const yoyChanges: number[] = []
  for (let i = 1; i < n; i++) {
    if (data[i - 1].value !== 0) {
      yoyChanges.push(
        ((data[i].value - data[i - 1].value) / Math.abs(data[i - 1].value)) * 100,
      )
    }
  }

  const avgYoyChange =
    yoyChanges.length > 0
      ? yoyChanges.reduce((s, v) => s + v, 0) / yoyChanges.length
      : null

  let volatility: number | null = null
  if (yoyChanges.length > 1 && avgYoyChange != null) {
    const variance =
      yoyChanges.reduce((s, v) => s + (v - avgYoyChange) ** 2, 0) / (yoyChanges.length - 1)
    volatility = Math.sqrt(variance)
  }

  const stdDev = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / n)
  const cv = mean !== 0 ? (stdDev / Math.abs(mean)) * 100 : null

  const totalChange =
    first.value !== 0
      ? ((last.value - first.value) / Math.abs(first.value)) * 100
      : null

  return {
    count: n,
    mean,
    min: { value: minPoint.value, year: minPoint.year },
    max: { value: maxPoint.value, year: maxPoint.year },
    last: { value: last.value, year: last.year },
    cagr,
    avgYoyChange,
    volatility,
    cv,
    totalChange,
  }
}

function StatItem({
  label,
  value,
  caption,
  accent,
}: {
  label: string
  value: string
  caption?: string
  accent?: 'positive' | 'negative'
}) {
  return (
    <div className="indicators-stat-item">
      <div className="indicators-stat-label">{label}</div>
      <div className={`indicators-stat-value${accent ? ` ${accent}` : ''}`}>{value}</div>
      {caption && <div className="indicators-stat-caption">{caption}</div>}
    </div>
  )
}
