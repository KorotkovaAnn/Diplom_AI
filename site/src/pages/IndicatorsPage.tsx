import { useEffect, useMemo, useState } from 'react'

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

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' })
    void fetchIndicatorOptions()
  }, [])

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
