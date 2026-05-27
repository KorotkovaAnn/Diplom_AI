import { useCallback, useEffect, useMemo, useState } from 'react'
import { observer } from 'mobx-react-lite'
import { useRootStore, type ApiModel, type IndicatorCategory } from '../stores/rootStore.tsx'

interface ForecastPoint {
  indicator_id: number
  year: number
  scenario: string
  value: number
}

interface HistoryPoint {
  year: number
  value: number
}

interface ForecastTableColumn {
  year: number
  label: string
}

interface ForecastTableRow {
  id: string
  indicatorName: string
  categoryLabel: string
  unit: string
  scenario: string
  valuesByYear: Map<number, number>
}

const CATEGORY_LABELS: Record<IndicatorCategory, string> = {
  economy: 'Экономика',
  investment: 'Инвестиции и основные фонды',
  construction: 'Строительство',
  demography: 'Демография',
  labour: 'Рынок труда',
  social: 'Уровень жизни и социальная сфера',
}

const SCENARIO_ORDER = ['Базовый', 'Консервативный', 'Оценка']
const ESTIMATE_SCENARIO = 'Оценка'

export const ForecastsPage = observer(function ForecastsPage() {
  const { indicators } = useRootStore()
  const [selectedCategory, setSelectedCategory] = useState<IndicatorCategory | ''>('')
  const [models, setModels] = useState<ApiModel[]>([])
  const [selectedModelId, setSelectedModelId] = useState('')
  const [forecastPoints, setForecastPoints] = useState<ForecastPoint[]>([])
  const [historyByIndicatorId, setHistoryByIndicatorId] = useState<Map<number, HistoryPoint[]>>(new Map())
  const [isTableVisible, setIsTableVisible] = useState(false)
  const [isModelsLoading, setIsModelsLoading] = useState(false)
  const [isForecastsLoading, setIsForecastsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    indicators.fetchIndicators('target')
    void fetchModels()
  }, [indicators])

  async function fetchModels() {
    setIsModelsLoading(true)
    try {
      const res = await fetch('/api/models')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: ApiModel[] = await res.json()
      setModels(data)
      setSelectedModelId((current) => current || (data[0] ? String(data[0].id) : ''))
    } catch {
      setModels([])
    } finally {
      setIsModelsLoading(false)
    }
  }

  const targetCategories = useMemo(() => {
    const unique = new Set(indicators.indicators.map((indicator) => indicator.category))
    return Array.from(unique)
  }, [indicators.indicators])

  useEffect(() => {
    if (!selectedCategory && targetCategories.length > 0) {
      setSelectedCategory(targetCategories[0])
    }
  }, [selectedCategory, targetCategories])

  const targetIndicators = useMemo(() => {
    return indicators.indicators.filter(
      (indicator) => !selectedCategory || indicator.category === selectedCategory,
    )
  }, [indicators.indicators, selectedCategory])

  const selectedModel = useMemo(() => {
    return models.find((model) => String(model.id) === selectedModelId) ?? null
  }, [models, selectedModelId])

  const tableColumns = useMemo<ForecastTableColumn[]>(() => {
    const columns = new Map<number, ForecastTableColumn>()

    for (const history of historyByIndicatorId.values()) {
      history.slice(-2).forEach((point) => {
        columns.set(point.year, { year: point.year, label: `${point.year} факт` })
      })
    }

    for (const point of forecastPoints) {
      const label = point.scenario === ESTIMATE_SCENARIO ? `${point.year} оценка` : String(point.year)
      columns.set(point.year, { year: point.year, label })
    }

    return Array.from(columns.values()).sort((a, b) => a.year - b.year)
  }, [forecastPoints, historyByIndicatorId])

  const tableRows = useMemo<ForecastTableRow[]>(() => {
    const indicatorById = new Map(targetIndicators.map((indicator) => [Number(indicator.id), indicator]))
    const pointsByIndicator = groupForecastPoints(forecastPoints)
    const rows: ForecastTableRow[] = []

    for (const [indicatorId, byScenario] of pointsByIndicator) {
      const indicator = indicatorById.get(indicatorId)
      if (!indicator) continue

      const estimateValues = byScenario.get(ESTIMATE_SCENARIO) ?? new Map<number, number>()
      const scenarioNames = Array.from(byScenario.keys()).filter((scenario) => scenario !== ESTIMATE_SCENARIO)
      const visibleScenarios = scenarioNames.length > 0 ? scenarioNames : [ESTIMATE_SCENARIO]

      for (const scenario of visibleScenarios) {
        const valuesByYear = new Map<number, number>()

        const history = historyByIndicatorId.get(indicatorId) ?? []
        history.slice(-2).forEach((point) => valuesByYear.set(point.year, point.value))
        estimateValues.forEach((value, year) => valuesByYear.set(year, value))
        ;(byScenario.get(scenario) ?? new Map<number, number>()).forEach((value, year) => {
          valuesByYear.set(year, value)
        })

        rows.push({
          id: `${indicatorId}:${scenario}`,
          indicatorName: indicator.name,
          categoryLabel: CATEGORY_LABELS[indicator.category],
          unit: indicator.unit,
          scenario,
          valuesByYear,
        })
      }
    }

    return rows.sort((a, b) => {
      const byIndicator = a.indicatorName.localeCompare(b.indicatorName, 'ru')
      if (byIndicator !== 0) return byIndicator
      return scenarioRank(a.scenario) - scenarioRank(b.scenario)
    })
  }, [forecastPoints, historyByIndicatorId, targetIndicators])

  const handleRenderTable = useCallback(async () => {
    if (targetIndicators.length === 0 || models.length === 0) return

    setIsForecastsLoading(true)
    setError(null)
    try {
      const modelQuery = selectedModelId ? `?model_id=${selectedModelId}` : ''
      const [forecastRes, histories] = await Promise.all([
        fetch(`/api/forecasts${modelQuery}`),
        fetchLatestHistory(targetIndicators.map((indicator) => Number(indicator.id))),
      ])

      if (!forecastRes.ok) throw new Error(`HTTP ${forecastRes.status}`)

      const data: ForecastPoint[] = await forecastRes.json()
      setForecastPoints(data)
      setHistoryByIndicatorId(histories)
      setIsTableVisible(true)
    } catch {
      setError('Не удалось загрузить прогнозы. Убедитесь, что API-сервер запущен.')
      setIsTableVisible(false)
    } finally {
      setIsForecastsLoading(false)
    }
  }, [models.length, selectedModelId, targetIndicators])

  useEffect(() => {
    if (
      selectedModelId &&
      targetIndicators.length > 0 &&
      models.length > 0 &&
      !indicators.isLoading &&
      !isModelsLoading
    ) {
      void handleRenderTable()
    }
  }, [
    handleRenderTable,
    indicators.isLoading,
    isModelsLoading,
    models.length,
    selectedModelId,
    targetIndicators.length,
  ])

  function handleExportExcel() {
    exportForecastTable({
      rows: tableRows,
      columns: tableColumns,
      modelLabel: selectedModel ? modelOptionLabel(selectedModel) : '-',
    })
  }

  return (
    <div>
      <h1 className="page-title">Сводка прогнозов</h1>
      <p className="page-subtitle">
        Табличная сводка прогнозов по target-показателям из базы результатов моделей.
      </p>

      {error && <div className="forecast-error">{error}</div>}

      <section className="glass-panel" style={{ padding: '12px 18px 10px', marginBottom: 16 }}>
        <div className="forecast-toolbar">
          <div className="dashboard-filter-group">
            <div className="dashboard-filter-label">Категория target-показателя</div>
            <select
              className="dashboard-select"
              value={selectedCategory}
              onChange={(event) => {
                setSelectedCategory(event.target.value as IndicatorCategory)
              }}
              disabled={indicators.isLoading || targetCategories.length === 0}
            >
              {indicators.isLoading ? (
                <option value="">Загрузка...</option>
              ) : targetCategories.length > 0 ? (
                targetCategories.map((category) => (
                  <option key={category} value={category}>
                    {CATEGORY_LABELS[category]}
                  </option>
                ))
              ) : (
                <option value="">Нет target-показателей</option>
              )}
            </select>
          </div>

          <div className="dashboard-filter-group">
            <div className="dashboard-filter-label">Модель прогноза</div>
            <select
              className="dashboard-select"
              value={selectedModelId}
              onChange={(event) => {
                setSelectedModelId(event.target.value)
              }}
              disabled={isModelsLoading || models.length === 0}
            >
              {isModelsLoading ? (
                <option value="">Загрузка моделей...</option>
              ) : models.length > 0 ? (
                models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {modelOptionLabel(model)}
                  </option>
                ))
              ) : (
                <option value="">Нет обученных моделей</option>
              )}
            </select>
          </div>

          <button
            type="button"
            className="btn btn-outline btn-small forecast-render-button"
            onClick={handleRenderTable}
            disabled={
              isForecastsLoading ||
              indicators.isLoading ||
              isModelsLoading ||
              targetCategories.length === 0 ||
              models.length === 0
            }
          >
            {isForecastsLoading ? 'Обновление...' : 'Обновить'}
          </button>
        </div>
      </section>

      <section className="forecast-summary-grid">
        <ForecastSummaryCard
          label="Модель"
          value={selectedModel ? modelTypeLabel(selectedModel.type) : '-'}
          caption={selectedModel?.name ?? 'Нет выбранной модели'}
        />
        <ForecastSummaryCard
          label="MAPE"
          value={selectedModel?.mape != null ? `${(selectedModel.mape * 100).toFixed(1)}%` : '-'}
          caption={selectedModel?.algorithm ?? 'Метрика качества'}
          accent={selectedModel?.mape != null ? 'positive' : undefined}
        />
        <ForecastSummaryCard
          label="Горизонт"
          value={forecastHorizonLabel(tableColumns)}
          caption={isForecastsLoading ? 'Обновляем таблицу' : `${tableRows.length} строк прогноза`}
        />
      </section>

      {isTableVisible && (
        <section className="glass-panel forecast-table-panel">
          <div className="forecast-table-actions">
            <span>{isForecastsLoading ? 'Обновление данных...' : 'Данные обновляются автоматически'}</span>
            <button
              type="button"
              className="btn btn-outline btn-small"
              onClick={handleExportExcel}
              disabled={tableRows.length === 0}
            >
              Экспорт в Excel
            </button>
          </div>
          <div className="forecast-table-wrapper">
            <table className="forecast-table">
              <thead>
                <tr>
                  <th className="forecast-sticky-col forecast-category-col">Категория</th>
                  <th className="forecast-sticky-col forecast-name-col">Показатель</th>
                  <th>Модель</th>
                  <th>Сценарий</th>
                  {tableColumns.map((column) => (
                    <th key={column.year}>{column.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableRows.map((row) => (
                  <tr key={row.id}>
                    <td className="forecast-sticky-col forecast-category-col">{row.categoryLabel}</td>
                    <td className="forecast-sticky-col forecast-name-col">
                      <div className="forecast-name">{row.indicatorName}</div>
                      <div className="forecast-unit">{row.unit}</div>
                    </td>
                    <td>{selectedModel ? modelOptionLabel(selectedModel) : '-'}</td>
                    <td>
                      <span className={`scenario-badge ${scenarioClass(row.scenario)}`}>
                        {row.scenario}
                      </span>
                    </td>
                    {tableColumns.map((column, index) => (
                      <ForecastValueCell
                        key={column.year}
                        value={row.valuesByYear.get(column.year)}
                        previousValue={previousColumnValue(row, tableColumns, index)}
                      />
                    ))}
                  </tr>
                ))}
                {tableRows.length === 0 && (
                  <tr>
                    <td colSpan={4 + tableColumns.length} className="forecast-empty-cell">
                      Нет прогнозов для выбранной target-категории
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
})

async function fetchLatestHistory(indicatorIds: number[]): Promise<Map<number, HistoryPoint[]>> {
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

function groupForecastPoints(points: ForecastPoint[]): Map<number, Map<string, Map<number, number>>> {
  const grouped = new Map<number, Map<string, Map<number, number>>>()

  for (const point of points) {
    let byScenario = grouped.get(point.indicator_id)
    if (!byScenario) {
      byScenario = new Map()
      grouped.set(point.indicator_id, byScenario)
    }

    let byYear = byScenario.get(point.scenario)
    if (!byYear) {
      byYear = new Map()
      byScenario.set(point.scenario, byYear)
    }

    byYear.set(point.year, point.value)
  }

  return grouped
}

function ForecastValueCell({
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

function previousColumnValue(
  row: ForecastTableRow,
  columns: ForecastTableColumn[],
  currentIndex: number,
): number | undefined {
  for (let index = currentIndex - 1; index >= 0; index -= 1) {
    const value = row.valuesByYear.get(columns[index].year)
    if (value != null) return value
  }
  return undefined
}

function formatValue(value: number | undefined): string {
  if (value == null) return '-'
  return value.toLocaleString('ru-RU', { maximumFractionDigits: 1 })
}

function scenarioRank(scenario: string): number {
  const index = SCENARIO_ORDER.indexOf(scenario)
  return index === -1 ? SCENARIO_ORDER.length : index
}

function modelTypeLabel(type: string): string {
  if (type === 'machine_learning' || type === 'ml') return 'ML'
  if (type === 'neural_network' || type === 'lstm') return 'LSTM'
  return type.toUpperCase()
}

function modelOptionLabel(model: ApiModel): string {
  const mape = model.mape != null ? `, MAPE ${(model.mape * 100).toFixed(1)}%` : ''
  return `${modelTypeLabel(model.type)} · ${model.name} · ${model.algorithm}${mape}`
}

function scenarioClass(scenario: string): string {
  if (scenario === 'Базовый') return 'base'
  if (scenario === 'Консервативный') return 'conservative'
  if (scenario === 'Оценка') return 'estimate'
  return 'neutral'
}

function forecastHorizonLabel(columns: ForecastTableColumn[]): string {
  const forecastYears = columns.filter((column) => !column.label.includes('факт')).length
  if (forecastYears === 0) return '-'
  return `${forecastYears} года`
}

function ForecastSummaryCard({
  label,
  value,
  caption,
  accent,
}: {
  label: string
  value: string
  caption: string
  accent?: 'positive'
}) {
  return (
    <div className="forecast-summary-card glass-panel">
      <div className="dashboard-kpi-label">{label}</div>
      <div className={`forecast-summary-value${accent ? ` ${accent}` : ''}`}>{value}</div>
      <div className="dashboard-kpi-caption">{caption}</div>
    </div>
  )
}

function exportForecastTable({
  rows,
  columns,
  modelLabel,
}: {
  rows: ForecastTableRow[]
  columns: ForecastTableColumn[]
  modelLabel: string
}) {
  const header = ['Категория', 'Показатель', 'Модель', 'Сценарий', ...columns.map((column) => column.label)]
  const tableRows = rows.map((row) => [
    row.categoryLabel,
    row.indicatorName,
    modelLabel,
    row.scenario,
    ...columns.map((column) => formatValue(row.valuesByYear.get(column.year))),
  ])

  const html = `
    <html>
      <head><meta charset="UTF-8" /></head>
      <body>
        <table border="1">
          <thead><tr>${header.map((cell) => `<th>${escapeHtml(cell)}</th>`).join('')}</tr></thead>
          <tbody>
            ${tableRows
              .map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`)
              .join('')}
          </tbody>
        </table>
      </body>
    </html>
  `

  const blob = new Blob([html], { type: 'application/vnd.ms-excel;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'forecast-summary.xls'
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}
