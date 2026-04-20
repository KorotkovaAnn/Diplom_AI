import { observer } from 'mobx-react-lite'
import { useRootStore } from '../stores/rootStore.tsx'

interface ForecastRow {
  id: string
  name: string
  unit: string
  current: number
  y1: number
  y2: number
  y3: number
  y4: number
}

export const ForecastsPage = observer(function ForecastsPage() {
  const { indicators } = useRootStore()

  const rows: ForecastRow[] = indicators.indicators.map((ind) => {
    const base = ind.currentValue
    const growth = 1 + ind.yoyChangePercent / 100
    const g1 = base * growth
    const g2 = g1 * growth
    const g3 = g2 * growth
    const g4 = g3 * growth
    return {
      id: ind.id,
      name: ind.name,
      unit: ind.unit,
      current: base,
      y1: g1,
      y2: g2,
      y3: g3,
      y4: g4,
    }
  })

  return (
    <div>
      <h1 className="page-title">Сводка прогнозов</h1>
      <p className="page-subtitle">
        Быстрый обзор прогнозов по всем показателям на горизонте 4 года с возможностью
        сравнения динамики.
      </p>

      <section className="glass-panel" style={{ padding: '12px 18px 10px', marginBottom: 16 }}>
        <div className="forecast-toolbar">
          <div className="dashboard-filter-group">
            <div className="dashboard-filter-label">Сценарий</div>
            <div className="dashboard-segmented">
              <button type="button" className="dashboard-segment dashboard-segment-active">
                Базовый
              </button>
              <button type="button" className="dashboard-segment">
                Оптимистичный
              </button>
              <button type="button" className="dashboard-segment">
                Пессимистичный
              </button>
            </div>
          </div>
          <div className="dashboard-filter-group">
            <div className="dashboard-filter-label">Категория</div>
            <select className="dashboard-select">
              <option>Все категории</option>
              <option>Экономика</option>
              <option>Инвестиции</option>
              <option>Демография</option>
            </select>
          </div>
          <button type="button" className="btn btn-outline btn-small">
            Экспорт в Excel
          </button>
        </div>
      </section>

      <section className="glass-panel" style={{ padding: '0 0 8px' }}>
        <div className="forecast-table-wrapper">
          <table className="forecast-table">
            <thead>
              <tr>
                <th>Показатель</th>
                <th>Текущее значение</th>
                <th>Прогноз +1 год</th>
                <th>+2 года</th>
                <th>+3 года</th>
                <th>+4 года</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <td>
                    <div className="forecast-name">{row.name}</div>
                    <div className="forecast-unit">{row.unit}</div>
                  </td>
                  <td>{formatValue(row.current)}</td>
                  <ForecastCell base={row.current} value={row.y1} />
                  <ForecastCell base={row.current} value={row.y2} />
                  <ForecastCell base={row.current} value={row.y3} />
                  <ForecastCell base={row.current} value={row.y4} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
})

function formatValue(v: number): string {
  return v.toLocaleString('ru-RU', { maximumFractionDigits: 1 })
}

function ForecastCell({ base, value }: { base: number; value: number }) {
  const change = ((value - base) / base) * 100
  let cls = 'forecast-cell-change neutral'
  if (change > 2) cls = 'forecast-cell-change positive'
  else if (change < -2) cls = 'forecast-cell-change negative'

  return (
    <td className={cls}>
      <div>{formatValue(value)}</div>
      <span>{change >= 0 ? '▲' : '▼'} {Math.abs(change).toFixed(1)}%</span>
    </td>
  )
}

