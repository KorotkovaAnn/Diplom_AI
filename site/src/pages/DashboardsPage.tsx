import { useEffect } from 'react'
import { observer } from 'mobx-react-lite'
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useRootStore, type TimeSeriesPoint } from '../stores/rootStore.tsx'

export const DashboardsPage = observer(function DashboardsPage() {
  const { dashboards, indicators } = useRootStore()

  useEffect(() => {
    dashboards.fetchDashboard()
    indicators.fetchIndicators()
  }, [dashboards, indicators])

  const series = dashboards.timeSeries
  const shap = dashboards.shapContributions
  const kpis = dashboards.kpis

  const forecastCards = series.filter((p) => p.forecast != null)
  const positiveFactors = shap.filter((f) => f.contribution > 0).slice(0, 3)
  const negativeFactors = shap.filter((f) => f.contribution < 0).slice(0, 3)

  return (
    <div>
      <h1 className="page-title">Аналитические дашборды</h1>
      <p className="page-subtitle">
        Выберите показатель для анализа — прогноз, доверительные интервалы и SHAP-интерпретация
        обновятся автоматически.
      </p>

      {/* Строка ошибки */}
      {dashboards.error && (
        <div
          style={{
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: 8,
            padding: '12px 16px',
            marginBottom: 18,
            color: '#dc2626',
            fontSize: 14,
          }}
        >
          {dashboards.error}
        </div>
      )}

      {/* Панель фильтров */}
      <section className="glass-panel" style={{ padding: '14px 18px 16px', marginBottom: 18 }}>
        <div className="dashboard-filters">
          <div className="dashboard-filter-group">
            <div className="dashboard-filter-label">Показатель</div>
            <select
              className="dashboard-select"
              value={dashboards.selectedIndicatorId}
              onChange={(e) => dashboards.setSelectedIndicator(e.target.value)}
            >
              {indicators.isLoading ? (
                <option disabled>Загрузка…</option>
              ) : (
                groupByCategory(indicators.indicators).map((group) => (
                  <optgroup key={group.category} label={group.label}>
                    {group.items.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </optgroup>
                ))
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
                  className={segmentClass(dashboards.timeRange === range)}
                  onClick={() => dashboards.setTimeRange(range)}
                >
                  {range === 'all' ? 'Весь ряд' : range === 'last10' ? '10 лет' : '5 лет'}
                </button>
              ))}
            </div>
          </div>

          <div className="dashboard-filter-group">
            <div className="dashboard-filter-label">Сценарий прогноза</div>
            <div className="dashboard-segmented">
              {(['base', 'optimistic', 'pessimistic'] as const).map((sc) => (
                <button
                  key={sc}
                  type="button"
                  className={segmentClass(dashboards.scenarioType === sc)}
                  onClick={() => dashboards.setScenarioType(sc)}
                >
                  {sc === 'base' ? 'Базовый' : sc === 'optimistic' ? 'Оптимистичный' : 'Пессимистичный'}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Скелетон при первой загрузке */}
      {dashboards.isLoading && !dashboards.dashboardData && (
        <div style={{ textAlign: 'center', padding: '60px', color: '#6b7280', fontSize: 15 }}>
          Загрузка данных из базы…
        </div>
      )}

      {/* Основной контент — показываем сразу после загрузки */}
      {(dashboards.dashboardData || !dashboards.isLoading) && (
        <>
          <section className="dashboard-grid">
            {/* Карточки прогнозов */}
            <aside className="dashboard-forecast-cards glass-panel">
              <h2 className="home-section-title">Прогноз по годам</h2>
              <p className="home-section-subtitle">
                Прогноз на горизонт вперёд с динамикой относительно последнего фактического значения.
              </p>
              <div className="forecast-cards-grid">
                {forecastCards.map((p) => {
                  const base = kpis.currentValue
                  const change =
                    base != null && p.forecast != null ? ((p.forecast - base) / base) * 100 : null

                  const changeClass =
                    change == null ? 'neutral' : change > 0.5 ? 'positive' : change < -0.5 ? 'negative' : 'neutral'

                  return (
                    <div key={p.year} className="forecast-card">
                      <div className="forecast-card-year">{p.year}</div>
                      <div className="forecast-card-value">
                        {p.forecast?.toLocaleString('ru-RU', { maximumFractionDigits: 0 })}
                      </div>
                      <div className={`forecast-card-change ${changeClass}`}>
                        {change != null ? (
                          <>
                            <span>{change >= 0 ? '▲' : '▼'}</span>
                            <span>{Math.abs(change).toFixed(1)}%</span>
                          </>
                        ) : (
                          <span>—</span>
                        )}
                      </div>
                    </div>
                  )
                })}
                {forecastCards.length === 0 && !dashboards.isLoading && (
                  <div style={{ color: '#9ca3af', fontSize: 13 }}>Нет данных прогноза</div>
                )}
              </div>

              <div className="dashboard-kpis">
                <div className="dashboard-kpi">
                  <div className="dashboard-kpi-label">
                    Последний фактический год{kpis.currentYear ? ` (${kpis.currentYear})` : ''}
                  </div>
                  <div className="dashboard-kpi-value">
                    {kpis.currentValue != null
                      ? kpis.currentValue.toLocaleString('ru-RU', { maximumFractionDigits: 0 })
                      : '—'}
                  </div>
                  <div className="dashboard-kpi-caption">
                    {dashboards.targetName} · {dashboards.targetUnit}
                  </div>
                </div>

                <div className="dashboard-kpi">
                  <div className="dashboard-kpi-label">Изменение к прошлому году</div>
                  <div
                    className={`dashboard-kpi-value dashboard-kpi-value-small ${
                      kpis.yoyChange == null ? '' : kpis.yoyChange >= 0 ? 'positive' : 'negative'
                    }`}
                  >
                    {kpis.yoyChange != null
                      ? `${kpis.yoyChange >= 0 ? '+' : ''}${kpis.yoyChange.toFixed(1)}%`
                      : '—'}
                  </div>
                  <div className="dashboard-kpi-caption">по данным последнего периода</div>
                </div>

                <div className="dashboard-kpi">
                  <div className="dashboard-kpi-label">Среднегодовой темп роста (CAGR)</div>
                  <div className="dashboard-kpi-value dashboard-kpi-value-small">
                    {kpis.cagr != null ? `+${kpis.cagr.toFixed(1)}% в год` : '—'}
                  </div>
                  <div className="dashboard-kpi-caption">за весь исторический период</div>
                </div>

                {dashboards.modelInfo && (
                  <div className="dashboard-kpi">
                    <div className="dashboard-kpi-label">Точность модели (MAPE)</div>
                    <div className="dashboard-kpi-value dashboard-kpi-value-small positive">
                      {dashboards.modelInfo.mape != null
                        ? `${dashboards.modelInfo.mape.toFixed(1)}%`
                        : '—'}
                    </div>
                    <div className="dashboard-kpi-caption">
                      {dashboards.modelInfo.name}
                    </div>
                  </div>
                )}
              </div>
            </aside>

            {/* Основной график */}
            <section className="dashboard-chart-panel glass-panel">
              <h2 className="home-section-title">Динамика показателя и прогноз</h2>
              <p className="home-section-subtitle">
                Синяя линия — фактические значения, фиолетовая — прогноз с доверительным интервалом.
              </p>
              <div className="dashboard-main-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={applyTimeRange(series, dashboards.timeRange)}>
                    <defs>
                      <linearGradient id="forecastArea" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="forecastBand" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor="#c4b5fd" stopOpacity={0.4} />
                        <stop offset="100%" stopColor="#ddd6fe" stopOpacity={0.1} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis
                      tickFormatter={(v: number) =>
                        v >= 1000 ? `${(v / 1000).toFixed(0)}\u00a0тыс.` : String(v)
                      }
                    />
                    <Tooltip
                      // recharts 3 Formatter type intersection requires cast
                      formatter={
                        ((value: number) =>
                          value.toLocaleString('ru-RU', {
                            maximumFractionDigits: 0,
                          })) as never
                      }
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="upper"
                      stroke="none"
                      fill="url(#forecastBand)"
                      yAxisId={0}
                    />
                    <Area
                      type="monotone"
                      dataKey="lower"
                      stroke="none"
                      fill="#ffffff"
                      fillOpacity={1}
                      yAxisId={0}
                    />
                    <Area
                      type="monotone"
                      dataKey="forecast"
                      stroke="none"
                      fill="url(#forecastArea)"
                      isAnimationActive
                      yAxisId={0}
                    />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Факт"
                      stroke="#2563eb"
                      strokeWidth={2.4}
                      dot={{ r: 3 }}
                      activeDot={{ r: 4.5 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="forecast"
                      name="Прогноз"
                      stroke="#8b5cf6"
                      strokeDasharray="5 5"
                      strokeWidth={2.2}
                      dot={{ r: 3 }}
                      activeDot={{ r: 4.5 }}
                    />
                    {kpis.currentYear && (
                      <ReferenceLine
                        x={kpis.currentYear}
                        stroke="#9ca3af"
                        strokeDasharray="3 3"
                        label={{
                          value: 'Последний факт',
                          position: 'top',
                          fontSize: 11,
                          fill: '#6b7280',
                        }}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="dashboard-legend-row">
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: '#2563eb' }} />
                  Фактические значения
                </span>
                <span className="legend-item">
                  <span className="legend-line legend-line-dashed" style={{ borderColor: '#8b5cf6' }} />
                  Прогноз
                </span>
                <span className="legend-item">
                  <span className="legend-band" style={{ background: 'rgba(167,139,250,0.25)' }} />
                  Доверительный интервал
                </span>
              </div>
            </section>
          </section>

          {/* SHAP-анализ */}
          <section className="dashboard-grid" style={{ marginTop: 18 }}>
            <section className="shap-panel glass-panel">
              <h2 className="home-section-title">SHAP-анализ вклада факторов</h2>
              <p className="home-section-subtitle">
                Положительный вклад (зелёный) увеличивает прогнозируемое значение,
                отрицательный (красный) — снижает. Значения нормированы по сумме вкладов.
              </p>
              {shap.length > 0 ? (
                <div className="shap-chart-wrapper">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={shap}
                      layout="vertical"
                      margin={{ left: 16, right: 40, top: 8, bottom: 8 }}
                    >
                      <defs>
                        <linearGradient id="shapPositive" x1="0" x2="1" y1="0" y2="0">
                          <stop offset="0%" stopColor="#bbf7d0" />
                          <stop offset="100%" stopColor="#22c55e" />
                        </linearGradient>
                        <linearGradient id="shapNegative" x1="0" x2="1" y1="0" y2="0">
                          <stop offset="0%" stopColor="#fecaca" />
                          <stop offset="100%" stopColor="#ef4444" />
                        </linearGradient>
                      </defs>
                      <CartesianGrid horizontal={false} stroke="#e5e7eb" strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                      />
                      <YAxis
                        type="category"
                        dataKey="name"
                        tick={{ fontSize: 11 }}
                        width={200}
                      />
                      <Tooltip
                        formatter={(value: number | undefined) => [
                          value != null ? `${(value * 100).toFixed(1)}%` : '—',
                          'Вклад в прогноз',
                        ]}
                      />
                      <Bar
                        dataKey="contribution"
                        barSize={18}
                        radius={9}
                        label={{
                          position: 'right',
                          formatter: (value: unknown) =>
                            typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '',
                          fontSize: 11,
                        }}
                      >
                        {shap.map((entry) => (
                          <Cell
                            key={entry.id}
                            fill={entry.contribution >= 0 ? 'url(#shapPositive)' : 'url(#shapNegative)'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div style={{ padding: '24px 0', color: '#9ca3af', fontSize: 13 }}>
                  {dashboards.isLoading ? 'Загрузка SHAP-данных…' : 'Нет данных SHAP для выбранного сценария'}
                </div>
              )}
              <p className="shap-caption">
                Факторы отсортированы по абсолютной величине влияния — самые важные находятся
                в верхней части диаграммы.
              </p>
            </section>

            <aside className="shap-insights glass-panel">
              <h2 className="home-section-title">Ключевые инсайты</h2>
              <p className="home-section-subtitle">
                Автоматически выделенные драйверы роста и факторы снижения по текущему показателю.
              </p>
              <div className="shap-insights-grid">
                <div>
                  <div className="shap-insight-title">Топ-3 фактора роста</div>
                  <ul className="shap-insight-list">
                    {positiveFactors.length > 0 ? (
                      positiveFactors.map((f) => (
                        <li key={f.id}>
                          <span className="shap-factor-name">{f.name}</span>
                          <span className="shap-factor-value">
                            +{(f.contribution * 100).toFixed(1)}%
                          </span>
                        </li>
                      ))
                    ) : (
                      <li style={{ color: '#9ca3af' }}>Нет данных</li>
                    )}
                  </ul>
                </div>
                <div>
                  <div className="shap-insight-title">Топ-3 фактора снижения</div>
                  <ul className="shap-insight-list">
                    {negativeFactors.length > 0 ? (
                      negativeFactors.map((f) => (
                        <li key={f.id}>
                          <span className="shap-factor-name">{f.name}</span>
                          <span className="shap-factor-value shap-factor-negative">
                            {(f.contribution * 100).toFixed(1)}%
                          </span>
                        </li>
                      ))
                    ) : (
                      <li style={{ color: '#9ca3af' }}>Нет данных</li>
                    )}
                  </ul>
                </div>
              </div>
            </aside>
          </section>

          {/* Расширенная аналитика */}
          <section
            className="glass-panel"
            style={{ marginTop: 18, padding: '14px 18px 16px' }}
          >
            <h2 className="home-section-title">Расширенная аналитика</h2>
            <p className="home-section-subtitle">
              Дополнительные инструменты анализа: сценарии, корреляции, декомпозиция временного ряда
              и метрики качества моделей.
            </p>
            <div className="advanced-grid">
              <div className="advanced-block">
                <div className="advanced-title">Сравнение сценариев</div>
                <p className="advanced-text">
                  Таблица с базовым, оптимистичным и пессимистичным сценариями прогноза с
                  визуальным сравнением всех трёх линий.
                </p>
              </div>
              <div className="advanced-block">
                <div className="advanced-title">Корреляционная матрица</div>
                <p className="advanced-text">
                  Тепловая карта корреляций между факторами позволяет выявить мультиколлинеарность и
                  скорректировать спецификацию моделей.
                </p>
              </div>
              <div className="advanced-block">
                <div className="advanced-title">Декомпозиция временного ряда</div>
                <p className="advanced-text">
                  Разложение на тренд, сезонность и остатки помогает объяснить структуру показателя и
                  обосновать модельные допущения.
                </p>
              </div>
              <div className="advanced-block">
                <div className="advanced-title">Метрики качества моделей</div>
                <p className="advanced-text">
                  {dashboards.modelInfo
                    ? `Лучшая модель: ${dashboards.modelInfo.name} — MAE: ${dashboards.modelInfo.mae?.toLocaleString('ru-RU', { maximumFractionDigits: 0 })}, RMSE: ${dashboards.modelInfo.rmse?.toLocaleString('ru-RU', { maximumFractionDigits: 0 })}, MAPE: ${dashboards.modelInfo.mape?.toFixed(1)}%.`
                    : 'Набор ключевых метрик (RMSE, MAE, MAPE) показывает точность прогнозов и устойчивость моделей на исторических данных.'}
                </p>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  )
})

// ---------------------------------------------------------------------------
// Вспомогательные функции
// ---------------------------------------------------------------------------

function segmentClass(active: boolean): string {
  return active ? 'dashboard-segment dashboard-segment-active' : 'dashboard-segment'
}

function applyTimeRange(
  series: TimeSeriesPoint[],
  range: 'all' | 'last5' | 'last10',
): TimeSeriesPoint[] {
  if (range === 'all') return series
  const years = series.map((p) => p.year)
  const maxYear = Math.max(...years)
  const minYear = range === 'last5' ? maxYear - 4 : maxYear - 9
  return series.filter((p) => p.year >= minYear)
}

function groupByCategory(indicators: { id: string; name: string; category: string }[]) {
  const labels: Record<string, string> = {
    economy: 'Экономика',
    investment: 'Инвестиции и основные фонды',
    construction: 'Строительство',
    demography: 'Демография',
    labour: 'Рынок труда',
    social: 'Уровень жизни и социальная сфера',
  }

  const groups: { category: string; label: string; items: typeof indicators }[] = []

  for (const ind of indicators) {
    let group = groups.find((g) => g.category === ind.category)
    if (!group) {
      group = {
        category: ind.category,
        label: labels[ind.category] ?? ind.category,
        items: [],
      }
      groups.push(group)
    }
    group.items.push(ind)
  }

  return groups
}

