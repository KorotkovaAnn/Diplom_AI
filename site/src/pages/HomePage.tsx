import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { observer } from 'mobx-react-lite'
import { Box, Button, Typography } from '@mui/material'
import { useRootStore } from '../stores/rootStore.tsx'

export const HomePage = observer(function HomePage() {
  const { dashboards } = useRootStore()

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' })
  }, [])

  useEffect(() => {
    void dashboards.initializeDashboard()
  }, [dashboards])

  return (
    <div className="home-page home-page-with-bg">
      
      <HeroSection />
      <CapabilitiesAndMetrics />
      <ProcessAndQuickActions />
    </div>
  )
})

const SCENARIO_BASE_NAME = 'Базовый'
const SCENARIO_CONSERVATIVE_NAME = 'Консервативный'

const HeroSection = observer(function HeroSection() {
  const { dashboards } = useRootStore()
  const kpis = dashboards.kpis
  const modelInfo = dashboards.modelInfo
  const forecastYearCount = getForecastYearCount(dashboards.dashboardData?.forecasts)
  const targetTitle =
    dashboards.targetName && dashboards.targetName !== '—'
      ? dashboards.targetName
      : 'Инвестиции в основной капитал'
  const latestBaseForecast = getLatestForecastValue(
    dashboards.dashboardData?.forecasts,
    SCENARIO_BASE_NAME,
  )
  const latestConservativeForecast = getLatestForecastValue(
    dashboards.dashboardData?.forecasts,
    SCENARIO_CONSERVATIVE_NAME,
  )
  const estimateYear = dashboards.dashboardData?.forecasts['Оценка']?.[0]?.year

  return (
    <Box component="section" className="hero-section">
      <Box className="hero-panel glass-panel">
        <Box className="hero-tag">
          <span className="hero-tag-dot" />
          <span>B2G платформа</span>
          <span>•</span>
          <span>AI &amp; ML</span>
          <span>•</span>
          <span>Государственное управление</span>
        </Box>

        <Typography component="h1" className="hero-title">
          Цифровая платформа мониторинга и прогнозирования
          социально-экономических показателей региона
        </Typography>

        <Box className="hero-subtitle">
          <Typography component="p">
            Комплексный анализ данных с применением машинного обучения и нейронных
            сетей для точного прогнозирования показателей на 4 года вперёд.
          </Typography>
          <Typography component="p" className="hero-subtitle-accent">
            Интерпретируемые результаты для обоснованных управленческих решений.
          </Typography>
        </Box>

        <Box
          className="hero-audience-grid"
          sx={{
            fontSize: 12,
          }}
        >
          <Box className="hero-audience-card">
            <Typography variant="subtitle2" sx={{ fontSize: 13, mb: 0.5 }}>
              Для руководителей региона
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
              Сводка ключевых трендов и рисков в одном окне для подготовки решений.
            </Typography>
          </Box>
          <Box className="hero-audience-card">
            <Typography variant="subtitle2" sx={{ fontSize: 13, mb: 0.5 }}>
              Для аналитиков и экономистов
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
              Доступ к детализации показателей, методологии и качеству прогнозов.
            </Typography>
          </Box>
          <Box className="hero-audience-card">
            <Typography variant="subtitle2" sx={{ fontSize: 13, mb: 0.5 }}>
              Для инвесторов и партнёров
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
              Понятная картина инвестиционного климата и сценариев роста региона.
            </Typography>
          </Box>
        </Box>

        <Box
          className="hero-actions"
          sx={{
            mt: 1,
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'flex-start',
            gap: 1.25,
          }}
        >
          <Button
            component={Link}
            to="/dashboards"
            variant="contained"
            color="primary"
            className="btn btn-primary"
          >
            Перейти к дашбордам
          </Button>
        </Box>

        <Box className="hero-benefits">
          <span className="hero-benefit-item">
            Прогноз на 4 года со сценарным коридором
          </span>
          <span className="hero-benefit-item">SHAP-анализ факторов влияния</span>
        </Box>
      </Box>

      <Box component="aside" className="hero-visual glass-panel">
        <Box className="hero-grid" aria-hidden="true" />

        <Box className="hero-visual-inner">
          <Box className="hero-overlay-card">
            <Box className="hero-overlay-title">{targetTitle}</Box>
            <Box className="hero-overlay-subtitle">
              {dashboards.isLoading && !dashboards.dashboardData
                ? 'Загружаем данные из базы для главного показателя.'
                : 'Фактические данные, прогноз и метрики модели подтягиваются из базы.'}
            </Box>

            <Box className="hero-metric-row">
              <Box>
                <Box className="hero-metric-label">
                  {kpis.currentYear ? `Факт за ${kpis.currentYear}` : 'Текущее значение'}
                </Box>
                <Box className="hero-metric-value">
                  {formatIndicatorValue(kpis.currentValue, dashboards.targetUnit)}
                </Box>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Box className="hero-metric-label">Год к году</Box>
                <Box
                  className={`hero-metric-change ${
                    kpis.yoyChange == null ? '' : kpis.yoyChange >= 0 ? 'positive' : 'negative'
                  }`}
                >
                  {formatPercent(kpis.yoyChange)}
                </Box>
              </Box>
            </Box>

            <Box className="hero-metric-row">
              <Box>
                <Box className="hero-metric-label">Горизонт прогноза</Box>
                <Box className="hero-metric-value">
                  {forecastYearCount > 0 ? `+${forecastYearCount} года` : '—'}
                </Box>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Box className="hero-metric-label">Точность моделей (MAPE)</Box>
                <Box className="hero-metric-change positive">
                  {modelInfo?.mape != null ? formatPlainPercent(modelInfo.mape * 100) : '—'}
                </Box>
              </Box>
            </Box>
          </Box>

          <Box className="hero-overlay-card">
            <Box className="hero-overlay-title">{targetTitle}</Box>
            <Box className="hero-overlay-subtitle">
              Сценарный анализ: оценка первого года, базовый и консервативный варианты
              динамики инвестиций.
            </Box>

            <Box className="hero-metric-grid">
              <Box className="hero-metric-cell">
                <Box className="hero-metric-label">Базовый сценарий</Box>
                <Box className="hero-metric-value">
                  {formatIndicatorValue(latestBaseForecast?.value ?? null, dashboards.targetUnit)}
                </Box>
              </Box>
              <Box className="hero-metric-cell hero-metric-cell-accent">
                <Box className="hero-metric-label">Оценка первого года</Box>
                <Box className="hero-metric-change">{estimateYear ?? '—'}</Box>
              </Box>

              <Box className="hero-metric-cell">
                <Box className="hero-metric-label">Консервативный сценарий</Box>
                <Box className="hero-metric-value">
                  {formatIndicatorValue(
                    latestConservativeForecast?.value ?? null,
                    dashboards.targetUnit,
                  )}
                </Box>
              </Box>
              <Box className="hero-metric-cell hero-metric-cell-accent">
                <Box className="hero-metric-label">Год прогноза</Box>
                <Box className="hero-metric-change">
                  {latestBaseForecast?.year ?? latestConservativeForecast?.year ?? '—'}
                </Box>
              </Box>
            </Box>
          </Box>
        </Box>
      </Box>
    </Box>
  )
})

type ForecastMap = Record<string, { year: number; value: number }[]> | undefined

function getForecastYearCount(forecasts: ForecastMap) {
  if (!forecasts) return 0

  const years = new Set<number>()
  Object.values(forecasts).forEach((points) => {
    points.forEach((point) => years.add(point.year))
  })

  return years.size
}

function getLatestForecastValue(forecasts: ForecastMap, scenarioName: string) {
  const points = forecasts?.[scenarioName] ?? []
  if (!points.length) return null

  return [...points].sort((a, b) => b.year - a.year)[0]
}

function formatIndicatorValue(value: number | null | undefined, unit: string) {
  if (value == null) return '—'

  if (unit.toLowerCase().includes('млн') && unit.toLowerCase().includes('руб')) {
    return `${(value / 1000).toLocaleString('ru-RU', {
      maximumFractionDigits: 1,
    })} млрд ₽`
  }

  return `${value.toLocaleString('ru-RU', { maximumFractionDigits: 0 })}${unit ? ` ${unit}` : ''}`
}

function formatPercent(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return '—'

  return `${value >= 0 ? '+' : ''}${value.toLocaleString('ru-RU', {
    maximumFractionDigits: 1,
  })}%`
}

function formatPlainPercent(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return '—'

  return `${value.toLocaleString('ru-RU', {
    maximumFractionDigits: 1,
  })}%`
}

function CapabilitiesAndMetrics() {
  const [stats, setStats] = useState<HomeStats | null>(null)

  useEffect(() => {
    let ignore = false

    fetch('/api/stats')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json() as Promise<HomeStats>
      })
      .then((data) => {
        if (!ignore) setStats(data)
      })
      .catch(() => {
        if (!ignore) setStats(null)
      })

    return () => {
      ignore = true
    }
  }, [])

  return (
    <section className="home-grid">
      <div className="glass-panel capabilities-metrics-panel">
        <div style={{ padding: '14px 18px 14px' }}>
          <h2 className="home-section-title">Ключевые возможности платформы</h2>
          <p className="home-section-subtitle">
            Модерновый стек технологий, прозрачная аналитика и интерпретируемые прогнозы
            для принятия управленческих решений.
          </p>
          <div className="capabilities-grid">
            <div className="capability-card">
              <div
                className="capability-icon"
                style={{
                  backgroundImage: 'linear-gradient(145deg, #3b82f6, #06b6d4)',
                }}
              >
                📊
              </div>
              <div className="capability-title">Анализ динамики</div>
              <div className="capability-text">
                Пространственный анализ и социально-экономический мониторинг региона:
                динамика, различия территорий и ключевые точки изменения.
              </div>
            </div>

            <div className="capability-card">
              <div
                className="capability-icon"
                style={{
                  backgroundImage: 'linear-gradient(145deg, #8b5cf6, #ec4899)',
                }}
              >
                📈
              </div>
              <div className="capability-title">Прогноз на 4 года</div>
              <div className="capability-text">
                Сценарные прогнозы на базе ансамблей моделей, нейронных сетей и
                классических эконометрических подходов.
              </div>
            </div>

            <div className="capability-card">
              <div
                className="capability-icon"
                style={{
                  backgroundImage: 'linear-gradient(145deg, #f97316, #facc15)',
                }}
              >
                🧠
              </div>
              <div className="capability-title">Интерпретация SHAP</div>
              <div className="capability-text">
                Визуальный SHAP-анализ, показывающий вклад каждого фактора в итоговый
                прогноз и чувствительность показателей.
              </div>
            </div>

            <div className="capability-card">
              <div
                className="capability-icon"
                style={{
                  backgroundImage: 'linear-gradient(145deg, #10b981, #06b6d4)',
                }}
              >
                🗺️
              </div>
              <div className="capability-title">Поддержка решений</div>
              <div className="capability-text">
                Система сценарного моделирования для оценки вариантов развития и
                подготовки обоснованных управленческих решений.
              </div>
            </div>
          </div>
        </div>
      </div>

      <aside className="metrics-panel glass-panel">
        <h2 className="home-section-title">Ключевые метрики системы</h2>
        <p className="home-section-subtitle">
          Сфокусированные показатели готовы к презентации руководству в один клик.
        </p>
        <div className="metrics-grid">
          <div className="metric-item">
            <div className="metric-label">Показателей в системе</div>
            <div className="metric-value">{stats?.indicator_count ?? '—'}</div>
            <div className="metric-caption">
              целевые и факторные показатели, используемые моделями
            </div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Горизонт прогноза</div>
            <div className="metric-value">
              {stats?.forecast_horizon ? `+${stats.forecast_horizon} года` : '+4 года'}
            </div>
            <div className="metric-caption">оценка, базовый и консервативный сценарии</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Частота обновления</div>
            <div className="metric-value">год</div>
            <div className="metric-caption">обновление после публикации годовой статистики</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Прогнозные показатели</div>
            <div className="metric-value">{stats?.target_indicator_count ?? '—'}</div>
            <div className="metric-caption">показатели с ролью target в прогнозных запусках</div>
          </div>
        </div>
      </aside>
    </section>
  )
}

interface HomeStats {
  indicator_count: number
  target_indicator_count: number
  forecast_horizon: number
  best_mape: number | null
  model_count: number
}

function ProcessAndQuickActions() {
  return (
    <section className="home-grid">
      <div className="timeline-section glass-panel">
        <h2 className="home-section-title">Как это работает</h2>
        <p className="home-section-subtitle">
          Пять шагов от ручной загрузки данных до интерактивной визуализации прогноза.
        </p>
        <div className="timeline-row">
          <TimelineStep
            number={1}
            color="linear-gradient(145deg, #3b82f6, #06b6d4)"
            title="Сбор данных"
            text="Пользователь вручную загружает исходные данные в систему для дальнейшей проверки и расчётов."
          />
          <TimelineStep
            number={2}
            color="linear-gradient(145deg, #8b5cf6, #6366f1)"
            title="Обработка и анализ"
            text="Очистка, валидация, обогащение и формирование витрин данных для прогнозных моделей."
          />
          <TimelineStep
            number={3}
            color="linear-gradient(145deg, #ec4899, #f97316)"
            title="Машинное обучение"
            text="Оптимизация гиперпараметров ML-моделей и LSTM для выбора наиболее точной конфигурации."
          />
          <TimelineStep
            number={4}
            color="linear-gradient(145deg, #f59e0b, #fbbf24)"
            title="Интерпретация"
            text="SHAP-анализ, сценарный коридор и сравнение вариантов для прозрачности решений."
          />
          <TimelineStep
            number={5}
            color="linear-gradient(145deg, #10b981, #06b6d4)"
            title="Визуализация"
            text="Интерактивные дашборды показывают фактическую динамику, прогнозные сценарии и вклад факторов."
          />
        </div>
      </div>

      <aside className="quick-actions glass-panel">
        <div style={{ gridColumn: '1 / -1', marginBottom: 4 }}>
          <h2 className="home-section-title">Быстрые действия</h2>
          <p className="home-section-subtitle">
            Один клик до ключевых разделов платформы.
          </p>
        </div>
        <QuickAction
          to="/dashboards"
          color="linear-gradient(145deg, #22c55e, #16a34a)"
          icon="📊"
          title="Дашборды"
          subtitle="Панель мониторинга и визуализации"
        />
        <QuickAction
          to="/forecasts"
          color="linear-gradient(145deg, #0ea5e9, #2563eb)"
          icon="📈"
          title="Прогнозы"
          subtitle="Сводка по всем target-показателям"
        />
        <QuickAction
          to="/indicators"
          color="linear-gradient(145deg, #a855f7, #ec4899)"
          icon="🧮"
          title="Показатели"
          subtitle="Сопоставление прогнозных и факторных данных"
        />
        <QuickAction
          to="/data-upload"
          color="linear-gradient(145deg, #f97316, #facc15)"
          icon="📁"
          title="Загрузка данных"
          subtitle="Ручная загрузка исходных наборов"
        />
      </aside>
    </section>
  )
}

interface TimelineStepProps {
  number: number
  color: string
  title: string
  text: string
}

function TimelineStep({ number, color, title, text }: TimelineStepProps) {
  return (
    <div className="timeline-step">
      <div className="timeline-number" style={{ backgroundImage: color }}>
        {number}
      </div>
      <div className="timeline-title">{title}</div>
      <div className="timeline-text">{text}</div>
      {number < 5 ? <div className="timeline-connector" /> : null}
    </div>
  )
}

interface QuickActionProps {
  to: string
  color: string
  icon: string
  title: string
  subtitle: string
}

function QuickAction({ to, color, icon, title, subtitle }: QuickActionProps) {
  return (
    <Link to={to} className="quick-action-button">
      <div className="quick-action-icon" style={{ backgroundImage: color }}>
        {icon}
      </div>
      <div>
        <div className="quick-action-title">{title}</div>
        <div className="quick-action-subtitle">{subtitle}</div>
      </div>
    </Link>
  )
}
