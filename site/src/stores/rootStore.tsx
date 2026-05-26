/* eslint-disable react-refresh/only-export-components */
import { createContext, type ReactNode, useContext } from 'react'
import { makeAutoObservable, runInAction } from 'mobx'

export type LanguageCode = 'ru' | 'en'
export type ScenarioType = 'base' | 'conservative'
export type IndicatorCategory =
  | 'economy'
  | 'investment'
  | 'construction'
  | 'demography'
  | 'labour'
  | 'social'

export interface SparkPoint {
  year: number
  value: number
}

export interface Indicator {
  id: string
  name: string
  category: IndicatorCategory
  description: string
  source: string
  updateFrequency: string
  currentValue: number
  unit: string
  yoyChangePercent: number
  sparkline: SparkPoint[]
}

export interface TimeSeriesPoint {
  year: number
  actual?: number
  forecast?: number
  lower?: number
  upper?: number
  scenario?: number
}

export interface ShapFactorContribution {
  id: string
  name: string
  contribution: number
}

// ---------------------------------------------------------------------------
// API-типы
// ---------------------------------------------------------------------------

interface ApiIndicator {
  id: number
  sphere: string
  name: string
  unit: string
}

interface ApiHistoryPoint {
  year: number
  value: number
}

interface ApiShapItem {
  indicator_id: number
  indicator_name: string
  contribution: number
  direction: string
  rank: number
}

export interface ApiModel {
  id: number
  run_id: number
  name: string
  type: string
  algorithm: string
  status: string
  mae: number | null
  rmse: number | null
  mape: number | null
}

interface ApiDashboard {
  target_indicator: ApiIndicator | null
  history: ApiHistoryPoint[]
  forecasts: Record<string, ApiHistoryPoint[]>
  shap: Record<string, ApiShapItem[]>
  model: ApiModel | null
}

// ---------------------------------------------------------------------------
// Вспомогательные константы
// ---------------------------------------------------------------------------

const SPHERE_MAP: Record<string, IndicatorCategory> = {
  'Экономика': 'economy',
  'Основные фонды': 'investment',
  'Внешние макроэкономические факторы': 'economy',
  'Внешняя торговля': 'economy',
  'Финансы': 'investment',
  'Уровень жизни населения': 'social',
  'Цены и тарифы': 'economy',
  'Население': 'demography',
  'Торговля и услуги населению': 'social',
  'Организации': 'economy',
}

const SCENARIO_RU: Record<ScenarioType, string> = {
  base: 'Базовый',
  conservative: 'Консервативный',
}

// ---------------------------------------------------------------------------
// UiStore
// ---------------------------------------------------------------------------

class UiStore {
  language: LanguageCode = 'ru'
  notifications: {
    id: string
    type: 'info' | 'warning' | 'success'
    title: string
    time: string
  }[] = [
    { id: 'n1', type: 'info',    title: 'Данные загружены из базы indicators.db', time: 'сегодня' },
    { id: 'n2', type: 'warning', title: 'Существенное отклонение прогноза по инвестициям', time: 'вчера' },
    { id: 'n3', type: 'success', title: 'Добавлена нейросетевая модель LSTM', time: '3 дня назад' },
  ]

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true })
  }

  setLanguage(language: LanguageCode) {
    this.language = language
  }
}

// ---------------------------------------------------------------------------
// IndicatorStore
// ---------------------------------------------------------------------------

class IndicatorStore {
  indicators: Indicator[] = []
  favorites = new Set<string>()
  searchQuery = ''
  activeCategory: IndicatorCategory | 'all' = 'all'
  isLoading = false

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true })
  }

  async fetchIndicators(role?: 'target' | 'feature') {
    this.isLoading = true
    try {
      const query = role ? `?role=${role}` : ''
      const res = await fetch(`/api/indicators${query}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: ApiIndicator[] = await res.json()
      runInAction(() => {
        this.indicators = data.map((ind) => ({
          id: String(ind.id),
          name: ind.name,
          category: SPHERE_MAP[ind.sphere] ?? 'economy',
          description: '',
          source: 'Росстат',
          updateFrequency: 'ежегодно',
          currentValue: 0,
          unit: ind.unit,
          yoyChangePercent: 0,
          sparkline: [],
        }))
        this.isLoading = false
      })
    } catch {
      runInAction(() => { this.isLoading = false })
    }
  }

  toggleFavorite(id: string) {
    if (this.favorites.has(id)) {
      this.favorites.delete(id)
    } else {
      this.favorites.add(id)
    }
  }

  setSearchQuery(query: string) {
    this.searchQuery = query
  }

  setActiveCategory(category: IndicatorCategory | 'all') {
    this.activeCategory = category
  }

  get filteredIndicators(): Indicator[] {
    const byCategory =
      this.activeCategory === 'all'
        ? this.indicators
        : this.indicators.filter((ind) => ind.category === this.activeCategory)

    if (!this.searchQuery.trim()) return byCategory

    const query = this.searchQuery.toLowerCase()
    return byCategory.filter(
      (ind) =>
        ind.name.toLowerCase().includes(query) ||
        ind.description.toLowerCase().includes(query),
    )
  }
}

// ---------------------------------------------------------------------------
// DashboardStore
// ---------------------------------------------------------------------------

class DashboardStore {
  indicatorStore: IndicatorStore
  selectedIndicatorId = '15'
  scenarioType: ScenarioType = 'base'
  timeRange: 'all' | 'last5' | 'last10' = 'last10'
  selectedModelId: number | null = null

  modelOptions: ApiModel[] = []
  dashboardData: ApiDashboard | null = null
  isModelsLoading = false
  isLoading = false
  error: string | null = null

  constructor(indicatorStore: IndicatorStore) {
    this.indicatorStore = indicatorStore
    makeAutoObservable(this, { indicatorStore: false }, { autoBind: true })
  }

  async initializeDashboard() {
    await this.fetchModels()
    await this.fetchDashboard()
  }

  async fetchModels() {
    this.isModelsLoading = true
    try {
      const res = await fetch('/api/models')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: ApiModel[] = await res.json()
      runInAction(() => {
        this.modelOptions = data
        if (this.selectedModelId == null && data.length > 0) {
          this.selectedModelId = data[0].id
        }
        this.isModelsLoading = false
      })
    } catch {
      runInAction(() => {
        this.modelOptions = []
        this.isModelsLoading = false
      })
    }
  }

  async fetchDashboard(modelId: number | null = this.selectedModelId) {
    this.isLoading = true
    this.error = null
    try {
      const params = new URLSearchParams()
      if (modelId != null) {
        params.set('model_id', String(modelId))
      }
      const query = params.toString()
      const res = await fetch(query ? `/api/dashboard?${query}` : '/api/dashboard')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: ApiDashboard = await res.json()
      runInAction(() => {
        this.dashboardData = data
        this.selectedModelId = data.model?.id ?? modelId
        if (data.target_indicator) {
          this.selectedIndicatorId = String(data.target_indicator.id)
        }
        this.isLoading = false
      })
    } catch {
      runInAction(() => {
        this.error = 'Не удалось загрузить данные дашборда. Убедитесь, что API-сервер запущен.'
        this.isLoading = false
      })
    }
  }

  setSelectedIndicator(id: string) {
    this.selectedIndicatorId = id
  }

  setSelectedModel(id: string) {
    const parsed = Number(id)
    this.selectedModelId = Number.isFinite(parsed) ? parsed : null
    void this.fetchDashboard(this.selectedModelId)
  }

  setScenarioType(type: ScenarioType) {
    this.scenarioType = type
  }

  setTimeRange(range: 'all' | 'last5' | 'last10') {
    this.timeRange = range
  }

  get selectedIndicator(): Indicator | undefined {
    return this.indicatorStore.indicators.find((ind) => ind.id === this.selectedIndicatorId)
  }

  get targetName(): string {
    return this.dashboardData?.target_indicator?.name ?? this.selectedIndicator?.name ?? '—'
  }

  get targetUnit(): string {
    return this.dashboardData?.target_indicator?.unit ?? this.selectedIndicator?.unit ?? ''
  }

  get scenarioApiName(): string {
    return SCENARIO_RU[this.scenarioType]
  }

  get timeSeries(): TimeSeriesPoint[] {
    if (!this.dashboardData) return []

    const history: TimeSeriesPoint[] = this.dashboardData.history.map((h) => ({
      year: h.year,
      actual: h.value,
    }))

    const estimate = this.dashboardData.forecasts['Оценка'] ?? []
    const selected = this.dashboardData.forecasts[this.scenarioApiName] ?? []
    const base = this.dashboardData.forecasts['Базовый'] ?? []
    const conservative = this.dashboardData.forecasts['Консервативный'] ?? []

    const bandByYear = new Map<number, { lower?: number; upper?: number }>()
    ;[...estimate, ...base].forEach((point) => {
      bandByYear.set(point.year, {
        ...(bandByYear.get(point.year) ?? {}),
        upper: point.value,
      })
    })
    ;[...estimate, ...conservative].forEach((point) => {
      bandByYear.set(point.year, {
        ...(bandByYear.get(point.year) ?? {}),
        lower: point.value,
      })
    })

    const forecastSource = [...estimate, ...selected].sort((a, b) => a.year - b.year)
    const forecast: TimeSeriesPoint[] = forecastSource.map((f) => ({
      year: f.year,
      forecast: f.value,
      upper: bandByYear.get(f.year)?.upper,
      lower: bandByYear.get(f.year)?.lower,
    }))

    return [...history, ...forecast]
  }

  get shapContributions(): ShapFactorContribution[] {
    if (!this.dashboardData) return []

    const forecasts = this.dashboardData.forecasts[this.scenarioApiName] ?? []
    if (!forecasts.length) return []

    const targetYear = forecasts[forecasts.length - 1].year
    const key = `${targetYear}_${this.scenarioApiName}`
    const items = this.dashboardData.shap[key] ?? []
    if (!items.length) return []

    const totalAbs = items.reduce((sum, item) => sum + Math.abs(item.contribution), 0)

    return items.map((item) => ({
      id: String(item.indicator_id),
      name: item.indicator_name,
      contribution:
        totalAbs > 0
          ? (item.direction === 'negative'
              ? -Math.abs(item.contribution)
              : Math.abs(item.contribution)) / totalAbs
          : 0,
    }))
  }

  get kpis(): {
    currentValue: number | null
    currentYear: number | null
    yoyChange: number | null
    cagr: number | null
  } {
    const history = this.dashboardData?.history ?? []
    if (history.length < 2) {
      return { currentValue: null, currentYear: null, yoyChange: null, cagr: null }
    }

    const last = history[history.length - 1]
    const prev = history[history.length - 2]
    const first = history[0]
    const years = last.year - first.year

    return {
      currentValue: last.value,
      currentYear: last.year,
      yoyChange: ((last.value - prev.value) / prev.value) * 100,
      cagr: years > 0 ? (Math.pow(last.value / first.value, 1 / years) - 1) * 100 : null,
    }
  }

  get modelInfo() {
    return this.dashboardData?.model ?? null
  }
}

// ---------------------------------------------------------------------------
// RootStore
// ---------------------------------------------------------------------------

export class RootStore {
  ui: UiStore
  indicators: IndicatorStore
  dashboards: DashboardStore

  constructor() {
    this.ui = new UiStore()
    this.indicators = new IndicatorStore()
    this.dashboards = new DashboardStore(this.indicators)
  }
}

const rootStore = new RootStore()

const RootStoreContext = createContext<RootStore | null>(null)

export function RootStoreProvider({ children }: { children: ReactNode }) {
  return (
    <RootStoreContext.Provider value={rootStore}>{children}</RootStoreContext.Provider>
  )
}

export function useRootStore(): RootStore {
  const ctx = useContext(RootStoreContext)
  if (!ctx) {
    throw new Error('useRootStore must be used within RootStoreProvider')
  }
  return ctx
}
