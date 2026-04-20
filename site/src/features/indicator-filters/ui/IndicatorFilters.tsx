import type { IndicatorCategory } from '../../../stores/rootStore'

type CategoryFilter = IndicatorCategory | 'all'

interface IndicatorFiltersProps {
  activeCategory: CategoryFilter
  searchQuery: string
  onSearchChange: (value: string) => void
  onCategoryChange: (category: CategoryFilter) => void
}

const categoryPills: { id: CategoryFilter; label: string }[] = [
  { id: 'all', label: 'Все' },
  { id: 'economy', label: 'Экономика' },
  { id: 'investment', label: 'Инвестиции' },
  { id: 'construction', label: 'Строительство' },
  { id: 'demography', label: 'Демография' },
  { id: 'labour', label: 'Рынок труда' },
  { id: 'social', label: 'Социальная сфера' },
]

export function IndicatorFilters({
  activeCategory,
  searchQuery,
  onSearchChange,
  onCategoryChange,
}: IndicatorFiltersProps) {
  return (
    <div className="indicator-filters">
      <div className="indicator-search">
        <input
          type="search"
          placeholder="Найдите показатель по названию или описанию..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </div>
      <div className="indicator-categories">
        {categoryPills.map((pill) => (
          <button
            key={pill.id}
            type="button"
            className={
              activeCategory === pill.id
                ? 'indicator-pill indicator-pill-active'
                : 'indicator-pill'
            }
            onClick={() => onCategoryChange(pill.id)}
          >
            {pill.label}
          </button>
        ))}
      </div>
    </div>
  )
}

