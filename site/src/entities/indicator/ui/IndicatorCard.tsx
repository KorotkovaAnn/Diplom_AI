import type { Indicator } from '../../../stores/rootStore'

interface IndicatorCardProps {
  indicator: Indicator
  isFavorite: boolean
  onToggleFavorite: () => void
  categoryLabel: string
}

export function IndicatorCard({
  indicator,
  isFavorite,
  onToggleFavorite,
  categoryLabel,
}: IndicatorCardProps) {
  return (
    <article className="indicator-card glass-panel">
      <header className="indicator-card-header">
        <div>
          <div className="indicator-name">{indicator.name}</div>
          <div className="indicator-meta">
            <span className="indicator-category-pill">{categoryLabel}</span>
            <span className="indicator-meta-separator">•</span>
            <span>{indicator.updateFrequency}</span>
          </div>
        </div>
        <button
          type="button"
          className={isFavorite ? 'icon-star icon-star-active' : 'icon-star'}
          onClick={onToggleFavorite}
          aria-label="Добавить в избранное"
        />
      </header>
      <p className="indicator-description">{indicator.description}</p>

      {/* График и базовая статистика оставляем в pages/IndicatorsPage,
          чтобы не усложнять зависимость entities от Recharts.
          При желании их можно вынести в отдельный ui-компонент. */}
    </article>
  )
}

