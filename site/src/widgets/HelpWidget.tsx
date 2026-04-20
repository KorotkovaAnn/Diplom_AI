import { useState } from 'react'

export function HelpWidget() {
  const [open, setOpen] = useState(false)

  return (
    <div className="help-widget">
      {open ? (
        <div className="help-widget-panel glass-panel">
          <div className="help-widget-header">
            <span>Нужна помощь?</span>
            <button
              type="button"
              aria-label="Закрыть окно помощи"
              className="icon-button"
              onClick={() => setOpen(false)}
            >
              ×
            </button>
          </div>
          <div className="help-widget-body">
            <p>
              Это быстрый доступ к справке и FAQ. Выберите раздел, чтобы узнать
              больше о возможностях платформы.
            </p>
            <ul className="help-widget-links">
              <li>Как прочитать прогноз и доверительный интервал?</li>
              <li>Что означают зелёные и красные столбцы SHAP?</li>
              <li>Как настроить сценарное моделирование?</li>
              <li>Где скачать отчёт в PDF и Excel?</li>
            </ul>
          </div>
        </div>
      ) : null}

      <button
        type="button"
        className="help-widget-button"
        aria-label="Открыть окно помощи"
        onClick={() => setOpen(true)}
      >
        ?
      </button>
    </div>
  )
}

