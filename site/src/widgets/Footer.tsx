export function Footer() {
  return (
    <footer className="app-footer">
      <div className="app-footer-inner">
        <div className="footer-column">
          <div className="footer-logo">
            <div className="logo-mark small">
              <span className="logo-bar logo-bar-1" />
              <span className="logo-bar logo-bar-2" />
              <span className="logo-bar logo-bar-3" />
            </div>
            <div className="footer-logo-text">
              <span className="logo-title">REGION AI FORECAST</span>
            </div>
          </div>
          <p className="footer-text">
            Цифровая платформа мониторинга и прогнозирования
            социально-экономических показателей региона.
          </p>
        </div>

        <div className="footer-column">
          <h3 className="footer-heading">Разделы</h3>
          <ul className="footer-links">
            <li>Главная</li>
            <li>Дашборды</li>
            <li>Показатели</li>
            <li>Прогнозы</li>
          </ul>
        </div>

        <div className="footer-column">
          <h3 className="footer-heading">Для разработчиков</h3>
          <ul className="footer-links">
            <li>Документация</li>
            <li>API</li>
            <li>Поддержка</li>
          </ul>
        </div>

        <div className="footer-column">
          <h3 className="footer-heading">Контакты</h3>
          <ul className="footer-links">
            <li>Аналитический центр региона</li>
            <li>support@region-ai.gov</li>
            <li>+7 (000) 000-00-00</li>
          </ul>
        </div>
      </div>

      <div className="app-footer-bottom">
        <span>© 2026 REGON AI FORECAST. Все права защищены.</span>
        <span className="footer-meta">
          Версия платформы 1.0 • Политика конфиденциальности • Условия использования
        </span>
      </div>
    </footer>
  )
}

