const sections = [
  { id: 'getting-started', title: 'Начало работы' },
  { id: 'user-guide', title: 'Руководство пользователя' },
  { id: 'methodology', title: 'Методология' },
  { id: 'api', title: 'API' },
  { id: 'faq', title: 'FAQ' },
  { id: 'glossary', title: 'Глоссарий' },
]

export function DocsPage() {
  return (
    <div className="docs-layout">
      <aside className="docs-sidebar glass-panel">
        <div className="docs-sidebar-title">Документация</div>
        <nav>
          <ul className="docs-nav">
            {sections.map((section) => (
              <li key={section.id}>
                <a href={`#${section.id}`}>{section.title}</a>
              </li>
            ))}
          </ul>
        </nav>
      </aside>

      <main className="docs-content glass-panel">
        <div className="docs-toc">
          <div className="docs-toc-title">На этой странице</div>
          <ul>
            {sections.map((section) => (
              <li key={section.id}>
                <a href={`#${section.id}`}>{section.title}</a>
              </li>
            ))}
          </ul>
        </div>

        <article className="docs-article">
          <section id="getting-started">
            <h1 className="page-title">Начало работы</h1>
            <p className="page-subtitle">
              Шаги для первого входа в платформу, настройки аккаунта и запуска базовых
              дашбордов.
            </p>
            <ol>
              <li>Получите доступ и авторизуйтесь в системе.</li>
              <li>Настройте роль и права доступа в соответствии с функциями пользователя.</li>
              <li>Ознакомьтесь с основными дашбордами и категориями показателей.</li>
              <li>Создайте первый отчёт и экспортируйте его для совещания.</li>
            </ol>
          </section>

          <section id="user-guide">
            <h2>Руководство пользователя</h2>
            <p>
              Этот раздел описывает навигацию по интерфейсу, работу с фильтрами, настройку
              графиков, сохранение избранных показателей и сценарное моделирование.
            </p>
          </section>

          <section id="methodology">
            <h2>Методология</h2>
            <p>
              Здесь подробно описаны применяемые методы машинного обучения, эконометрии и
              интерпретации, а также критерии выбора моделей и подходы к валидации.
            </p>
          </section>

          <section id="api">
            <h2>API</h2>
            <p>
              Платформа предоставляет REST API для интеграции с внешними системами. В этом
              разделе описываются основные endpoints, форматы запросов и ответов.
            </p>
          </section>

          <section id="faq">
            <h2>FAQ</h2>
            <p>Ответы на наиболее частые вопросы по работе с платформой и методологии.</p>
          </section>

          <section id="glossary">
            <h2>Глоссарий</h2>
            <p>
              Справочник терминов, используемых в интерфейсе, отчётах и документации платформы.
            </p>
          </section>
        </article>
      </main>
    </div>
  )
}

