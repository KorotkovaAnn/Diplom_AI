import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Box, Button, Typography } from '@mui/material'

export function HomePage() {
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' })
  }, [])

  return (
    <div className="home-page home-page-with-bg">
      
      <HeroSection />
      <CapabilitiesAndMetrics />
      <ProcessAndQuickActions />
      <TechStackSection />
      <UpdatesAndFaq />
    </div>
  )
}

function HeroSection() {
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

        <Typography component="p" className="hero-subtitle">
          Комплексный анализ данных с применением машинного обучения и нейронных сетей
          для точного прогнозирования показателей на 4 года вперёд. Интерпретируемые
          результаты для обоснованных управленческих решений.
        </Typography>

        <Box
          sx={{
            mt: 2,
            mb: 1.5,
            display: 'grid',
            gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
            gap: 2,
            fontSize: 12,
          }}
        >
          <Box>
            <Typography variant="subtitle2" sx={{ fontSize: 13, mb: 0.5 }}>
              Для руководителей региона
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
              Сводка ключевых трендов и рисков в одном окне для подготовки решений.
            </Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" sx={{ fontSize: 13, mb: 0.5 }}>
              Для аналитиков и экономистов
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
              Доступ к детализации показателей, методологии и качеству прогнозов.
            </Typography>
          </Box>
          <Box>
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
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            <Button
              component={Link}
              to="/dashboards"
              variant="contained"
              color="primary"
              className="btn btn-primary"
            >
              Перейти к дашбордам
            </Button>

            <Button
              variant="outlined"
              color="error"
              className="btn btn-ghost"
              sx={{
                justifyContent: 'center',
                borderColor: 'rgba(220, 38, 38, 0.35)',
                color: 'rgba(185, 28, 28, 0.9)',
                '&:hover': {
                  borderColor: 'rgba(220, 38, 38, 0.55)',
                  backgroundColor: 'rgba(248, 113, 113, 0.04)',
                },
              }}
            >
              Скачать демо-отчёт (PDF)
            </Button>
          </Box>

          <Button
            component={Link}
            to="/about"
            variant="outlined"
            color="primary"
            className="btn btn-outline"
          >
            О возможностях
          </Button>
        </Box>

        <Box className="hero-benefits">
          <span className="hero-benefit-item">48 социально-экономических показателей</span>
          <span className="hero-benefit-item">
            Прогноз на 4 года с доверительными интервалами
          </span>
          <span className="hero-benefit-item">SHAP-анализ факторов влияния</span>
          <span className="hero-benefit-item">Экспорт в PDF и Excel для отчётов</span>
        </Box>
      </Box>

      <Box component="aside" className="hero-visual glass-panel">
        <Box className="hero-grid" aria-hidden="true" />

        <Box className="hero-visual-inner">
          <Box className="hero-overlay-card">
            <Box className="hero-overlay-title">Инвестиции в основной капитал</Box>
            <Box className="hero-overlay-subtitle">
              Прогноз +4 года с доверительным интервалом и ключевыми драйверами роста.
            </Box>

            <Box className="hero-metric-row">
              <Box>
                <Box className="hero-metric-label">Текущее значение</Box>
                <Box className="hero-metric-value">1 250 млрд ₽</Box>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Box className="hero-metric-label">Год к году</Box>
                <Box className="hero-metric-change">+6,4%</Box>
              </Box>
            </Box>

            <Box className="hero-metric-row">
              <Box>
                <Box className="hero-metric-label">Горизонт прогноза</Box>
                <Box className="hero-metric-value">+4 года</Box>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Box className="hero-metric-label">Точность моделей (MAPE)</Box>
                <Box className="hero-metric-change">3,1%</Box>
              </Box>
            </Box>
          </Box>

          <Box className="hero-overlay-card">
            <Box className="hero-overlay-title">Инвестиции в основной капитал</Box>
            <Box className="hero-overlay-subtitle">
              Сценарный анализ: базовый, оптимистичный и пессимистичный варианты динамики
              инвестиций.
            </Box>

            <Box className="hero-metric-row">
              <Box>
                <Box className="hero-metric-label">Базовый сценарий</Box>
                <Box className="hero-metric-value">+18% к 2028 году</Box>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Box className="hero-metric-label">Оптимистичный сценарий</Box>
                <Box className="hero-metric-change">+25%</Box>
              </Box>
            </Box>

            <Box className="hero-metric-row">
              <Box>
                <Box className="hero-metric-label">Пессимистичный сценарий</Box>
                <Box className="hero-metric-value">+9%</Box>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Box className="hero-metric-label">Основной драйвер</Box>
                <Box className="hero-metric-change">Инфраструктурные проекты</Box>
              </Box>
            </Box>
          </Box>
        </Box>
      </Box>
    </Box>
  )
}

function CapabilitiesAndMetrics() {
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
                Глубокий анализ временных рядов по 48 показателям: тренды, сезонность,
                экстремумы и структурные разрывы.
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
                Система сценарного моделирования и экспорта отчётов для подготовки
                управленческих документов и докладов.
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
            <div className="metric-value">48</div>
            <div className="metric-caption">
              экономика • инвестиции • демография • социальная сфера
            </div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Горизонт прогноза</div>
            <div className="metric-value">+4 года</div>
            <div className="metric-caption">базовый, оптимистичный и пессимистичный</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Частота обновления</div>
            <div className="metric-value">ежемесячно</div>
            <div className="metric-caption">автоматическая интеграция с API госорганов</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Модели и SHAP</div>
            <div className="metric-value">3,1% MAPE</div>
            <div className="metric-caption">высокая точность и интерпретируемость</div>
          </div>
        </div>
      </aside>
    </section>
  )
}

function ProcessAndQuickActions() {
  return (
    <section className="home-grid">
      <div className="timeline-section glass-panel">
        <h2 className="home-section-title">Как это работает</h2>
        <p className="home-section-subtitle">
          Пять шагов от сбора данных до готового отчёта для руководства региона.
        </p>
        <div className="timeline-row">
          <TimelineStep
            number={1}
            color="linear-gradient(145deg, #3b82f6, #06b6d4)"
            title="Сбор данных"
            text="Автоматическая интеграция с API госорганов, исторические ряды, пространственные данные и ручная загрузка."
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
            text="ARIMA, ансамблевые модели, RNN/LSTM и AutoML для выбора наилучшей архитектуры."
          />
          <TimelineStep
            number={4}
            color="linear-gradient(145deg, #f59e0b, #fbbf24)"
            title="Интерпретация"
            text="SHAP-анализ, доверительные интервалы и сценарные сравнения для прозрачности решений."
          />
          <TimelineStep
            number={5}
            color="linear-gradient(145deg, #10b981, #06b6d4)"
            title="Визуализация и отчёты"
            text="Интерактивные дашборды, экспорт в PDF/Excel и готовые аналитические записки."
          />
        </div>
      </div>

      <aside className="quick-actions glass-panel">
        <div style={{ gridColumn: '1 / -1', marginBottom: 4 }}>
          <h2 className="home-section-title">Быстрые действия</h2>
          <p className="home-section-subtitle">
            Один клик до ключевых дашбордов и отчётов.
          </p>
        </div>
        <QuickAction
          color="linear-gradient(145deg, #22c55e, #16a34a)"
          icon="💰"
          title="Инвестиции"
          subtitle="Перейти к дашборду по инвестициям"
        />
        <QuickAction
          color="linear-gradient(145deg, #0ea5e9, #2563eb)"
          icon="📉"
          title="Дефлятор"
          subtitle="Индексы-дефляторы и ценовые показатели"
        />
        <QuickAction
          color="linear-gradient(145deg, #a855f7, #ec4899)"
          icon="🧮"
          title="Сценарное моделирование"
          subtitle="What-if анализ и чувствительность"
        />
        <QuickAction
          color="linear-gradient(145deg, #f97316, #facc15)"
          icon="📄"
          title="Экспорт отчёта"
          subtitle="Сформировать PDF/Excel для совещания"
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
  color: string
  icon: string
  title: string
  subtitle: string
}

function QuickAction({ color, icon, title, subtitle }: QuickActionProps) {
  return (
    <button type="button" className="quick-action-button">
      <div className="quick-action-icon" style={{ backgroundImage: color }}>
        {icon}
      </div>
      <div>
        <div className="quick-action-title">{title}</div>
        <div className="quick-action-subtitle">{subtitle}</div>
      </div>
    </button>
  )
}

function TechStackSection() {
  return (
    <section className="tech-stack glass-panel">
      <h2 className="home-section-title">Технологический стек платформы</h2>
      <p className="home-section-subtitle">
        Современные инструменты для высокой точности прогнозов, масштабируемости и
        интерпретируемости результатов.
      </p>
      <div className="tech-columns">
        <div className="tech-column">
          <div className="tech-column-title">Данные</div>
          <div className="tech-pill">PostgreSQL</div>
          <div className="tech-pill">MinIO</div>
          <div className="tech-pill">Pandas</div>
          <div className="tech-pill">NumPy</div>
        </div>
        <div className="tech-column">
          <div className="tech-column-title">ML &amp; AI</div>
          <div className="tech-pill">Scikit-learn</div>
          <div className="tech-pill">PyTorch</div>
          <div className="tech-pill">TensorFlow</div>
          <div className="tech-pill">LAMA AutoML</div>
          <div className="tech-pill">SHAP</div>
        </div>
        <div className="tech-column">
          <div className="tech-column-title">Визуализация</div>
          <div className="tech-pill">Plotly</div>
          <div className="tech-pill">Matplotlib</div>
          <div className="tech-pill">Seaborn</div>
          <div className="tech-pill">Apache Superset</div>
        </div>
        <div className="tech-column">
          <div className="tech-column-title">Backend</div>
          <div className="tech-pill">Python</div>
          <div className="tech-pill">FastAPI</div>
          <div className="tech-pill">Flask</div>
        </div>
        <div className="tech-column">
          <div className="tech-column-title">DevOps</div>
          <div className="tech-pill">Docker</div>
          <div className="tech-pill">Kubernetes</div>
          <div className="tech-pill">Airflow</div>
          <div className="tech-pill">Nginx</div>
        </div>
      </div>
    </section>
  )
}

const faqItems = [
  {
    id: 'q1',
    question: 'Что такое SHAP и зачем он нужен?',
    answer:
      'SHAP показывает вклад каждого фактора в итоговое значение прогноза. Зелёные столбцы увеличивают прогноз, красные — уменьшают.',
  },
  {
    id: 'q2',
    question: 'Можно ли доверять прогнозам при резких изменениях экономики?',
    answer:
      'Платформа учитывает исторические шоки и использует ансамбли моделей, однако для экстремальных сценариев рекомендуется сценарный анализ.',
  },
  {
    id: 'q3',
    question: 'Как часто обновляются данные?',
    answer:
      'Основные показатели обновляются ежемесячно или ежеквартально, в зависимости от доступности данных источников.',
  },
  {
    id: 'q4',
    question: 'Можно ли выгружать результаты в отчёты?',
    answer:
      'Да, поддерживается экспорт в PDF и Excel, а также прямое включение графиков в презентации.',
  },
]

function UpdatesAndFaq() {
  const [openId, setOpenId] = useState<string | null>(faqItems[0]?.id ?? null)

  return (
    <section className="updates-faq">
      <div className="updates-timeline glass-panel">
        <h2 className="home-section-title">Обновления платформы</h2>
        <p className="home-section-subtitle">
          Лента релизов и улучшений для прозрачности развития системы.
        </p>
        <ul className="updates-list">
          <li className="update-item">
            <div className="update-dot green" />
            <div className="update-card">
              <div className="update-title">
                Добавлен новый блок сценарного моделирования по инвестициям
              </div>
              <div className="update-time">сегодня</div>
            </div>
          </li>
          <li className="update-item">
            <div className="update-dot blue" />
            <div className="update-card">
              <div className="update-title">
                Улучшена точность моделей по демографии и рынку труда
              </div>
              <div className="update-time">вчера</div>
            </div>
          </li>
          <li className="update-item">
            <div className="update-dot orange" />
            <div className="update-card">
              <div className="update-title">Исправлены аномальные значения в данных за 2020 год</div>
              <div className="update-time">3 дня назад</div>
            </div>
          </li>
        </ul>
      </div>

      <aside className="faq-panel glass-panel">
        <h2 className="home-section-title">Частые вопросы</h2>
        <p className="home-section-subtitle">
          Краткие ответы по интерпретации прогнозов и работе с платформой.
        </p>
        {faqItems.map((item) => (
          <div key={item.id} className="faq-item">
            <button
              type="button"
              className="faq-header"
              onClick={() => setOpenId((prev) => (prev === item.id ? null : item.id))}
            >
              <span>{item.question}</span>
              <span>{openId === item.id ? '−' : '+'}</span>
            </button>
            {openId === item.id ? <div className="faq-answer">{item.answer}</div> : null}
          </div>
        ))}
      </aside>
    </section>
  )
}

