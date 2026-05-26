import { useEffect } from 'react'

export function DataUploadPage() {
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' })
  }, [])

  return (
    <div>
      <h1 className="page-title">Загрузка данных</h1>
      <p className="page-subtitle">
        Раздел для ручной загрузки исходных данных. Наполнение интерфейса будет
        добавлено следующим этапом.
      </p>
    </div>
  )
}
