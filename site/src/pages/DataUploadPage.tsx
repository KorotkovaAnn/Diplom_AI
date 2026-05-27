import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ClearIcon from '@mui/icons-material/Clear'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import DownloadIcon from '@mui/icons-material/Download'
import SaveIcon from '@mui/icons-material/Save'
import UploadFileIcon from '@mui/icons-material/UploadFile'

interface ApiIndicator {
  id: number
  sphere: string
  name: string
  unit: string
}

interface FormIndicator {
  id: number
  sphere: string
  name: string
  unit: string
  role: 'target' | 'feature'
}

interface ParseTemplateResponse {
  status: 'ok' | 'error'
  year: number
  target_id: number
  values: Record<string, number>
  count: number
  errors: Array<{
    row: number | null
    indicator_id: number | null
    message: string
  }>
}

type UserMessage = { type: 'success' | 'error' | 'info'; text: string }

export function DataUploadPage() {
  const [targetIndicators, setTargetIndicators] = useState<ApiIndicator[]>([])
  const [selectedTargetId, setSelectedTargetId] = useState('')
  const [formIndicators, setFormIndicators] = useState<FormIndicator[]>([])
  const [nextYear, setNextYear] = useState<number | null>(null)
  const [values, setValues] = useState<Record<number, string>>({})
  const [errors, setErrors] = useState<Record<number, string>>({})
  const [pageError, setPageError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isTemplateLoading, setIsTemplateLoading] = useState(false)
  const [isTemplateParsing, setIsTemplateParsing] = useState(false)
  const [submitMessage, setSubmitMessage] = useState<UserMessage | null>(null)
  const [templateMessage, setTemplateMessage] = useState<UserMessage | null>(null)

  const [deleteYear, setDeleteYear] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)
  const [deleteMessage, setDeleteMessage] = useState<UserMessage | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchFormIndicators = useCallback(async (targetId: string) => {
    if (!targetId) {
      setFormIndicators([])
      return
    }

    setIsLoading(true)
    setSubmitMessage(null)
    setTemplateMessage(null)
    setDeleteMessage(null)
    try {
      const res = await fetch(`/api/dataset/form-indicators?target_id=${targetId}`)
      if (!res.ok) throw new Error('Ошибка загрузки показателей')
      const data: FormIndicator[] = await res.json()
      setFormIndicators(data)
      setValues(buildEmptyValues(data))
      setErrors({})
    } catch (error) {
      setFormIndicators([])
      setSubmitMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Ошибка загрузки',
      })
    } finally {
      setIsLoading(false)
    }
  }, [])

  const fetchTargets = useCallback(async () => {
    setPageError(null)
    try {
      const res = await fetch('/api/indicators?role=target')
      if (!res.ok) throw new Error()
      const data: ApiIndicator[] = await res.json()
      setTargetIndicators(data)
      if (data.length > 0) {
        setSelectedTargetId((current) => {
          if (current) return current
          void fetchFormIndicators(String(data[0].id))
          return String(data[0].id)
        })
      }
    } catch {
      setPageError('Не удалось загрузить target-показатели. Проверьте, что API-сервер запущен.')
    }
  }, [fetchFormIndicators])

  const fetchNextYear = useCallback(async () => {
    try {
      const res = await fetch('/api/dataset/next-year')
      if (!res.ok) throw new Error()
      const data: { next_year: number } = await res.json()
      setNextYear(data.next_year)
      setDeleteYear(String(data.next_year))
    } catch {
      setPageError('Не удалось определить следующий год для загрузки.')
    }
  }, [])

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' })
    void fetchTargets()
    void fetchNextYear()
  }, [fetchNextYear, fetchTargets])

  function handleTargetChange(event: ChangeEvent<HTMLSelectElement>) {
    const value = event.target.value
    setSelectedTargetId(value)
    void fetchFormIndicators(value)
  }

  function normalizeNumericInput(raw: string): string {
    return raw.replace(',', '.').replace(/\s/g, '')
  }

  function validateValue(raw: string): string | null {
    if (!raw.trim()) return 'Значение обязательно'
    const normalized = normalizeNumericInput(raw)
    if (!/^-?\d+(\.\d+)?$/.test(normalized)) return 'Допустимы только цифры, точка или запятая'
    return null
  }

  function handleValueChange(id: number, raw: string) {
    setValues((prev) => ({ ...prev, [id]: raw }))
    const error = validateValue(raw)
    setErrors((prev) => {
      const next = { ...prev }
      if (error) next[id] = error
      else delete next[id]
      return next
    })
  }

  function validateAll(): boolean {
    const newErrors: Record<number, string> = {}
    let valid = true
    formIndicators.forEach((indicator) => {
      const error = validateValue(values[indicator.id] ?? '')
      if (error) {
        newErrors[indicator.id] = error
        valid = false
      }
    })
    setErrors(newErrors)
    return valid
  }

  function clearForm() {
    setValues(buildEmptyValues(formIndicators))
    setErrors({})
    setSubmitMessage(null)
    setTemplateMessage({ type: 'info', text: 'Форма очищена' })
  }

  async function handleDownloadTemplate() {
    if (!selectedTargetId || nextYear === null) return

    setIsTemplateLoading(true)
    setTemplateMessage(null)
    try {
      const res = await fetch(`/api/dataset/template?target_id=${selectedTargetId}&year=${nextYear}`)
      if (!res.ok) {
        const error = await res.json().catch(() => ({}))
        throw new Error(error.detail || 'Не удалось скачать шаблон')
      }

      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `dataset_template_${selectedTargetId}_${nextYear}.xlsx`
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
      setTemplateMessage({ type: 'success', text: 'Шаблон скачан. Заполните колонку «Значение» и загрузите файл обратно.' })
    } catch (error) {
      setTemplateMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Не удалось скачать шаблон',
      })
    } finally {
      setIsTemplateLoading(false)
    }
  }

  async function handleTemplateFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file || !selectedTargetId || nextYear === null) return

    setIsTemplateParsing(true)
    setTemplateMessage(null)
    setSubmitMessage(null)
    try {
      const contentBase64 = await fileToBase64(file)
      const res = await fetch('/api/dataset/template/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_id: Number(selectedTargetId),
          year: nextYear,
          file_name: file.name,
          content_base64: contentBase64,
        }),
      })

      if (!res.ok) {
        const error = await res.json().catch(() => ({}))
        throw new Error(error.detail || 'Не удалось прочитать шаблон')
      }

      const result: ParseTemplateResponse = await res.json()
      const parsedValues: Record<number, string> = buildEmptyValues(formIndicators)
      Object.entries(result.values).forEach(([id, value]) => {
        parsedValues[Number(id)] = String(value)
      })
      setValues(parsedValues)

      const parsedErrors: Record<number, string> = {}
      result.errors.forEach((error) => {
        if (error.indicator_id !== null) parsedErrors[error.indicator_id] = error.message
      })
      setErrors(parsedErrors)

      if (result.errors.length > 0) {
        setTemplateMessage({
          type: 'error',
          text: `Шаблон прочитан частично: ${result.count} значений, ошибок: ${result.errors.length}. Исправьте подсвеченные поля.`,
        })
      } else {
        setTemplateMessage({
          type: 'success',
          text: `Шаблон загружен: ${result.count} значений готовы к сохранению.`,
        })
      }
    } catch (error) {
      setTemplateMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Не удалось загрузить шаблон',
      })
    } finally {
      setIsTemplateParsing(false)
    }
  }

  async function handleSubmit() {
    if (!validateAll() || nextYear === null || !selectedTargetId) return

    setIsSaving(true)
    setSubmitMessage(null)
    try {
      const payload: Record<number, number> = {}
      formIndicators.forEach((indicator) => {
        payload[indicator.id] = parseFloat(normalizeNumericInput(values[indicator.id]))
      })

      const res = await fetch('/api/dataset/year', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_id: Number(selectedTargetId), year: nextYear, values: payload }),
      })

      if (!res.ok) {
        const error = await res.json().catch(() => ({}))
        throw new Error(error.detail || 'Ошибка сохранения')
      }

      const result: { year: number; count: number } = await res.json()
      setSubmitMessage({
        type: 'success',
        text: `Данные за ${result.year} год сохранены (${result.count} показателей)`,
      })
      setValues(buildEmptyValues(formIndicators))
      setErrors({})
      void fetchNextYear()
    } catch (error) {
      setSubmitMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Ошибка сохранения',
      })
    } finally {
      setIsSaving(false)
    }
  }

  async function handleDelete() {
    if (!deleteYear.trim() || !selectedTargetId) return
    const yearNumber = parseInt(deleteYear, 10)
    if (Number.isNaN(yearNumber)) {
      setDeleteMessage({ type: 'error', text: 'Укажите корректный год' })
      return
    }

    const confirmed = window.confirm(`Удалить данные за ${yearNumber} год для выбранного target-показателя?`)
    if (!confirmed) return

    setIsDeleting(true)
    setDeleteMessage(null)
    try {
      const res = await fetch(
        `/api/dataset/year?year=${yearNumber}&target_id=${selectedTargetId}`,
        { method: 'DELETE' },
      )
      if (!res.ok) {
        const error = await res.json().catch(() => ({}))
        throw new Error(error.detail || 'Ошибка удаления')
      }
      const result: { deleted_count: number; year: number } = await res.json()
      setDeleteMessage({
        type: 'success',
        text: `Удалено ${result.deleted_count} записей за ${result.year} год`,
      })
      void fetchNextYear()
    } catch (error) {
      setDeleteMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Ошибка удаления',
      })
    } finally {
      setIsDeleting(false)
    }
  }

  const selectedTarget = useMemo(() => {
    return targetIndicators.find((indicator) => String(indicator.id) === selectedTargetId)
  }, [selectedTargetId, targetIndicators])

  const sortedIndicators = useMemo(() => {
    return [...formIndicators].sort((a, b) => {
      if (a.role !== b.role) return a.role === 'target' ? -1 : 1
      return a.id - b.id
    })
  }, [formIndicators])

  const filledCount = useMemo(() => {
    return formIndicators.filter((indicator) => values[indicator.id]?.trim()).length
  }, [formIndicators, values])

  const hasErrors = Object.keys(errors).length > 0
  const allFilled = formIndicators.length > 0 && filledCount === formIndicators.length
  const progressPercent = formIndicators.length > 0 ? Math.round((filledCount / formIndicators.length) * 100) : 0
  const canUseTemplate = Boolean(selectedTargetId && nextYear !== null && formIndicators.length > 0)

  return (
    <div>
      <h1 className="page-title">Загрузка данных</h1>
      <p className="page-subtitle">
        Добавление фактических значений показателей за новый год в базу данных.
      </p>

      {pageError && <div className="forecast-error">{pageError}</div>}

      <section className="glass-panel data-upload-control-panel">
        <div className="forecast-toolbar data-upload-toolbar">
          <div className="dashboard-filter-group indicators-target-filter">
            <div className="dashboard-filter-label">Target-показатель</div>
            <select
              className="dashboard-select"
              value={selectedTargetId}
              onChange={handleTargetChange}
              disabled={targetIndicators.length === 0}
            >
              {targetIndicators.length > 0 ? (
                targetIndicators.map((target) => (
                  <option key={target.id} value={target.id}>
                    {target.name}
                  </option>
                ))
              ) : (
                <option value="">Нет target-показателей</option>
              )}
            </select>
          </div>

          <div className="data-upload-year-card">
            <span>Год загрузки</span>
            <strong>{nextYear ?? '...'}</strong>
          </div>

          <div className="data-upload-progress-card">
            <div className="data-upload-progress-head">
              <span>Заполнено</span>
              <strong>
                {filledCount}/{formIndicators.length || 0}
              </strong>
            </div>
            <div className="data-upload-progress-track">
              <div style={{ width: `${progressPercent}%` }} />
            </div>
            <small>{hasErrors ? `${Object.keys(errors).length} ошибок` : `${progressPercent}% готово`}</small>
          </div>
        </div>
      </section>

      <section className="glass-panel data-upload-template-panel">
        <div className="data-upload-template-copy">
          <h2>Excel-шаблон</h2>
          <p>
            Скачайте готовый файл для выбранного target, заполните колонку «Значение» и загрузите его обратно.
          </p>
        </div>
        <div className="data-upload-actions">
          <button
            type="button"
            className="btn btn-outline btn-small"
            onClick={handleDownloadTemplate}
            disabled={!canUseTemplate || isTemplateLoading}
          >
            <DownloadIcon fontSize="small" />
            {isTemplateLoading ? 'Подготовка...' : 'Скачать шаблон'}
          </button>
          <button
            type="button"
            className="btn btn-outline btn-small"
            onClick={() => fileInputRef.current?.click()}
            disabled={!canUseTemplate || isTemplateParsing}
          >
            <UploadFileIcon fontSize="small" />
            {isTemplateParsing ? 'Проверка...' : 'Загрузить шаблон'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".xlsx"
            onChange={handleTemplateFileChange}
            hidden
          />
        </div>
      </section>

      {templateMessage && <MessageBox message={templateMessage} />}

      {isLoading && (
        <div className="glass-panel data-upload-empty">
          Загрузка показателей...
        </div>
      )}

      {!isLoading && formIndicators.length === 0 && !pageError && (
        <div className="glass-panel data-upload-empty">
          Выберите target-показатель, чтобы подготовить форму загрузки.
        </div>
      )}

      {!isLoading && sortedIndicators.length > 0 && nextYear !== null && (
        <section className="glass-panel data-upload-table-panel">
          <div className="data-upload-table-header">
            <div>
              <h2>Ввод данных за {nextYear} год</h2>
              <p>{selectedTarget?.name ?? 'Выбранный target-показатель'}</p>
            </div>
            <button type="button" className="btn btn-ghost btn-small" onClick={clearForm}>
              <ClearIcon fontSize="small" />
              Очистить
            </button>
          </div>

          <div className="data-upload-table-wrapper">
            <table className="forecast-table data-upload-table">
              <thead>
                <tr>
                  <th>Тип</th>
                  <th>Сфера</th>
                  <th>Показатель</th>
                  <th>Ед. изм.</th>
                  <th>Значение</th>
                </tr>
              </thead>
              <tbody>
                {sortedIndicators.map((indicator) => (
                  <tr key={indicator.id} className={indicator.role === 'target' ? 'indicators-target-row' : undefined}>
                    <td>
                      <span className={`indicator-role-pill ${indicator.role}`}>
                        {indicator.role === 'target' ? 'Target' : 'Feature'}
                      </span>
                    </td>
                    <td>{indicator.sphere}</td>
                    <td className="forecast-name">{indicator.name}</td>
                    <td className="forecast-unit">{indicator.unit}</td>
                    <td>
                      <div className="data-upload-value-cell">
                        <input
                          type="text"
                          inputMode="decimal"
                          value={values[indicator.id] ?? ''}
                          onChange={(event) => handleValueChange(indicator.id, event.target.value)}
                          placeholder="0.00"
                          className={errors[indicator.id] ? 'has-error' : undefined}
                        />
                        {errors[indicator.id] && <span>{errors[indicator.id]}</span>}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {submitMessage && <MessageBox message={submitMessage} />}

          <div className="data-upload-submit-row">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={isSaving || hasErrors || !allFilled}
            >
              <SaveIcon fontSize="small" />
              {isSaving ? 'Сохранение...' : `Сохранить данные за ${nextYear} год`}
            </button>
          </div>
        </section>
      )}

      {selectedTargetId && (
        <section className="glass-panel data-upload-danger-panel">
          <div>
            <h2>Удаление данных за год</h2>
            <p>
              Удаляются только записи, привязанные к выбранному target-показателю через реестр загрузок.
            </p>
          </div>
          <div className="data-upload-delete-controls">
            <div className="dashboard-filter-group">
              <div className="dashboard-filter-label">Год</div>
              <input
                type="text"
                className="dashboard-select"
                placeholder="2024"
                value={deleteYear}
                onChange={(event) => setDeleteYear(event.target.value)}
              />
            </div>
            <button
              type="button"
              className="btn btn-small data-upload-delete-button"
              onClick={handleDelete}
              disabled={isDeleting || !deleteYear.trim()}
            >
              <DeleteOutlineIcon fontSize="small" />
              {isDeleting ? 'Удаление...' : 'Удалить'}
            </button>
          </div>
          {deleteMessage && <MessageBox message={deleteMessage} />}
        </section>
      )}
    </div>
  )
}

function buildEmptyValues(indicators: FormIndicator[]): Record<number, string> {
  const initial: Record<number, string> = {}
  indicators.forEach((indicator) => {
    initial[indicator.id] = ''
  })
  return initial
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = String(reader.result)
      resolve(result.includes(',') ? result.split(',')[1] : result)
    }
    reader.onerror = () => reject(new Error('Не удалось прочитать файл'))
    reader.readAsDataURL(file)
  })
}

function MessageBox({ message }: { message: UserMessage }) {
  return (
    <div className={`data-upload-message ${message.type}`}>
      {message.text}
    </div>
  )
}
