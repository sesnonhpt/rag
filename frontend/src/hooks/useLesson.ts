import { useState, useRef } from 'react'
import { lessonApi } from '@/api/lesson'
import type { LessonPlanRequest, LessonPlanResponse, LessonProgressEvent, LessonErrorEvent } from '@/types/lesson'

export function useLesson() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<LessonPlanResponse | null>(null)
  const [progressEvent, setProgressEvent] = useState<LessonProgressEvent | null>(null)
  const abortRef = useRef<(() => void) | null>(null)

  const generateStream = async (data: LessonPlanRequest, onResult?: (res: LessonPlanResponse) => void) => {
    abortRef.current?.()

    setLoading(true)
    setError(null)
    setResult(null)
    setProgressEvent(null)

    const abort = await lessonApi.generateStream(
      data,
      (eventName, payload) => {
        if (eventName === 'progress') {
          setProgressEvent(payload as LessonProgressEvent)
        } else if (eventName === 'result') {
          setResult(payload as LessonPlanResponse)
          onResult?.(payload as LessonPlanResponse)
          setLoading(false)
        } else if (eventName === 'error') {
          const err = payload as LessonErrorEvent
          setError(err.message)
          setLoading(false)
        }
      },
      () => {
        setLoading(false)
      },
      (errMsg) => {
        setError(errMsg)
        setLoading(false)
      }
    )

    abortRef.current = abort
  }

  const exportDocx = async (content: string, title: string) => {
    try {
      const blob = await lessonApi.exportDocx(content, title)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${title}.docx`
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (err: any) {
      setError(err.message || '导出失败')
    }
  }

  return {
    loading,
    error,
    result,
    progressEvent,
    generateStream,
    exportDocx,
  }
}
