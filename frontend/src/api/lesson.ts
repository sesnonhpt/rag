import apiClient from './client'
import type { LessonPlanRequest, LessonPlanResponse } from '@/types/lesson'

const API_BASE_URL = import.meta.env.DEV
  ? ''
  : (import.meta.env.VITE_API_BASE_URL || '')

export const lessonApi = {
  generate: async (data: LessonPlanRequest): Promise<LessonPlanResponse> => {
    const response = await apiClient.post<LessonPlanResponse>('/lesson-plan', data)
    return response.data
  },

  generateStream: async (
    data: LessonPlanRequest,
    onEvent: (eventName: string, payload: any) => void,
    onDone: () => void,
    onError: (err: string) => void
  ): Promise<() => void> => {
    const controller = new AbortController()

    fetch(`${API_BASE_URL}/lesson-plan/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(data),
      signal: controller.signal,
    }).then(async (response) => {
      if (!response.ok) {
        onError(`请求失败: ${response.status} ${response.statusText}`)
        onDone()
        return
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          let currentEvent = 'message'
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEvent = line.slice(7).trim()
            } else if (line.startsWith('data: ')) {
              try {
                const payload = JSON.parse(line.slice(6))
                onEvent(currentEvent, payload)
              } catch {
                // ignore malformed JSON
              }
              currentEvent = 'message'
            }
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          onError(err.message || '连接中断')
        }
      } finally {
        onDone()
      }
    }).catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err.message || '连接失败')
        onDone()
      }
    })

    return () => controller.abort()
  },

  exportDocx: async (contentHtml: string, title: string) => {
    const response = await apiClient.post(
      '/lesson-plan/export-docx',
      { content_html: contentHtml, title },
      { responseType: 'blob' }
    )
    return response.data
  },
}
