import axios from 'axios'
import type {
  TemplateListResponse,
  TemplateContent,
  VersionListResponse,
  AIModifyResponse,
  CourseDraftStreamRequest,
} from '@/types/template'

const API_BASE = import.meta.env.DEV
  ? ''
  : (import.meta.env.VITE_API_BASE_URL || '')

/**
 * Encode filename for URL path, preserving forward slashes for subdirectories
 * Encodes each path segment separately to support subdirectory paths
 */
function encodeFilePath(filename: string): string {
  return filename.split('/').map(segment => encodeURIComponent(segment)).join('/')
}

export const templateApi = {
  /**
   * List all templates
   */
  async list(search?: string): Promise<TemplateListResponse> {
    const params = search ? { search } : {}
    const response = await axios.get(`${API_BASE}/templates/list`, { params })
    return response.data
  },

  /**
   * Get template content for editing
   */
  async getContent(filename: string): Promise<TemplateContent> {
    const response = await axios.get(`${API_BASE}/templates/${encodeFilePath(filename)}/content`)
    return response.data
  },

  /**
   * Save template content
   */
  async saveContent(
    filename: string,
    contentHtml: string,
    createVersion: boolean = true,
    changeSummary?: string
  ): Promise<{ success: boolean; version_id: string }> {
    const response = await axios.put(`${API_BASE}/templates/${encodeFilePath(filename)}/content`, {
      content_html: contentHtml,
      create_version: createVersion,
      change_summary: changeSummary,
    })
    return response.data
  },

  /**
   * Get version history
   */
  async getVersions(filename: string): Promise<VersionListResponse> {
    const response = await axios.get(`${API_BASE}/templates/${encodeFilePath(filename)}/versions`)
    return response.data
  },

  /**
   * Restore a version
   */
  async restoreVersion(filename: string, versionId: string): Promise<{ success: boolean }> {
    const response = await axios.post(
      `${API_BASE}/templates/${encodeFilePath(filename)}/versions/${versionId}/restore`
    )
    return response.data
  },

  /**
   * AI-assisted content modification
   */
  async aiModify(originalText: string, instruction: string): Promise<AIModifyResponse> {
    const response = await axios.post(`${API_BASE}/templates/ai-modify`, {
      original_text: originalText,
      instruction: instruction,
    })
    return response.data
  },

  async generateCourseDraftStream(
    data: CourseDraftStreamRequest,
    onEvent: (eventName: string, payload: any) => void,
    onDone: () => void,
    onError: (err: string) => void
  ): Promise<() => void> {
    const controller = new AbortController()
    const formData = new FormData()

    if (data.platform) formData.append('platform', data.platform)
    if (data.teacher_name) formData.append('teacher_name', data.teacher_name)
    if (data.subject) formData.append('subject', data.subject)
    if (data.grade) formData.append('grade', data.grade)
    if (data.topic) formData.append('topic', data.topic)
    if (typeof data.duration_minutes === 'number') formData.append('duration_minutes', String(data.duration_minutes))
    if (data.notes) formData.append('notes', data.notes)
    if (data.source_text) formData.append('source_text', data.source_text)
    formData.append('prefer_physics_pilot', data.prefer_physics_pilot ? 'true' : 'false')
    if (data.source_file) formData.append('source_file', data.source_file)

    fetch(`${API_BASE}/templates/course-to-draft/stream`, {
      method: 'POST',
      body: formData,
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
                onEvent(currentEvent, JSON.parse(line.slice(6)))
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

  /**
   * Download template
   */
  getDownloadUrl(filename: string): string {
    return `${API_BASE}/templates/download/${encodeFilePath(filename)}`
  },

  /**
   * Export template to specified format
   */
  async exportTemplate(filename: string, format: 'docx' | 'pdf' | 'md'): Promise<{
    success: boolean
    download_url: string
    format: string
    file_size: number
  }> {
    const response = await axios.post(
      `${API_BASE}/templates/${encodeFilePath(filename)}/export`,
      { format }
    )
    return response.data
  },
}
