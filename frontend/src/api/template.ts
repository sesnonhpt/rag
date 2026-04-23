import axios from 'axios'
import type {
  TemplateListResponse,
  TemplateContent,
  VersionListResponse,
  AIModifyResponse,
} from '@/types/template'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

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
    const response = await axios.get(`${API_BASE}/templates/${encodeURIComponent(filename)}/content`)
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
    const response = await axios.put(`${API_BASE}/templates/${encodeURIComponent(filename)}/content`, {
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
    const response = await axios.get(`${API_BASE}/templates/${encodeURIComponent(filename)}/versions`)
    return response.data
  },

  /**
   * Restore a version
   */
  async restoreVersion(filename: string, versionId: string): Promise<{ success: boolean }> {
    const response = await axios.post(
      `${API_BASE}/templates/${encodeURIComponent(filename)}/versions/${versionId}/restore`
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

  /**
   * Download template
   */
  getDownloadUrl(filename: string): string {
    return `${API_BASE}/templates/download/${encodeURIComponent(filename)}`
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
      `${API_BASE}/templates/${encodeURIComponent(filename)}/export`,
      { format }
    )
    return response.data
  },
}
