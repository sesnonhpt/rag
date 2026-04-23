export interface TemplateFileInfo {
  filename: string
  size_bytes: number
  size_display: string
  modified_at: string
  file_type: string
}

export interface TemplateListResponse {
  templates: TemplateFileInfo[]
  total: number
  directory: string
}

export interface TemplateContent {
  template_id: string
  filename: string
  content_html: string
  version_id: string | null
  metadata: Record<string, any>
}

export interface VersionInfo {
  version_id: string
  created_at: string
  change_summary: string | null
}

export interface VersionListResponse {
  versions: VersionInfo[]
  total: number
}

export interface AIModifyResponse {
  modified_text: string
  processing_time_ms: number
}
