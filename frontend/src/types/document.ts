/**
 * Document import and processing types
 */

/**
 * Uploaded document information
 */
export interface UploadedDocument {
  document_id: string
  filename: string
  file_size: number
  uploaded_at: string
  text_preview: string
  metadata: {
    parser?: string
    word_count?: number
    page_count?: number
    warnings?: number
    images?: number
    [key: string]: any
  }
}

/**
 * Processing options enum
 */
export enum ProcessingOption {
  EXTRACT_EXERCISES = 'extract_exercises',
  SUMMARIZE = 'summarize',
  EXTRACT_TEACHING_THOUGHTS = 'extract_teaching_thoughts',
  CUSTOM = 'custom'
}

/**
 * Processing option labels for UI
 */
export const ProcessingOptionLabels: Record<string, string> = {
  'extract_exercises': '提取习题',
  'summarize': '归纳总结',
  'extract_teaching_thoughts': '提取教学思路',
  'custom': '自定义处理'
}

/**
 * Processing option descriptions for UI
 */
export const ProcessingOptionDescriptions: Record<string, string> = {
  'extract_exercises': '自动识别并提取文档中的选择题、填空题、简答题等各类习题',
  'summarize': '提取核心知识点、关键概念,生成结构化总结',
  'extract_teaching_thoughts': '识别教学环节、方法和设计意图,提取教学思路',
  'custom': '使用自定义指令处理文档内容'
}

/**
 * Document processing request
 */
export interface DocumentProcessingRequest {
  document_id: string
  processing_option: ProcessingOption | string
  custom_prompt?: string
}

/**
 * Batch document processing request
 */
export interface BatchDocumentProcessingRequest {
  document_ids: string[]
  processing_option: ProcessingOption | string
  custom_prompt?: string
}

/**
 * Document processing result
 */
export interface ProcessingResult {
  processing_id: string
  result: string
  processing_option: string
  processed_at: string
  metadata: {
    model?: string
    processing_time_ms?: number
    document_filename?: string
    custom_prompt?: string
    [key: string]: any
  }
}

/**
 * Processing history item
 */
export interface ProcessingHistoryItem {
  processing_id: string
  document_filename: string
  processing_option: string
  custom_prompt?: string
  result_preview: string
  processed_at: string
}

/**
 * Processing history response
 */
export interface ProcessingHistoryResponse {
  items: ProcessingHistoryItem[]
  total: number
  has_more: boolean
}

/**
 * Document upload error
 */
export class DocumentUploadError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'DocumentUploadError'
  }
}

/**
 * Document processing error
 */
export class DocumentProcessingError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'DocumentProcessingError'
  }
}

/**
 * Document processing timeout error
 */
export class DocumentProcessingTimeoutError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'DocumentProcessingTimeoutError'
  }
}
