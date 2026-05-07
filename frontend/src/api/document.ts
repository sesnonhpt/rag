/**
 * Document import and processing API client
 */

import axios from 'axios'
import type {
  UploadedDocument,
  DocumentProcessingRequest,
  ProcessingResult,
  ProcessingHistoryResponse,
  DocumentUploadError,
  DocumentProcessingError,
  DocumentProcessingTimeoutError,
} from '@/types/document'

const API_BASE = import.meta.env.DEV
  ? ''
  : (import.meta.env.VITE_API_BASE_URL || '')

/**
 * Document API client
 */
export const documentApi = {
  /**
   * Upload a document file
   * 
   * @param file - File to upload (.doc, .docx, or .pdf)
   * @returns Uploaded document information
   * @throws DocumentUploadError if upload fails
   */
  async uploadDocument(file: File): Promise<UploadedDocument> {
    try {
      // Validate file format
      const validExtensions = ['.doc', '.docx', '.pdf']
      const fileName = file.name.toLowerCase()
      const isValidFormat = validExtensions.some(ext => fileName.endsWith(ext))
      
      if (!isValidFormat) {
        throw new Error('不支持的文件格式,请上传Word或PDF文档')
      }
      
      // Validate file size (10MB max)
      const maxSize = 10 * 1024 * 1024
      if (file.size > maxSize) {
        throw new Error(`文件大小超过限制,请上传小于10MB的文件。当前文件大小: ${(file.size / 1024 / 1024).toFixed(1)}MB`)
      }
      
      // Create form data
      const formData = new FormData()
      formData.append('file', file)
      
      // Upload file
      const response = await axios.post<UploadedDocument>(
        `${API_BASE}/api/documents/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 30000, // 30 seconds for upload
        }
      )
      
      return response.data
    } catch (error: any) {
      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.detail || error.message
        throw new Error(message) as DocumentUploadError
      }
      throw error
    }
  },

  /**
   * Get document information by ID
   * 
   * @param documentId - Document ID
   * @returns Document information
   */
  async getDocumentInfo(documentId: string): Promise<UploadedDocument> {
    try {
      const response = await axios.get<UploadedDocument>(
        `${API_BASE}/api/documents/${documentId}`
      )
      return response.data
    } catch (error: any) {
      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.detail || error.message
        throw new Error(message)
      }
      throw error
    }
  },

  /**
   * Process document with AI
   * 
   * @param request - Processing request
   * @returns Processing result
   * @throws DocumentProcessingError if processing fails
   * @throws DocumentProcessingTimeoutError if processing times out
   */
  async processDocument(request: DocumentProcessingRequest): Promise<ProcessingResult> {
    try {
      const response = await axios.post<ProcessingResult>(
        `${API_BASE}/api/documents/process`,
        request,
        {
          timeout: 70000, // 70 seconds timeout (longer than backend 60s)
        }
      )
      
      return response.data
    } catch (error: any) {
      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.detail || error.message
        
        // Check for timeout error
        if (error.code === 'ECONNABORTED' || error.response?.status === 408) {
          throw new Error(message) as DocumentProcessingTimeoutError
        }
        
        throw new Error(message) as DocumentProcessingError
      }
      throw error
    }
  },

  /**
   * Get processing history with pagination
   * 
   * @param limit - Maximum number of items to return (default: 10, max: 50)
   * @param offset - Number of items to skip (default: 0)
   * @returns Processing history response
   */
  async getProcessingHistory(
    limit: number = 10,
    offset: number = 0
  ): Promise<ProcessingHistoryResponse> {
    try {
      const response = await axios.get<ProcessingHistoryResponse>(
        `${API_BASE}/api/documents/processing-history`,
        {
          params: { limit, offset },
        }
      )
      
      return response.data
    } catch (error: any) {
      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.detail || error.message
        throw new Error(message)
      }
      throw error
    }
  },

  /**
   * Get full processing result by processing ID
   * 
   * @param processingId - Processing ID
   * @returns Full processing result
   */
  async getProcessingResult(processingId: string): Promise<{ processing_id: string; result: string }> {
    try {
      const response = await axios.get<{ processing_id: string; result: string }>(
        `${API_BASE}/api/documents/processing-history/${processingId}/result`
      )
      
      return response.data
    } catch (error: any) {
      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.detail || error.message
        throw new Error(message)
      }
      throw error
    }
  },

  /**
   * Clear all processing history
   * 
   * @returns Number of items deleted
   */
  async clearProcessingHistory(): Promise<{ deleted_count: number }> {
    try {
      const response = await axios.delete<{ deleted_count: number }>(
        `${API_BASE}/api/documents/processing-history`
      )
      
      return response.data
    } catch (error: any) {
      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.detail || error.message
        throw new Error(message)
      }
      throw error
    }
  },
}

export default documentApi
