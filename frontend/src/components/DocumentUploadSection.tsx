/**
 * Document Upload Section Component
 * 
 * Handles multiple document uploads with drag-and-drop
 */

import { useState, useRef, DragEvent } from 'react'
import { documentApi } from '@/api/document'
import type { UploadedDocument } from '@/types/document'
import Button from './ui/Button'
import Card from './ui/Card'

interface DocumentUploadSectionProps {
  uploadedDocuments: UploadedDocument[]
  onDocumentUploaded: (document: UploadedDocument) => void
  onRemoveDocument: (documentId: string) => void
  onContinue: () => void
}

export default function DocumentUploadSection({
  uploadedDocuments,
  onDocumentUploaded,
  onRemoveDocument,
  onContinue,
}: DocumentUploadSectionProps) {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragActive, setIsDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return

    setError(null)
    setUploading(true)

    try {
      // Upload files one by one
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const document = await documentApi.uploadDocument(file)
        onDocumentUploaded(document)
      }
    } catch (err: any) {
      setError(err.message || '文档上传失败')
    } finally {
      setUploading(false)
    }
  }

  const handleDragEnter = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(true)
  }

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
  }

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    const files = e.dataTransfer.files
    handleFiles(files)
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files)
  }

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <Card>
        <div
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={handleClick}
          className={`
            cursor-pointer rounded-lg border-2 border-dashed p-12 text-center transition-colors
            ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'}
            ${uploading ? 'cursor-not-allowed opacity-50' : ''}
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".doc,.docx,.pdf"
            onChange={handleFileInputChange}
            disabled={uploading}
            className="hidden"
          />
          
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>

          <p className="mt-4 text-base font-medium text-gray-900">
            {isDragActive ? '释放文件以上传' : '拖拽文件到此处，或点击选择文件'}
          </p>
          <p className="mt-2 text-sm text-gray-500">
            支持 .doc, .docx, .pdf 格式，单个文件最大 10MB
          </p>
          <p className="mt-1 text-sm text-gray-500">
            可同时上传多个文档
          </p>

          {uploading && (
            <div className="mt-4 flex items-center justify-center gap-2">
              <svg className="h-5 w-5 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <span className="text-sm text-blue-600">上传中...</span>
            </div>
          )}
        </div>
      </Card>

      {/* Error Message */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <div className="flex items-start gap-3">
            <svg className="h-5 w-5 text-red-500 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </Card>
      )}

      {/* Uploaded Documents List */}
      {uploadedDocuments.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-900">
            已上传文档 ({uploadedDocuments.length})
          </h3>
          {uploadedDocuments.map((doc) => (
            <Card key={doc.document_id} className="hover:shadow-md transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0">
                  <svg className="h-10 w-10 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 truncate">{doc.filename}</p>
                  <p className="text-sm text-gray-500 mt-1">
                    {(doc.file_size / 1024).toFixed(1)} KB · {doc.metadata.word_count || 0} 字
                  </p>
                  <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                    {doc.text_preview}
                  </p>
                </div>
                <button
                  onClick={() => onRemoveDocument(doc.document_id)}
                  className="flex-shrink-0 rounded-lg p-2 text-gray-400 hover:bg-red-50 hover:text-red-600 transition-colors"
                >
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        {/* <Button
          variant="secondary"
          onClick={onViewHistory}
          className="flex-1"
        >
          查看历史记录
        </Button> */}
        <Button
          variant="primary"
          onClick={onContinue}
          disabled={uploadedDocuments.length === 0}
          className="flex-1"
        >
          继续处理 ({uploadedDocuments.length})
        </Button>
      </div>
    </div>
  )
}
