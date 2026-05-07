/**
 * Document Import Zone Component
 * 
 * Provides drag-and-drop file upload interface for document import
 */

import { useState, useRef, DragEvent, ChangeEvent } from 'react'
import { documentApi } from '@/api/document'
import type { UploadedDocument } from '@/types/document'
import Button from './ui/Button'

interface DocumentImportZoneProps {
  onFileUploaded: (document: UploadedDocument) => void
  onError: (error: string) => void
  maxFileSize?: number // in bytes, default 10MB
  acceptedFormats?: string[] // default ['.doc', '.docx', '.pdf']
}

export default function DocumentImportZone({
  onFileUploaded,
  onError,
  maxFileSize = 10 * 1024 * 1024,
  acceptedFormats = ['.doc', '.docx', '.pdf'],
}: DocumentImportZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Format file size for display
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  // Validate file
  const validateFile = (file: File): string | null => {
    // Check file format
    const fileName = file.name.toLowerCase()
    const isValidFormat = acceptedFormats.some(ext => fileName.endsWith(ext))
    
    if (!isValidFormat) {
      return `不支持的文件格式,请上传Word或PDF文档。支持的格式: ${acceptedFormats.join(', ')}`
    }

    // Check file size
    if (file.size > maxFileSize) {
      return `文件大小超过限制,请上传小于${formatFileSize(maxFileSize)}的文件。当前文件大小: ${formatFileSize(file.size)}`
    }

    return null
  }

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    // Validate file
    const validationError = validateFile(file)
    if (validationError) {
      onError(validationError)
      return
    }

    setIsUploading(true)
    setUploadProgress(0)

    try {
      // Simulate progress (since we don't have real progress from axios)
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 200)

      // Upload file
      const uploadedDoc = await documentApi.uploadDocument(file)

      // Complete progress
      clearInterval(progressInterval)
      setUploadProgress(100)

      // Notify parent
      setTimeout(() => {
        onFileUploaded(uploadedDoc)
        setIsUploading(false)
        setUploadProgress(0)
      }, 300)
    } catch (error: any) {
      setIsUploading(false)
      setUploadProgress(0)
      onError(error.message || '文档上传失败,请稍后重试')
    }
  }

  // Drag and drop handlers
  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    
    // Only set dragging to false if leaving the drop zone itself
    if (e.currentTarget === e.target) {
      setIsDragging(false)
    }
  }

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      handleFileUpload(files[0])
    }
  }

  // File input change handler
  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileUpload(files[0])
    }
    // Reset input value to allow selecting the same file again
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Click to browse files
  const handleBrowseClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="w-full">
      {/* Drop Zone */}
      <div
        className={`
          relative rounded-lg border-2 border-dashed p-8 text-center transition-all duration-200
          ${isDragging 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 bg-gray-50 hover:border-gray-400 hover:bg-gray-100'
          }
          ${isUploading ? 'pointer-events-none opacity-60' : 'cursor-pointer'}
        `}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleBrowseClick}
      >
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedFormats.join(',')}
          onChange={handleFileInputChange}
          className="hidden"
        />

        {/* Upload Icon */}
        <div className="mb-4 flex justify-center">
          {isUploading ? (
            <div className="h-16 w-16 animate-spin rounded-full border-4 border-gray-200 border-t-blue-500" />
          ) : (
            <svg
              className={`h-16 w-16 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`}
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
          )}
        </div>

        {/* Upload Text */}
        {isUploading ? (
          <div>
            <p className="mb-2 text-lg font-medium text-gray-700">正在上传文档...</p>
            <div className="mx-auto mt-4 h-2 w-64 overflow-hidden rounded-full bg-gray-200">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="mt-2 text-sm text-gray-500">{uploadProgress}%</p>
          </div>
        ) : (
          <div>
            <p className="mb-2 text-lg font-medium text-gray-700">
              {isDragging ? '释放文件以上传' : '拖拽文档到此处'}
            </p>
            <p className="mb-4 text-sm text-gray-500">
              或点击选择文件
            </p>
            <Button
              variant="primary"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                handleBrowseClick()
              }}
            >
              选择文件
            </Button>
            <p className="mt-4 text-xs text-gray-400">
              支持格式: {acceptedFormats.join(', ')} | 最大 {formatFileSize(maxFileSize)}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
