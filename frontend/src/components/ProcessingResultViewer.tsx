/**
 * Processing Result Viewer Component
 * 
 * Displays document processing results with actions
 */

import { useState } from 'react'
import type { ProcessingResult } from '@/types/document'
import { ProcessingOptionLabels } from '@/types/document'
import Button from './ui/Button'

interface ProcessingResultViewerProps {
  result: ProcessingResult
  onApplyToEditor: (result: string) => void
  onCopy: () => void
  onReprocess: () => void
  onClose: () => void
}

export default function ProcessingResultViewer({
  result,
  onApplyToEditor,
  onCopy,
  onReprocess,
  onClose,
}: ProcessingResultViewerProps) {
  const [isApplying, setIsApplying] = useState(false)
  const [copySuccess, setCopySuccess] = useState(false)

  // Format date
  const formatDate = (isoString: string): string => {
    const date = new Date(isoString)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  // Handle copy to clipboard
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.result)
      setCopySuccess(true)
      onCopy()
      
      // Reset copy success after 2 seconds
      setTimeout(() => {
        setCopySuccess(false)
      }, 2000)
    } catch (error) {
      console.error('Failed to copy:', error)
    }
  }

  // Handle apply to editor
  const handleApply = async () => {
    setIsApplying(true)
    try {
      await onApplyToEditor(result.result)
    } finally {
      setIsApplying(false)
    }
  }

  // Get processing option label
  const optionLabel = ProcessingOptionLabels[result.processing_option as keyof typeof ProcessingOptionLabels] 
    || result.processing_option

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <svg className="h-5 w-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <h2 className="text-lg font-semibold text-gray-900">处理完成</h2>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-gray-600">
              <span className="flex items-center gap-1">
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                </svg>
                {optionLabel}
              </span>
              <span className="flex items-center gap-1">
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {formatDate(result.processed_at)}
              </span>
              {result.metadata.processing_time_ms && (
                <span className="flex items-center gap-1">
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  {(result.metadata.processing_time_ms / 1000).toFixed(1)}秒
                </span>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="ml-4 rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Result Content */}
      <div className="flex-1 overflow-y-auto bg-gray-50 px-6 py-4">
        <div className="mx-auto max-w-4xl">
          <div className="rounded-lg bg-white p-6 shadow-sm">
            <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-gray-800">
              {result.result}
            </pre>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="border-t border-gray-200 bg-white px-6 py-4">
        <div className="mx-auto flex max-w-4xl gap-3">
          <Button
            variant="primary"
            onClick={handleApply}
            disabled={isApplying}
            className="flex-1"
          >
            {isApplying ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                应用中...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                应用到编辑器
              </span>
            )}
          </Button>
          
          <Button
            variant="secondary"
            onClick={handleCopy}
            className="flex-1"
          >
            {copySuccess ? (
              <span className="flex items-center justify-center gap-2 text-green-600">
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                已复制
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                复制结果
              </span>
            )}
          </Button>
          
          <Button
            variant="secondary"
            onClick={onReprocess}
            className="flex-1"
          >
            <span className="flex items-center justify-center gap-2">
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              重新处理
            </span>
          </Button>
        </div>
      </div>
    </div>
  )
}
