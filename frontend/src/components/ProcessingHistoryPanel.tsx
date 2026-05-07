/**
 * Processing History Panel Component
 * 
 * Displays processing history with ability to view and reuse results
 */

import { useState, useEffect } from 'react'
import { documentApi } from '@/api/document'
import type { ProcessingHistoryItem } from '@/types/document'
import { ProcessingOptionLabels } from '@/types/document'
import Button from './ui/Button'

interface ProcessingHistoryPanelProps {
  onSelectHistory: (processingId: string, result: string) => void
  onClearHistory: () => void
}

export default function ProcessingHistoryPanel({
  onSelectHistory,
  onClearHistory,
}: ProcessingHistoryPanelProps) {
  const [historyItems, setHistoryItems] = useState<ProcessingHistoryItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadingResultId, setLoadingResultId] = useState<string | null>(null)

  // Load history on mount
  useEffect(() => {
    loadHistory()
  }, [])

  // Load history from API
  const loadHistory = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await documentApi.getProcessingHistory(10, 0)
      setHistoryItems(response.items)
    } catch (err: any) {
      setError(err.message || '加载历史记录失败')
    } finally {
      setIsLoading(false)
    }
  }

  // Handle history item click
  const handleHistoryClick = async (item: ProcessingHistoryItem) => {
    setLoadingResultId(item.processing_id)
    
    try {
      const response = await documentApi.getProcessingResult(item.processing_id)
      onSelectHistory(item.processing_id, response.result)
    } catch (err: any) {
      setError(err.message || '加载处理结果失败')
    } finally {
      setLoadingResultId(null)
    }
  }

  // Handle clear history
  const handleClearHistory = async () => {
    if (!confirm('确定要清空所有处理历史吗?此操作不可恢复。')) {
      return
    }

    try {
      await documentApi.clearProcessingHistory()
      setHistoryItems([])
      onClearHistory()
    } catch (err: any) {
      setError(err.message || '清空历史记录失败')
    }
  }

  // Format date
  const formatDate = (isoString: string): string => {
    const date = new Date(isoString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return '刚刚'
    if (diffMins < 60) return `${diffMins}分钟前`
    if (diffHours < 24) return `${diffHours}小时前`
    if (diffDays < 7) return `${diffDays}天前`
    
    return date.toLocaleDateString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  // Get processing option label
  const getOptionLabel = (option: string): string => {
    return ProcessingOptionLabels[option as keyof typeof ProcessingOptionLabels] || option
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">处理历史</h2>
            <p className="mt-1 text-sm text-gray-600">
              最近{historyItems.length}条处理记录
            </p>
          </div>
          {historyItems.length > 0 && (
            <Button
              variant="secondary"
              size="sm"
              onClick={handleClearHistory}
            >
              清空历史
            </Button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto bg-gray-50 px-6 py-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <svg className="mx-auto h-12 w-12 animate-spin text-gray-400" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <p className="mt-4 text-sm text-gray-600">加载中...</p>
            </div>
          </div>
        ) : error ? (
          <div className="rounded-lg bg-red-50 p-4">
            <div className="flex items-start gap-3">
              <svg className="h-5 w-5 text-red-500 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div className="flex-1">
                <p className="text-sm text-red-700">{error}</p>
                <button
                  onClick={loadHistory}
                  className="mt-2 text-sm font-medium text-red-600 hover:text-red-700"
                >
                  重试
                </button>
              </div>
            </div>
          </div>
        ) : historyItems.length === 0 ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <svg className="mx-auto h-16 w-16 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="mt-4 text-sm font-medium text-gray-900">暂无处理历史</p>
              <p className="mt-1 text-sm text-gray-500">上传文档并处理后,历史记录会显示在这里</p>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {historyItems.map((item) => (
              <button
                key={item.processing_id}
                onClick={() => handleHistoryClick(item)}
                disabled={loadingResultId === item.processing_id}
                className="w-full rounded-lg border border-gray-200 bg-white p-4 text-left transition-all hover:border-blue-300 hover:shadow-md disabled:cursor-not-allowed disabled:opacity-50"
              >
                <div className="flex items-start gap-3">
                  {/* Icon */}
                  <div className="mt-1 flex-shrink-0">
                    {loadingResultId === item.processing_id ? (
                      <svg className="h-5 w-5 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                    ) : (
                      <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    {/* Filename and option */}
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="truncate font-medium text-gray-900">
                        {item.document_filename}
                      </p>
                      <span className="inline-flex items-center rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800">
                        {getOptionLabel(item.processing_option)}
                      </span>
                    </div>

                    {/* Custom prompt if exists */}
                    {item.custom_prompt && (
                      <p className="mt-1 text-xs text-gray-600 line-clamp-1">
                        指令: {item.custom_prompt}
                      </p>
                    )}

                    {/* Result preview */}
                    <p className="mt-2 text-sm text-gray-600 line-clamp-2">
                      {item.result_preview}
                    </p>

                    {/* Time */}
                    <p className="mt-2 text-xs text-gray-500">
                      {formatDate(item.processed_at)}
                    </p>
                  </div>

                  {/* Arrow icon */}
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
