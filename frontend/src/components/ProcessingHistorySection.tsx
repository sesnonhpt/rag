/**
 * Processing History Section Component
 */

import { useState, useEffect } from 'react'
import { documentApi } from '@/api/document'
import { ProcessingOptionLabels } from '@/types/document'
import type { ProcessingHistoryItem, ProcessingResult } from '@/types/document'
import Button from './ui/Button'
import Card from './ui/Card'

interface ProcessingHistorySectionProps {
  onSelectHistory: (result: ProcessingResult) => void
  onBack: () => void
}

export default function ProcessingHistorySection({
  onSelectHistory,
  onBack,
}: ProcessingHistorySectionProps) {
  const [history, setHistory] = useState<ProcessingHistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadHistory()
  }, [])

  const loadHistory = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await documentApi.getProcessingHistory(10, 0)
      setHistory(response.items)
    } catch (err: any) {
      setError(err.message || '加载历史记录失败')
    } finally {
      setLoading(false)
    }
  }

  const handleSelectHistory = async (item: ProcessingHistoryItem) => {
    try {
      const response = await documentApi.getProcessingResult(item.processing_id)
      
      // Convert to ProcessingResult format
      const result: ProcessingResult = {
        processing_id: item.processing_id,
        result: response.result,
        processing_option: item.processing_option,
        processed_at: item.processed_at,
        metadata: {
          document_filename: item.document_filename,
          custom_prompt: item.custom_prompt,
        },
      }
      
      onSelectHistory(result)
    } catch (err: any) {
      setError(err.message || '加载历史记录详情失败')
    }
  }

  const handleClearHistory = async () => {
    if (!confirm('确定要清空所有历史记录吗？')) return

    try {
      await documentApi.clearProcessingHistory()
      setHistory([])
    } catch (err: any) {
      setError(err.message || '清空历史记录失败')
    }
  }

  const formatTime = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)

    if (minutes < 1) return '刚刚'
    if (minutes < 60) return `${minutes}分钟前`
    if (hours < 24) return `${hours}小时前`
    if (days < 7) return `${days}天前`
    return date.toLocaleDateString('zh-CN')
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <svg className="h-8 w-8 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-gray-900">
          处理历史 ({history.length})
        </h3>
        {history.length > 0 && (
          <Button variant="danger" size="sm" onClick={handleClearHistory}>
            清空历史
          </Button>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <p className="text-sm text-red-700">{error}</p>
        </Card>
      )}

      {/* History List */}
      {history.length === 0 ? (
        <Card className="text-center py-12">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="mt-4 text-sm text-gray-500">暂无处理历史</p>
        </Card>
      ) : (
        <div className="space-y-3">
          {history.map((item) => (
            <div
              key={item.processing_id}
              className="cursor-pointer"
              onClick={() => handleSelectHistory(item)}
            >
              <Card className="hover:shadow-md transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0">
                  <svg className="h-10 w-10 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="font-medium text-gray-900 truncate">{item.document_filename}</p>
                    <span className="inline-flex items-center rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800">
                      {ProcessingOptionLabels[item.processing_option]}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-gray-500">{formatTime(item.processed_at)}</p>
                  <p className="mt-2 text-sm text-gray-600 line-clamp-2">{item.result_preview}</p>
                  {item.custom_prompt && (
                    <p className="mt-1 text-xs text-blue-600 italic line-clamp-1">
                      指令: {item.custom_prompt}
                    </p>
                  )}
                </div>
                <svg className="h-5 w-5 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </Card>
            </div>
          ))}
        </div>
      )}

      {/* Back Button */}
      <Button variant="secondary" onClick={onBack} className="w-full">
        返回
      </Button>
    </div>
  )
}
