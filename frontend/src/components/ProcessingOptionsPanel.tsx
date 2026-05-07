/**
 * Processing Options Panel Component
 * 
 * Allows users to select document processing options
 */

import { useState } from 'react'
import { documentApi } from '@/api/document'
import type { 
  UploadedDocument, 
  ProcessingResult 
} from '@/types/document'
import { 
  ProcessingOptionLabels, 
  ProcessingOptionDescriptions 
} from '@/types/document'
import Button from './ui/Button'

interface ProcessingOptionsPanelProps {
  document: UploadedDocument
  onProcess: (result: ProcessingResult) => void
  onCancel: () => void
  recentPrompts?: string[]
}

export default function ProcessingOptionsPanel({
  document,
  onProcess,
  onCancel,
  recentPrompts = [],
}: ProcessingOptionsPanelProps) {
  const [selectedOption, setSelectedOption] = useState<string | null>(null)
  const [customPrompt, setCustomPrompt] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Processing options
  const options = [
    'extract_exercises' as const,
    'summarize' as const,
    'extract_teaching_thoughts' as const,
    'custom' as const,
  ]

  // Handle option selection
  const handleOptionSelect = (option: string) => {
    setSelectedOption(option)
    setError(null)
    
    // Clear custom prompt if switching away from custom option
    if (option !== 'custom') {
      setCustomPrompt('')
    }
  }

  // Handle process button click
  const handleProcess = async () => {
    if (!selectedOption) {
      setError('请选择一个处理选项')
      return
    }

    // Validate custom prompt
    if (selectedOption === 'custom') {
      if (!customPrompt.trim()) {
        setError('请输入自定义处理指令')
        return
      }
      if (customPrompt.length < 10) {
        setError('自定义指令至少需要10个字符')
        return
      }
      if (customPrompt.length > 500) {
        setError('自定义指令不能超过500个字符')
        return
      }
    }

    setIsProcessing(true)
    setError(null)

    try {
      const result = await documentApi.processDocument({
        document_id: document.document_id,
        processing_option: selectedOption,
        custom_prompt: selectedOption === 'custom' ? customPrompt : undefined,
      })

      onProcess(result)
    } catch (err: any) {
      setError(err.message || 'AI处理失败,请稍后重试')
      setIsProcessing(false)
    }
  }

  // Handle recent prompt selection
  const handleRecentPromptSelect = (prompt: string) => {
    setCustomPrompt(prompt)
  }

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Document Info */}
      <div className="mb-6 rounded-lg bg-gray-50 p-4">
        <h3 className="mb-2 text-sm font-medium text-gray-700">已上传文档</h3>
        <div className="flex items-center gap-3">
          <svg className="h-8 w-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <div className="flex-1 min-w-0">
            <p className="truncate font-medium text-gray-900">{document.filename}</p>
            <p className="text-sm text-gray-500">
              {(document.file_size / 1024).toFixed(1)} KB · {document.metadata.word_count || 0} 字
            </p>
          </div>
        </div>
      </div>

      {/* Processing Options */}
      <div className="mb-6">
        <h3 className="mb-4 text-lg font-semibold text-gray-900">选择处理方式</h3>
        <div className="space-y-3">
          {options.map((option) => (
            <button
              key={option}
              onClick={() => handleOptionSelect(option)}
              disabled={isProcessing}
              className={`
                w-full rounded-lg border-2 p-4 text-left transition-all
                ${selectedOption === option
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
                }
                ${isProcessing ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}
              `}
            >
              <div className="flex items-start gap-3">
                <div className={`
                  mt-0.5 h-5 w-5 rounded-full border-2 flex items-center justify-center
                  ${selectedOption === option ? 'border-blue-500' : 'border-gray-300'}
                `}>
                  {selectedOption === option && (
                    <div className="h-3 w-3 rounded-full bg-blue-500" />
                  )}
                </div>
                <div className="flex-1">
                  <p className="font-medium text-gray-900">
                    {ProcessingOptionLabels[option]}
                  </p>
                  <p className="mt-1 text-sm text-gray-600">
                    {ProcessingOptionDescriptions[option]}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Custom Prompt Input */}
      {selectedOption === 'custom' && (
        <div className="mb-6">
          <label className="mb-2 block text-sm font-medium text-gray-700">
            自定义处理指令
          </label>
          <textarea
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            disabled={isProcessing}
            placeholder="请描述您希望如何处理这份文档...&#10;&#10;例如:&#10;- 提取文档中的关键词和核心概念&#10;- 总结每个章节的主要内容&#10;- 列出文档中提到的所有人物和事件"
            className="w-full rounded-lg border border-gray-300 p-3 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            rows={6}
            maxLength={500}
          />
          <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
            <span>10-500个字符</span>
            <span className={customPrompt.length > 500 ? 'text-red-500' : ''}>
              {customPrompt.length}/500
            </span>
          </div>

          {/* Recent Prompts */}
          {recentPrompts.length > 0 && (
            <div className="mt-4">
              <p className="mb-2 text-sm font-medium text-gray-700">最近使用的指令</p>
              <div className="space-y-2">
                {recentPrompts.slice(0, 3).map((prompt, index) => (
                  <button
                    key={index}
                    onClick={() => handleRecentPromptSelect(prompt)}
                    disabled={isProcessing}
                    className="w-full rounded border border-gray-200 bg-white p-2 text-left text-sm text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {prompt.length > 80 ? `${prompt.slice(0, 80)}...` : prompt}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mb-4 rounded-lg bg-red-50 p-4">
          <div className="flex items-start gap-3">
            <svg className="h-5 w-5 text-red-500 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button
          variant="secondary"
          onClick={onCancel}
          disabled={isProcessing}
          className="flex-1"
        >
          取消
        </Button>
        <Button
          variant="primary"
          onClick={handleProcess}
          disabled={!selectedOption || isProcessing}
          className="flex-1"
        >
          {isProcessing ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              处理中...
            </span>
          ) : (
            '开始处理'
          )}
        </Button>
      </div>

      {/* Processing Info */}
      {isProcessing && (
        <div className="mt-4 rounded-lg bg-blue-50 p-4">
          <div className="flex items-start gap-3">
            <svg className="h-5 w-5 text-blue-500 mt-0.5 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div className="flex-1">
              <p className="text-sm font-medium text-blue-900">AI正在处理您的文档</p>
              
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
