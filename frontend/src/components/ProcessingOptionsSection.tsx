/**
 * Processing Options Section Component
 */

import { useState } from 'react'
import { documentApi } from '@/api/document'
import type { UploadedDocument, ProcessingResult } from '@/types/document'
import { ProcessingOptionLabels, ProcessingOptionDescriptions } from '@/types/document'
import Button from './ui/Button'
import Card from './ui/Card'

interface ProcessingOptionsSectionProps {
  documents: UploadedDocument[]
  onProcessingComplete: (result: ProcessingResult) => void
  onBack: () => void
}

export default function ProcessingOptionsSection({
  documents,
  onProcessingComplete,
  onBack,
}: ProcessingOptionsSectionProps) {
  const [selectedOption, setSelectedOption] = useState<string | null>(null)
  const [customPrompt, setCustomPrompt] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const options = [
    'extract_exercises',
    'summarize',
    'extract_teaching_thoughts',
    'custom',
  ]

  const handleProcess = async () => {
    if (!selectedOption) return

    // Validate custom prompt
    if (selectedOption === 'custom' && (!customPrompt.trim() || customPrompt.length < 10)) {
      setError('自定义指令至少需要10个字符')
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      // Process first document (or combine multiple documents)
      const document = documents[0]
      const result = await documentApi.processDocument({
        document_id: document.document_id,
        processing_option: selectedOption,
        custom_prompt: selectedOption === 'custom' ? customPrompt : undefined,
      })

      onProcessingComplete(result)
    } catch (err: any) {
      setError(err.message || 'AI处理失败')
      setIsProcessing(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Documents Summary */}
      <Card>
        <h3 className="text-sm font-semibold text-gray-900 mb-3">
          待处理文档 ({documents.length})
        </h3>
        <div className="space-y-2">
          {documents.map((doc) => (
            <div key={doc.document_id} className="flex items-center gap-2 text-sm">
              <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span className="text-gray-700 truncate">{doc.filename}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Processing Options */}
      <div>
        <h3 className="text-base font-semibold text-gray-900 mb-4">选择处理方式</h3>
        <div className="space-y-3">
          {options.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => !isProcessing && setSelectedOption(option)}
              disabled={isProcessing}
              className={`w-full text-left rounded-lg border bg-white p-4 transition-all ${
                selectedOption === option
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
              } ${isProcessing ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
            >
              <div className="flex items-start gap-3">
                <div className={`mt-0.5 h-5 w-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                  selectedOption === option ? 'border-blue-500' : 'border-gray-300'
                }`}>
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
        <Card>
          <label className="block text-sm font-medium text-gray-900 mb-2">
            自定义处理指令
          </label>
          <textarea
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            disabled={isProcessing}
            placeholder="请描述您希望如何处理这份文档..."
            className="w-full rounded-lg border border-gray-300 p-3 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            rows={4}
            maxLength={500}
          />
          <div className="mt-2 flex justify-between text-xs text-gray-500">
            <span>10-500个字符</span>
            <span>{customPrompt.length}/500</span>
          </div>
        </Card>
      )}

      {/* Error Message */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <div className="flex items-start gap-3">
            <svg className="h-5 w-5 text-red-500 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </Card>
      )}

      {/* Processing Info */}
      {isProcessing && (
        <Card className="border-blue-200 bg-blue-50">
          <div className="flex items-start gap-3">
            <svg className="h-5 w-5 text-blue-500 mt-0.5 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="text-sm font-medium text-blue-900">AI正在处理您的文档</p>
            </div>
          </div>
        </Card>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button variant="secondary" onClick={onBack} disabled={isProcessing} className="flex-1">
          返回
        </Button>
        <Button
          variant="primary"
          onClick={handleProcess}
          disabled={!selectedOption || isProcessing}
          className="flex-1"
        >
          {isProcessing ? '处理中...' : '开始处理'}
        </Button>
      </div>
    </div>
  )
}
