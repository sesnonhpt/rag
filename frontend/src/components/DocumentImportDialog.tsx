/**
 * Document Import Dialog Component
 * 
 * Main dialog that orchestrates the document import and processing workflow
 */

import { useState } from 'react'
import type { UploadedDocument, ProcessingResult } from '@/types/document'
import DocumentImportZone from './DocumentImportZone'
import ProcessingOptionsPanel from './ProcessingOptionsPanel'
import ProcessingResultViewer from './ProcessingResultViewer'
import ProcessingHistoryPanel from './ProcessingHistoryPanel'

interface DocumentImportDialogProps {
  isOpen: boolean
  onClose: () => void
  onApplyToEditor: (content: string) => void
}

type DialogStep = 'upload' | 'options' | 'result' | 'history'

export default function DocumentImportDialog({
  isOpen,
  onClose,
  onApplyToEditor,
}: DocumentImportDialogProps) {
  const [currentStep, setCurrentStep] = useState<DialogStep>('upload')
  const [uploadedDocument, setUploadedDocument] = useState<UploadedDocument | null>(null)
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showToast, setShowToast] = useState(false)
  const [toastMessage, setToastMessage] = useState('')

  // Show toast notification
  const showNotification = (message: string) => {
    setToastMessage(message)
    setShowToast(true)
    setTimeout(() => setShowToast(false), 3000)
  }

  // Handle file uploaded
  const handleFileUploaded = (document: UploadedDocument) => {
    setUploadedDocument(document)
    setCurrentStep('options')
    setError(null)
  }

  // Handle upload error
  const handleUploadError = (errorMessage: string) => {
    setError(errorMessage)
  }

  // Handle processing complete
  const handleProcessingComplete = (result: ProcessingResult) => {
    setProcessingResult(result)
    setCurrentStep('result')
    setError(null)
  }

  // Handle apply to editor
  const handleApplyToEditor = async (content: string) => {
    try {
      await onApplyToEditor(content)
      showNotification('内容已应用到编辑器')
      // Close dialog after a short delay
      setTimeout(() => {
        handleClose()
      }, 1000)
    } catch (err: any) {
      setError(err.message || '应用到编辑器失败')
    }
  }

  // Handle copy
  const handleCopy = () => {
    showNotification('已复制到剪贴板')
  }

  // Handle reprocess
  const handleReprocess = () => {
    setCurrentStep('options')
    setProcessingResult(null)
  }

  // Handle cancel from options panel
  const handleCancelOptions = () => {
    setCurrentStep('upload')
    setUploadedDocument(null)
  }

  // Handle close result viewer
  const handleCloseResult = () => {
    setCurrentStep('upload')
    setUploadedDocument(null)
    setProcessingResult(null)
  }

  // Handle show history
  const handleShowHistory = () => {
    setCurrentStep('history')
  }

  // Handle select history item
  const handleSelectHistory = (processingId: string, result: string) => {
    // Create a mock ProcessingResult from history
    const mockResult: ProcessingResult = {
      processing_id: processingId,
      result: result,
      processing_option: 'history',
      processed_at: new Date().toISOString(),
      metadata: {},
    }
    setProcessingResult(mockResult)
    setCurrentStep('result')
  }

  // Handle clear history
  const handleClearHistory = () => {
    showNotification('历史记录已清空')
  }

  // Handle close dialog
  const handleClose = () => {
    setCurrentStep('upload')
    setUploadedDocument(null)
    setProcessingResult(null)
    setError(null)
    onClose()
  }

  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black bg-opacity-50 transition-opacity"
        onClick={handleClose}
      />

      {/* Dialog */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div
          className="relative w-full max-w-4xl max-h-[90vh] bg-white rounded-lg shadow-xl flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4">
            <div className="flex items-center gap-4">
              <h2 className="text-xl font-semibold text-gray-900">文档导入</h2>
              
              {/* Step indicator */}
              {currentStep !== 'history' && (
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <span className={currentStep === 'upload' ? 'font-medium text-blue-600' : ''}>
                    1. 上传
                  </span>
                  <span className="text-gray-400">→</span>
                  <span className={currentStep === 'options' ? 'font-medium text-blue-600' : ''}>
                    2. 处理
                  </span>
                  <span className="text-gray-400">→</span>
                  <span className={currentStep === 'result' ? 'font-medium text-blue-600' : ''}>
                    3. 应用
                  </span>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2">
              {/* History button */}
              {currentStep === 'upload' && (
                <button
                  onClick={handleShowHistory}
                  className="rounded-lg px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100"
                >
                  查看历史
                </button>
              )}
              
              {/* Close button */}
              <button
                onClick={handleClose}
                className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6">
            {/* Error message */}
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

            {/* Step content */}
            {currentStep === 'upload' && (
              <DocumentImportZone
                onFileUploaded={handleFileUploaded}
                onError={handleUploadError}
              />
            )}

            {currentStep === 'options' && uploadedDocument && (
              <ProcessingOptionsPanel
                document={uploadedDocument}
                onProcess={handleProcessingComplete}
                onCancel={handleCancelOptions}
              />
            )}

            {currentStep === 'result' && processingResult && (
              <div className="h-[60vh]">
                <ProcessingResultViewer
                  result={processingResult}
                  onApplyToEditor={handleApplyToEditor}
                  onCopy={handleCopy}
                  onReprocess={handleReprocess}
                  onClose={handleCloseResult}
                />
              </div>
            )}

            {currentStep === 'history' && (
              <div className="h-[60vh]">
                <ProcessingHistoryPanel
                  onSelectHistory={handleSelectHistory}
                  onClearHistory={handleClearHistory}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Toast notification */}
      {showToast && (
        <div className="fixed bottom-4 right-4 z-50 animate-fade-in">
          <div className="rounded-lg bg-green-500 px-4 py-3 text-white shadow-lg">
            <div className="flex items-center gap-2">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-sm font-medium">{toastMessage}</span>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
