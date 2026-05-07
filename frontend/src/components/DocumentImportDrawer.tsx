/**
 * Document Import Drawer Component
 * 
 * Main drawer component for document import and processing workflow
 */

import { useState } from 'react'
import Drawer from './ui/Drawer'
import DocumentUploadSection from './DocumentUploadSection'
import ProcessingOptionsSection from './ProcessingOptionsSection'
import ProcessingResultSection from './ProcessingResultSection'
import ProcessingHistorySection from './ProcessingHistorySection'
import type { UploadedDocument, ProcessingResult } from '@/types/document'

interface DocumentImportDrawerProps {
  open: boolean
  onClose: () => void
  onApplyToEditor: (content: string) => void
}

type WorkflowStep = 'upload' | 'options' | 'result' | 'history'

export default function DocumentImportDrawer({
  open,
  onClose,
  onApplyToEditor,
}: DocumentImportDrawerProps) {
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('upload')
  const [uploadedDocuments, setUploadedDocuments] = useState<UploadedDocument[]>([])
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null)

  // Handle document upload success
  const handleDocumentUploaded = (document: UploadedDocument) => {
    setUploadedDocuments(prev => [...prev, document])
  }

  // Handle remove document
  const handleRemoveDocument = (documentId: string) => {
    setUploadedDocuments(prev => prev.filter(doc => doc.document_id !== documentId))
  }

  // Handle continue to processing options
  const handleContinueToOptions = () => {
    if (uploadedDocuments.length > 0) {
      setCurrentStep('options')
    }
  }

  // Handle processing complete
  const handleProcessingComplete = (result: ProcessingResult) => {
    setProcessingResult(result)
    setCurrentStep('result')
  }

  // Handle apply to editor
  const handleApply = (content: string) => {
    onApplyToEditor(content)
    // Don't close drawer - let user continue working
  }

  // Handle view history
  const handleViewHistory = () => {
    setCurrentStep('history')
  }

  // Handle back navigation
  const handleBack = () => {
    if (currentStep === 'options') {
      setCurrentStep('upload')
    } else if (currentStep === 'result') {
      setCurrentStep('options')
    } else if (currentStep === 'history') {
      setCurrentStep('upload')
    }
  }

  // Handle close and reset
  const handleClose = () => {
    setCurrentStep('upload')
    setUploadedDocuments([])
    setProcessingResult(null)
    onClose()
  }

  return (
    <Drawer
      open={open}
      onClose={handleClose}
      title="文档导入与智能处理"
      width="xl"
    >
      <div className="h-full flex flex-col">
        {/* Step Indicator */}
        <div className="border-b border-gray-200 px-6 py-4">
          <div className="flex items-center gap-2">
            <StepIndicator
              step={1}
              label="上传文档"
              active={currentStep === 'upload'}
              completed={uploadedDocuments.length > 0}
            />
            <div className="h-px flex-1 bg-gray-200" />
            <StepIndicator
              step={2}
              label="选择处理方式"
              active={currentStep === 'options'}
              completed={currentStep === 'result'}
            />
            <div className="h-px flex-1 bg-gray-200" />
            <StepIndicator
              step={3}
              label="查看结果"
              active={currentStep === 'result'}
              completed={false}
            />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {currentStep === 'upload' && (
            <DocumentUploadSection
              uploadedDocuments={uploadedDocuments}
              onDocumentUploaded={handleDocumentUploaded}
              onRemoveDocument={handleRemoveDocument}
              onContinue={handleContinueToOptions}
              onViewHistory={handleViewHistory}
            />
          )}

          {currentStep === 'options' && (
            <ProcessingOptionsSection
              documents={uploadedDocuments}
              onProcessingComplete={handleProcessingComplete}
              onBack={handleBack}
            />
          )}

          {currentStep === 'result' && processingResult && (
            <ProcessingResultSection
              result={processingResult}
              onApplyToEditor={handleApply}
              onReprocess={handleBack}
              onClose={handleClose}
            />
          )}

          {currentStep === 'history' && (
            <ProcessingHistorySection
              onSelectHistory={(result) => {
                setProcessingResult(result)
                setCurrentStep('result')
              }}
              onBack={handleBack}
            />
          )}
        </div>
      </div>
    </Drawer>
  )
}

// Step Indicator Component
function StepIndicator({
  step,
  label,
  active,
  completed,
}: {
  step: number
  label: string
  active: boolean
  completed: boolean
}) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={`
          flex h-8 w-8 items-center justify-center rounded-full text-sm font-semibold
          ${completed ? 'bg-green-500 text-white' : active ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-600'}
        `}
      >
        {completed ? (
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        ) : (
          step
        )}
      </div>
      <span className={`text-sm font-medium ${active ? 'text-gray-900' : 'text-gray-500'}`}>
        {label}
      </span>
    </div>
  )
}
