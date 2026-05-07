/**
 * Processing Result Section Component
 * 
 * Displays processing result with markdown rendering and export options
 */

import { useState } from 'react'
import { renderMarkdown } from '@/utils/markdown'
import { ProcessingOptionLabels } from '@/types/document'
import type { ProcessingResult } from '@/types/document'
import Button from './ui/Button'
import Card from './ui/Card'

interface ProcessingResultSectionProps {
  result: ProcessingResult
  onApplyToEditor: (content: string) => void
  onReprocess: () => void
  onClose?: () => void
}

export default function ProcessingResultSection({
  result,
  onApplyToEditor,
  onReprocess,
}: ProcessingResultSectionProps) {
  const [copied, setCopied] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [applied, setApplied] = useState(false)

  // Convert markdown to plain text (remove markdown syntax)
  const markdownToPlainText = (markdown: string): string => {
    return markdown
      // Remove bold/italic markers
      .replace(/\*\*\*(.+?)\*\*\*/g, '$1')  // ***text***
      .replace(/\*\*(.+?)\*\*/g, '$1')      // **text**
      .replace(/\*(.+?)\*/g, '$1')          // *text*
      .replace(/__(.+?)__/g, '$1')          // __text__
      .replace(/_(.+?)_/g, '$1')            // _text_
      // Remove headers
      .replace(/^#{1,6}\s+/gm, '')
      // Remove links but keep text
      .replace(/\[(.+?)\]\(.+?\)/g, '$1')
      // Remove images
      .replace(/!\[.*?\]\(.+?\)/g, '')
      // Remove code blocks
      .replace(/```[\s\S]*?```/g, '')
      .replace(/`(.+?)`/g, '$1')
      // Remove blockquotes
      .replace(/^>\s+/gm, '')
      // Remove horizontal rules
      .replace(/^[-*_]{3,}$/gm, '')
      // Clean up extra whitespace
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  }

  // Handle copy to clipboard
  const handleCopy = async () => {
    try {
      // Copy plain text without markdown syntax
      const plainText = markdownToPlainText(result.result)
      
      // Try modern clipboard API first
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(plainText)
      } else {
        // Fallback for older browsers or non-HTTPS contexts
        const textArea = document.createElement('textarea')
        textArea.value = plainText
        textArea.style.position = 'fixed'
        textArea.style.left = '-999999px'
        textArea.style.top = '-999999px'
        document.body.appendChild(textArea)
        textArea.focus()
        textArea.select()
        try {
          document.execCommand('copy')
        } finally {
          document.body.removeChild(textArea)
        }
      }
      
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
      alert('复制失败，请手动选择文本复制')
    }
  }

  // Handle apply to editor
  const handleApply = () => {
    onApplyToEditor(result.result)
    setApplied(true)
    setTimeout(() => setApplied(false), 2000)
  }

  // Handle export to Word
  const handleExportWord = async () => {
    setExporting(true)
    try {
      // Convert markdown to HTML (only body content, no wrapper)
      const htmlContent = renderMarkdown(result.result)
      
      // Call backend API to export as DOCX
      const response = await fetch('/api/documents/export-docx', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content_html: htmlContent,  // Send only the body HTML, not full document
          title: result.metadata.document_filename?.replace(/\.[^/.]+$/, '') || '处理结果',
        }),
      })
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: '导出失败' }))
        throw new Error(errorData.detail || '导出失败')
      }
      
      // Download the file
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${result.metadata.document_filename?.replace(/\.[^/.]+$/, '') || '处理结果'}_${result.processing_option}.docx`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err: any) {
      console.error('Export failed:', err)
      alert(`导出失败: ${err.message}`)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Result Header */}
      <Card>
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-base font-semibold text-gray-900">
              {ProcessingOptionLabels[result.processing_option] || '处理结果'}
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              {result.metadata.document_filename} · {new Date(result.processed_at).toLocaleString('zh-CN')}
            </p>
            {result.metadata.processing_time_ms && (
              <p className="mt-1 text-xs text-gray-400">
                处理耗时: {(result.metadata.processing_time_ms / 1000).toFixed(1)}秒
              </p>
            )}
          </div>
        </div>
        
        {/* Action Buttons - More Prominent */}
        <div className="flex gap-3">
          <button
            onClick={handleCopy}
            className="flex-1 flex items-center justify-center gap-2 rounded-lg border-2 border-gray-300 bg-white px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 hover:border-gray-400 transition-all"
          >
            {copied ? (
              <>
                <svg className="h-5 w-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-green-600">已复制</span>
              </>
            ) : (
              <>
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>复制结果</span>
              </>
            )}
          </button>
          <button
            onClick={handleExportWord}
            disabled={exporting}
            className="flex-1 flex items-center justify-center gap-2 rounded-lg border-2 border-blue-500 bg-blue-50 px-4 py-3 text-sm font-medium text-blue-700 hover:bg-blue-100 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span>{exporting ? '导出中...' : '导出Word'}</span>
          </button>
        </div>
      </Card>

      {/* Result Content with Markdown Rendering */}
      <Card className="max-h-[500px] overflow-y-auto">
        <div
          className="prose prose-sm max-w-none
            [&_h1]:text-xl [&_h1]:font-bold [&_h1]:mb-3 [&_h1]:mt-4
            [&_h2]:text-lg [&_h2]:font-semibold [&_h2]:mb-2 [&_h2]:mt-3
            [&_h3]:text-base [&_h3]:font-semibold [&_h3]:mb-2 [&_h3]:mt-2
            [&_p]:my-2 [&_p]:text-gray-700 [&_p]:leading-relaxed
            [&_ul]:my-2 [&_ul]:pl-6 [&_ul]:list-disc
            [&_ol]:my-2 [&_ol]:pl-6 [&_ol]:list-decimal
            [&_li]:my-1 [&_li]:text-gray-700
            [&_strong]:font-semibold [&_strong]:text-gray-900
            [&_code]:bg-gray-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-sm
            [&_pre]:bg-gray-100 [&_pre]:p-3 [&_pre]:rounded-lg [&_pre]:overflow-x-auto
            [&_blockquote]:border-l-4 [&_blockquote]:border-gray-300 [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:text-gray-600"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(result.result) }}
        />
      </Card>

      {/* Custom Prompt Display */}
      {result.metadata.custom_prompt && (
        <Card className="border-blue-200 bg-blue-50">
          <p className="text-xs font-medium text-blue-900 mb-1">使用的自定义指令:</p>
          <p className="text-sm text-blue-700">{result.metadata.custom_prompt}</p>
        </Card>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button variant="secondary" onClick={onReprocess} className="flex-1">
          <span className="flex items-center justify-center gap-2">
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            重新处理
          </span>
        </Button>
        <Button variant="primary" onClick={handleApply} className="flex-1">
          <span className="flex items-center justify-center gap-2">
            {applied ? (
              <>
                <svg className="h-4 w-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                已应用
              </>
            ) : (
              <>
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                应用到编辑器
              </>
            )}
          </span>
        </Button>
      </div>
    </div>
  )
}
