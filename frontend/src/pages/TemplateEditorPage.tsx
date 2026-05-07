import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Button from '@/components/ui/Button'
import Loading from '@/components/ui/Loading'
import TeachingThoughtsTimeline from '@/components/TeachingThoughtsTimeline'
import ContentEnhancer from '@/components/ContentEnhancer'
import LessonDeepAnalysisPanel from '@/components/LessonDeepAnalysisPanel'
import DocumentImportDrawer from '@/components/DocumentImportDrawer'
import type {
  TemplateContent,
  TeachingThought,
  LessonAnalysis,
} from '@/types/template'
import { templateApi } from '@/api/template'
import { renderMarkdown } from '@/utils/markdown'

declare global {
  interface Window {
    Quill: any
  }
}

export default function TemplateEditorPage() {
  const params = useParams<{ '*': string }>()
  const navigate = useNavigate()
  const filename = params['*'] || ''

  const [loading, setLoading] = useState(true)
  const [content, setContent] = useState<TemplateContent | null>(null)
  const [aiInstruction, setAiInstruction] = useState('')
  const [aiLoading, setAiLoading] = useState(false)
  const [selectedText, setSelectedText] = useState('')
  const [selectionRange, setSelectionRange] = useState<any>(null)
  const [aiResult, setAiResult] = useState('')
  const [error, setError] = useState('')
  const [showPopover, setShowPopover] = useState(false)
  const [popoverPosition, setPopoverPosition] = useState({ top: 0, left: 0 })
  const [showContextMenu, setShowContextMenu] = useState(false)
  const [contextMenuPosition, setContextMenuPosition] = useState({ x: 0, y: 0 })
  const [imagePrompt, setImagePrompt] = useState('')
  const [imageLoading, setImageLoading] = useState(false)
  const [showImageDialog, setShowImageDialog] = useState(false)

  const [teachingThoughts, setTeachingThoughts] = useState<TeachingThought[]>([])
  const [lessonAnalysis, setLessonAnalysis] = useState<LessonAnalysis | null>(null)
  const [showDocumentImportDrawer, setShowDocumentImportDrawer] = useState(false)
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const editorRef = useRef<HTMLDivElement | null>(null)
  const quillRef = useRef<any>(null)
  const popoverRef = useRef<HTMLDivElement>(null)
  const contextMenuRef = useRef<HTMLDivElement>(null)
  const [quillLoaded, setQuillLoaded] = useState(false)
  const [editorMounted, setEditorMounted] = useState(false)
  useEffect(() => {
    if (window.Quill) {
      setQuillLoaded(true)
      return
    }

    const link = document.createElement('link')
    link.href = 'https://cdn.quilljs.com/1.3.7/quill.snow.css'
    link.rel = 'stylesheet'
    document.head.appendChild(link)

    const script = document.createElement('script')
    script.src = 'https://cdn.quilljs.com/1.3.7/quill.min.js'
    script.onload = () => setQuillLoaded(true)
    document.head.appendChild(script)
  }, [])

  useEffect(() => {
    if (!editorRef.current || !quillLoaded || !window.Quill || quillRef.current || !editorMounted) {
      console.log('Quill init check:', {
        hasEditorRef: !!editorRef.current,
        quillLoaded,
        hasWindowQuill: !!window.Quill,
        hasQuillRef: !!quillRef.current,
        editorMounted
      })
      return
    }

    console.log('Initializing Quill editor...')
    const quill = new window.Quill(editorRef.current, {
      theme: 'snow',
      modules: {
        toolbar: [
          [{ header: [1, 2, 3, false] }],
          ['bold', 'italic', 'underline', 'strike'],
          [{ list: 'ordered' }, { list: 'bullet' }],
          [{ indent: '-1' }, { indent: '+1' }],
          ['link', 'image'],
          ['clean'],
        ],
      },
      placeholder: '在这里编辑模板内容，或者把右侧生成的第一版教学设计写入这里...',
    })

    quillRef.current = quill
    console.log('Quill initialized successfully')
    
    if (content?.content_html) {
      console.log('Loading content into Quill')
      quill.root.innerHTML = content.content_html
    }

    quill.on('selection-change', (range: any) => {
      if (range && range.length > 0) {
        const text = quill.getText(range.index, range.length)
        setSelectedText(text.trim())
        setSelectionRange(range)

        const bounds = quill.getBounds(range.index, range.length)
        const editorRect = editorRef.current?.getBoundingClientRect()
        if (editorRect) {
          setPopoverPosition({
            top: bounds.top + bounds.height + 18,
            left: Math.max(bounds.left, 16),
          })
        }
      } else if (!popoverRef.current?.contains(document.activeElement)) {
        setSelectedText('')
        setSelectionRange(null)
      }
    })
  }, [quillLoaded, content, editorMounted])

  useEffect(() => {
    if (!filename) return
    loadContent()
  }, [filename])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        showPopover &&
        popoverRef.current &&
        !popoverRef.current.contains(event.target as Node) &&
        editorRef.current &&
        !editorRef.current.contains(event.target as Node)
      ) {
        handleClosePopover()
      }

      if (
        showContextMenu &&
        contextMenuRef.current &&
        !contextMenuRef.current.contains(event.target as Node)
      ) {
        setShowContextMenu(false)
      }

      // Close export menu when clicking outside
      if (showExportMenu && !(event.target as Element).closest('.relative')) {
        setShowExportMenu(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showPopover, showContextMenu, showExportMenu])

  useEffect(() => {
    const handleContextMenu = (e: MouseEvent) => {
      if (editorRef.current && editorRef.current.contains(e.target as Node)) {
        e.preventDefault()
        setContextMenuPosition({ x: e.clientX, y: e.clientY })
        setShowContextMenu(true)
      }
    }

    document.addEventListener('contextmenu', handleContextMenu)
    return () => document.removeEventListener('contextmenu', handleContextMenu)
  }, [])

  const loadContent = async () => {
    if (!filename) return

    try {
      setLoading(true)
      setError('')
      const data = await templateApi.getContent(filename)
      setContent(data)
      if (quillRef.current && data.content_html) {
        quillRef.current.root.innerHTML = data.content_html
      }
    } catch (err: any) {
      setError(err.message || '加载失败')
    } finally {
      setLoading(false)
    }
    
    // 不再自动加载备课思路，改为手动触发
  }

  const handleStartAnalysis = async () => {
    if (!filename || isAnalyzing) return
    
    setIsAnalyzing(true)
    try {
      const data = await templateApi.getLegacyGuidePilotPackage(filename)
      if (data.success) {
        setLessonAnalysis(data.analysis)
        setTeachingThoughts(data.thoughts || [])
      }
    } catch (err: any) {
      console.error('生成导学案拆解失败:', err)
      alert('分析失败，请稍后重试')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleAIModify = async () => {
    if (!selectedText || !aiInstruction.trim()) {
      alert('请先选中文本，并输入修改要求。')
      return
    }

    try {
      setAiLoading(true)
      setAiResult('')
      const result = await templateApi.aiModify(selectedText, aiInstruction)
      setAiResult(result.modified_text)
    } catch (err: any) {
      alert(`AI 修改失败: ${err.message}`)
    } finally {
      setAiLoading(false)
    }
  }

  const handleApplyAI = () => {
    if (!aiResult || !quillRef.current || !selectionRange) return
    quillRef.current.deleteText(selectionRange.index, selectionRange.length)
    quillRef.current.insertText(selectionRange.index, aiResult)
    setAiResult('')
    setAiInstruction('')
    setSelectedText('')
    setSelectionRange(null)
    setShowPopover(false)
  }

  const handleCancelAI = () => {
    setAiResult('')
    setAiInstruction('')
    setShowPopover(false)
  }

  const handleClosePopover = () => {
    setShowPopover(false)
    setSelectedText('')
    setSelectionRange(null)
    setAiInstruction('')
    setAiResult('')
  }

  const handleOpenTextEnhance = () => {
    setShowContextMenu(false)
    if (selectedText) {
      setShowPopover(true)
      return
    }
    alert('请先在左侧选中需要优化的内容。')
  }

  const handleOpenImageGeneration = () => {
    setShowContextMenu(false)
    setImagePrompt(selectedText || '')
    setShowImageDialog(true)
  }

  const handleCloseImageDialog = () => {
    setShowImageDialog(false)
    setImagePrompt('')
  }

  const generateImage = async (prompt: string) => {
    if (!quillRef.current || !prompt.trim()) return

    try {
      setImageLoading(true)
      await new Promise((resolve) => setTimeout(resolve, 2000))
      const imageUrl = `https://via.placeholder.com/400x300?text=${encodeURIComponent(prompt)}`
      const range = quillRef.current.getSelection() || { index: 0 }
      quillRef.current.insertEmbed(range.index, 'image', imageUrl)
      quillRef.current.setSelection(range.index + 1)
      setShowImageDialog(false)
      setImagePrompt('')
    } catch (err: any) {
      alert(`图片生成失败: ${err.message}`)
    } finally {
      setImageLoading(false)
    }
  }

  const handleExport = async (format: 'docx' | 'pdf' | 'md') => {
    if (!filename || !quillRef.current) return

    try {
      setError('')
      const html = quillRef.current.root.innerHTML
      await templateApi.saveContent(filename, html, false)
      const result = await templateApi.exportTemplate(filename, format)
      
      // Use relative path for download (will go through vite proxy in dev)
      const downloadUrl = `/api${result.download_url}`
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = ''
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      // Show success message after a short delay
      setTimeout(() => {
        alert(`导出成功，文件大小 ${(result.file_size / 1024).toFixed(1)} KB`)
      }, 100)
    } catch (err: any) {
      console.error('Export error:', err)
      alert(`导出失败: ${err.message}`)
    }
  }

  const handleDownloadOriginal = () => {
    if (!filename) return
    
    // Use relative path for download (will go through vite proxy in dev)
    const downloadUrl = `/api/templates/download/${encodeURIComponent(filename)}`
    window.open(downloadUrl, '_blank')
  }

  const applyAnalysisSkeleton = () => {
    if (!quillRef.current || !lessonAnalysis?.skeleton_markdown) return
    quillRef.current.root.innerHTML = renderMarkdown(lessonAnalysis.skeleton_markdown)
    quillRef.current.setSelection(0)
  }

  // Handle document import drawer
  const handleOpenDocumentImport = () => {
    setShowDocumentImportDrawer(true)
  }

  const handleCloseDocumentImport = () => {
    setShowDocumentImportDrawer(false)
  }

  const handleApplyDocumentToEditor = async (content: string) => {
    if (!quillRef.current) {
      console.error('Quill editor not initialized')
      alert('编辑器未初始化，请刷新页面重试')
      return
    }

    try {
      // Convert markdown to HTML
      const htmlContent = renderMarkdown(content)
      
      // Get current document length (Quill always has a trailing newline, so length is at least 1)
      const currentLength = quillRef.current.getLength()
      
      // Calculate insert position (before the trailing newline)
      const insertPosition = Math.max(0, currentLength - 1)
      
      // Add spacing if document is not empty
      if (insertPosition > 0) {
        quillRef.current.insertText(insertPosition, '\n\n', 'silent')
      }
      
      // Get the updated position after adding newlines
      const finalInsertPosition = quillRef.current.getLength() - 1
      
      // Insert the HTML content at the end
      quillRef.current.clipboard.dangerouslyPasteHTML(finalInsertPosition, htmlContent, 'silent')
      
      // Scroll to the newly inserted content
      const newLength = quillRef.current.getLength()
      quillRef.current.setSelection(newLength - 1, 0, 'silent')
      quillRef.current.scrollIntoView()
      
      console.log('Content applied to editor successfully')
    } catch (error) {
      console.error('Failed to apply content to editor:', error)
      alert('应用到编辑器失败，请重试')
    }
  }

  if (loading) {
    return <Loading message="正在加载模板编辑器..." />
  }

  if (error && !content) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-8">
        <div className="rounded-2xl border border-red-200 bg-red-50 p-6 text-center">
          <p className="mb-4 text-red-700">{error}</p>
          <Button onClick={() => navigate('/templates')}>返回列表</Button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#f6f9fc_0%,#eef3f8_100%)]">
      <div className="mx-auto max-w-[1520px] px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.4fr)_520px]">
          <section className="rounded-[12px] border border-slate-200 bg-white p-5 shadow-[0_24px_80px_rgba(15,23,42,0.08)] sm:p-6">
            <div className="mb-5 flex flex-col gap-4 border-b border-slate-200 pb-5 lg:flex-row lg:items-center lg:justify-between">
              <div>
                {/* <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700">教学设计正文</p> */}
                <h2 className="mt-2 text-2xl font-bold text-slate-900">{content?.filename}</h2>
                <div className="mt-3 flex flex-wrap gap-2">
                  {/* <div className="inline-flex rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">最终可导出内容</div>
                  <div className="inline-flex rounded-full bg-sky-50 px-3 py-1 text-xs font-medium text-sky-700">AI 生成后自动回写</div> */}
                </div>
              </div>
              <div className="flex gap-3">
                <Button variant="secondary" onClick={() => navigate('/templates')}>
                  返回列表
                </Button>
              </div>
            </div>

            <div className="relative">
              <div 
                ref={(el) => {
                  editorRef.current = el
                  if (el && !editorMounted) {
                    setEditorMounted(true)
                  }
                }} 
                style={{ minHeight: '780px' }} 
              />

              {showPopover && selectedText && (
                <div
                  ref={popoverRef}
                  className="absolute z-50 w-[420px] max-w-[calc(100%-32px)] rounded-3xl border border-slate-200 bg-white p-5 shadow-2xl"
                  style={{ top: `${popoverPosition.top}px`, left: `${popoverPosition.left}px` }}
                >
                  <button onClick={handleClosePopover} className="absolute right-4 top-4 text-slate-400 hover:text-slate-600">
                    <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>

                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700">AI 润色浮层</p>
                  <h3 className="mt-2 text-lg font-bold text-slate-900">针对当前选中文本给出修改建议</h3>
                  <div className="mt-4">
                    <label className="block text-xs font-semibold text-slate-500">已选内容</label>
                    <div className="mt-2 max-h-24 overflow-y-auto rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3 text-sm leading-6 text-slate-700">
                      {selectedText}
                    </div>
                  </div>
                  <div className="mt-4">
                    <label className="block text-xs font-semibold text-slate-500">修改要求</label>
                    <textarea
                      className="mt-2 w-full rounded-2xl border border-slate-300 px-3 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                      rows={4}
                      placeholder="例如：改成老师课堂导入时更顺口的话；增加一个学生容易理解的例子。"
                      value={aiInstruction}
                      onChange={(e) => setAiInstruction(e.target.value)}
                      autoFocus
                    />
                  </div>
                  <Button size="sm" className="mt-4 w-full" disabled={!aiInstruction.trim() || aiLoading} onClick={handleAIModify}>
                    {aiLoading ? 'AI 正在修改...' : '生成修改建议'}
                  </Button>
                  {aiResult && (
                    <div className="mt-4">
                      <label className="block text-xs font-semibold text-slate-500">建议版本</label>
                      <div className="mt-2 max-h-36 overflow-y-auto rounded-2xl border border-emerald-200 bg-emerald-50 px-3 py-3 text-sm leading-6 text-slate-700">
                        {aiResult}
                      </div>
                      <div className="mt-3 flex gap-2">
                        <Button size="sm" className="flex-1" onClick={handleApplyAI}>应用到正文</Button>
                        <Button size="sm" variant="secondary" className="flex-1" onClick={handleCancelAI}>先不用</Button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {showContextMenu && (
                <div
                  ref={contextMenuRef}
                  className="fixed z-50 min-w-[220px] rounded-2xl border border-slate-200 bg-white py-2 shadow-2xl"
                  style={{ top: `${contextMenuPosition.y}px`, left: `${contextMenuPosition.x}px` }}
                >
                  <button onClick={handleOpenTextEnhance} className="flex w-full items-center gap-2 px-4 py-2 text-left text-sm hover:bg-slate-100">
                    <span>✨</span>
                    <span>AI 润色这段文字</span>
                  </button>
                  <button onClick={handleOpenImageGeneration} className="flex w-full items-center gap-2 px-4 py-2 text-left text-sm hover:bg-slate-100">
                    <span>🎨</span>
                    <span>AI 生成配图</span>
                  </button>
                </div>
              )}

              {showImageDialog && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4">
                  <div className="w-full max-w-lg rounded-3xl bg-white p-6 shadow-2xl">
                    <h3 className="text-xl font-bold text-slate-900">AI 配图</h3>
                    <p className="mt-2 text-sm leading-7 text-slate-500">可以为当前模板段落补充插图。这里先保留现有演示逻辑。</p>
                    <div className="mt-4">
                      <label className="block text-sm font-semibold text-slate-700">图片说明</label>
                      <textarea
                        className="mt-2 w-full rounded-2xl border border-slate-300 px-3 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                        rows={4}
                        placeholder="例如：一个演示受力分析的课堂示意图"
                        value={imagePrompt}
                        onChange={(e) => setImagePrompt(e.target.value)}
                        autoFocus
                      />
                    </div>
                    <div className="mt-5 flex gap-3">
                      <Button className="flex-1" onClick={() => generateImage(imagePrompt)} disabled={!imagePrompt.trim() || imageLoading}>
                        {imageLoading ? '生成中...' : '生成图片'}
                      </Button>
                      <Button variant="secondary" className="flex-1" onClick={handleCloseImageDialog} disabled={imageLoading}>
                        取消
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <style>{`
              .ql-toolbar.ql-snow {
                border: 1px solid #e2e8f0;
                border-radius: 20px 20px 0 0;
                background: #f8fafc;
              }
              .ql-container.ql-snow {
                border: 1px solid #e2e8f0;
                border-top: none;
                border-radius: 0 0 24px 24px;
              }
              .ql-editor {
                font-family: "SimSun", "Songti SC", serif;
                font-size: 14px;
                line-height: 1.85;
                padding: 40px 60px;
                background: white;
              }
              .ql-editor h1 {
                font-size: 22px;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
                font-family: "SimHei", "PingFang SC", sans-serif;
              }
              .ql-editor h2 {
                font-size: 18px;
                font-weight: bold;
                margin: 16px 0 8px 0;
                font-family: "SimHei", "PingFang SC", sans-serif;
              }
              .ql-editor h3 {
                font-size: 16px;
                font-weight: bold;
                margin: 12px 0 6px 0;
              }
              .ql-editor p {
                margin: 8px 0;
                text-indent: 2em;
              }
              .ql-editor table {
                border-collapse: collapse;
                width: 100%;
                margin: 16px 0;
                font-size: 13px;
              }
              .ql-editor table td,
              .ql-editor table th {
                border: 1px solid #000;
                padding: 8px 12px;
                text-align: left;
              }
              .ql-editor table th {
                background-color: #f5f5f5;
                font-weight: bold;
              }
              .ql-editor ul,
              .ql-editor ol {
                padding-left: 2em;
                margin: 8px 0;
              }
              .ql-editor li {
                margin: 4px 0;
              }
              .ql-editor img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 16px auto;
              }
            `}</style>
          </section>

          <aside className="space-y-6 xl:sticky xl:top-6 xl:self-start">
            {/* 文档管理 */}
            <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
              <h3 className="text-sm font-semibold text-slate-900 mb-4">文档管理</h3>
              <div className="mt-4 flex gap-3">
                <Button variant="secondary" size="sm" onClick={handleOpenDocumentImport} className="flex-1">
                  <span className="flex items-center justify-center gap-1.5">
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    导入
                  </span>
                </Button>
                <div className="relative flex-1">
                  <Button size="sm" onClick={() => setShowExportMenu(!showExportMenu)} className="w-full">
                    <span className="flex items-center justify-center gap-1.5">
                      <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      导出
                    </span>
                  </Button>
                  {showExportMenu && (
                    <div className="absolute right-0 top-full mt-2 w-48 rounded-lg border border-slate-200 bg-white shadow-xl z-10">
                      <button
                        onClick={() => {
                          handleExport('docx')
                          setShowExportMenu(false)
                        }}
                        className="w-full px-4 py-2.5 text-left text-sm hover:bg-slate-50 flex items-center gap-2.5 border-b border-slate-100 rounded-t-lg transition-colors"
                      >
                        <svg className="h-4 w-4 text-sky-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <span className="font-medium text-slate-900">导出编辑后文档</span>
                      </button>
                      <button
                        onClick={() => {
                          handleDownloadOriginal()
                          setShowExportMenu(false)
                        }}
                        className="w-full px-4 py-2.5 text-left text-sm hover:bg-slate-50 flex items-center gap-2.5 rounded-b-lg transition-colors"
                      >
                        <svg className="h-4 w-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                        </svg>
                        <span className="font-medium text-slate-900">下载原始文档</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </section>

            <LessonDeepAnalysisPanel
              filename={filename}
              onApplySkeleton={applyAnalysisSkeleton}
              getEditorContent={() => quillRef.current?.root?.innerHTML ?? ''}
            />

            {/* 备课思路 - 横向流水线 */}
            <TeachingThoughtsTimeline 
              thoughts={teachingThoughts} 
              onStartAnalysis={handleStartAnalysis}
              isAnalyzing={isAnalyzing}
            />

            {/* 内容加强工具 */}
            <ContentEnhancer
              selectedText={selectedText}
              instruction={aiInstruction}
              onInstructionChange={setAiInstruction}
              onEnhance={handleAIModify}
              isLoading={aiLoading}
              result={aiResult}
              onApply={handleApplyAI}
              onCancel={handleCancelAI}
            />
          </aside>
        </div>
      </div>

      {/* Document Import Drawer */}
      <DocumentImportDrawer
        open={showDocumentImportDrawer}
        onClose={handleCloseDocumentImport}
        onApplyToEditor={handleApplyDocumentToEditor}
      />
    </div>
  )
}
